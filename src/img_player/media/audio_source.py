"""PyAV-based audio decoder paired with the video container.

A :class:`AudioSource` opens its own ``av.container`` (independent
from the video :class:`VideoSource` so seek state on the audio
stream doesn't fight with the video stream's cursor) and decodes the
first audio track of a file. Output is interleaved float32 in [-1, 1]
at the sounddevice device's sample rate — resampling is done by PyAV's
``AudioResampler`` so the device callback can write straight from
our buffers without per-callback resample overhead.

Like :class:`VideoSource` this is **synchronous** and not thread-safe.
The :class:`AudioOutput` (next slice) wraps it on a feeder thread; the
sounddevice callback never touches PyAV directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class AudioSource:
    """Open the audio track of a video file, decode frames at time T.

    ``output_sample_rate`` and ``output_channels`` define the format
    the consumer expects — PyAV resamples on read so the buffer we
    return is always in the device's native format. Mono ↔ stereo
    conversion is also handled by the resampler (mono source → both
    channels carry the same sample, stereo source mixed to mono via
    average).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        output_sample_rate: int = 48000,
        output_channels: int = 2,
    ) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)

        # Lazy import — pairs with how video_source.py loads PyAV.
        import av  # type: ignore[import-untyped]

        self._container = av.open(str(self._path))
        audio_streams = [s for s in self._container.streams if s.type == "audio"]
        if not audio_streams:
            self._container.close()
            raise ValueError(f"No audio stream in {self._path}")
        self._stream = audio_streams[0]
        self._stream.thread_type = "AUTO"

        self._output_sample_rate = int(output_sample_rate)
        self._output_channels = int(output_channels)
        # Resampler output: float **planar** (``fltp``). PyAV's
        # ``to_ndarray`` is consistent on planar — shape is always
        # ``(channels, samples)`` regardless of channel count, so we
        # can transpose to ``(samples, channels)`` (the layout
        # sounddevice + numpy stacking expect) without per-channel
        # bookkeeping. ``flt`` (interleaved) is a trap here: it
        # returns ``(1, samples*channels)`` and a naive transpose
        # gives ``(samples*channels, 1)`` — which sounds like the
        # signal pitched down an octave because every stereo pair
        # is read as two mono samples played at the same rate.
        layout = "mono" if output_channels == 1 else "stereo"
        self._resampler = av.AudioResampler(
            format="fltp",
            layout=layout,
            rate=output_sample_rate,
        )

        # Persistent decode generator + carry-over buffer. Decoder
        # frames don't align with arbitrary read sizes; we accumulate
        # whatever PyAV gives us and slice from the head.
        self._decoder = self._container.decode(self._stream)
        self._buffer = np.zeros((0, output_channels), dtype=np.float32)
        # Time (seconds) of the first sample currently in self._buffer.
        # Updated on every read / seek so callers can ask "where am I
        # in the stream?" — used by the AudioOutput sync logic.
        self._buffer_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sample_rate(self) -> int:
        return self._output_sample_rate

    @property
    def channels(self) -> int:
        return self._output_channels

    @property
    def native_sample_rate(self) -> int:
        return self._stream.codec_context.sample_rate or 0

    # ------------------------------------------------------------------
    # Read + seek
    # ------------------------------------------------------------------

    def seek(self, t_seconds: float) -> None:
        """Seek to the keyframe at or before ``t``. Drops the carry-over
        buffer; the next ``read`` block will start from at-or-before ``t``
        (audio is typically frame-aligned at ~20 ms boundaries, so the
        actual landing time may be a hair earlier than requested — the
        caller's sync logic handles fine alignment by skipping samples)."""
        target_us = int(max(0.0, t_seconds) * 1_000_000)
        self._container.seek(target_us, backward=True, any_frame=False)
        self._buffer = np.zeros((0, self._output_channels), dtype=np.float32)
        self._buffer_start_time = max(0.0, t_seconds)
        # Reset the resampler — internal state from before-seek samples
        # would otherwise pollute the first post-seek output block.
        # PyAV's API for resampler reset is to recreate it; cheap.
        import av  # type: ignore[import-untyped]
        layout = "mono" if self._output_channels == 1 else "stereo"
        self._resampler = av.AudioResampler(
            format="fltp",
            layout=layout,
            rate=self._output_sample_rate,
        )
        # Recreate the decode generator — same reason as in VideoSource.
        self._decoder = self._container.decode(self._stream)

    def read(self, n_samples: int) -> np.ndarray:
        """Return up to ``n_samples`` samples as ``(N, channels)`` float32.

        Pulls more from the decoder when the carry-over buffer is short.
        Returns an empty ``(0, channels)`` array on EOF — callers feed
        silence to keep the device callback flowing.
        """
        while self._buffer.shape[0] < n_samples:
            try:
                frame = next(self._decoder)
            except StopIteration:
                break
            except Exception as exc:  # PyAV EOFError variants
                if "End of file" in str(exc) or "EOF" in type(exc).__name__:
                    break
                raise
            # Each decoded frame can yield zero or more resampled
            # frames. With format=``fltp`` (planar), ``to_ndarray``
            # always returns ``(channels, samples)`` — transpose to
            # ``(samples, channels)`` so the consumer (sounddevice
            # callback) sees row-major sample frames.
            for resampled in self._resampler.resample(frame):
                arr = resampled.to_ndarray()
                if arr.ndim == 1:
                    # Mono planar can come out 1-D on some PyAV
                    # builds; promote to (1, N) before transposing.
                    arr = arr.reshape(1, -1)
                arr = arr.T  # (channels, samples) → (samples, channels)
                arr = arr.astype(np.float32, copy=False)
                # Defensive channel-count fix-up. The resampler
                # should already match output_channels; this catches
                # exotic source layouts (5.1 etc.) downmixed wrong.
                if arr.shape[1] != self._output_channels:
                    if arr.shape[1] == 1 and self._output_channels == 2:
                        arr = np.repeat(arr, 2, axis=1)
                    elif arr.shape[1] == 2 and self._output_channels == 1:
                        arr = arr.mean(axis=1, keepdims=True)
                self._buffer = np.concatenate([self._buffer, arr], axis=0)

        take = min(n_samples, self._buffer.shape[0])
        out = self._buffer[:take].copy()
        self._buffer = self._buffer[take:]
        # Advance the buffer-start time by however many samples we just
        # served, in seconds.
        self._buffer_start_time += take / self._output_sample_rate
        return out

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None  # type: ignore[assignment]
            self._buffer = np.zeros((0, self._output_channels), dtype=np.float32)

    def __enter__(self) -> AudioSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
