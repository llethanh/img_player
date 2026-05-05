"""The :class:`Layer` dataclass — one sequence + its position/state.

Pure data, no Qt. Mutation goes through :class:`LayerStack` so
signal emission stays centralised; the dataclass itself is
intentionally mutable (default for ``@dataclass``) because the user
edits offset / trim / visibility live and a frozen-and-replace
pattern would force a stack-wide rebroadcast on each tweak.

Video-backed layers (mp4 / mov / etc.) reuse the same dataclass with
``video_metadata`` populated — the constructor :meth:`from_video`
synthesises a one-virtual-frame-per-video-frame ``SequenceInfo`` so
all the existing geometry methods (``covers``, ``source_frame_at``,
master-range math) keep working unchanged. The renderer dispatches
on ``video_metadata is not None`` to take a separate decode path
(see ``img_player.media.video_source``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from img_player.media.video_probe import VideoMetadata
from img_player.sequence.channels import ChannelSelection
from img_player.sequence.models import FrameInfo, SequenceInfo


def _new_id() -> str:
    """uuid4 hex, used as a stable handle the UI / cache key on."""
    return uuid.uuid4().hex


@dataclass
class Layer:
    """One sequence + its position/state on the master timeline.

    Time positioning model:

    * ``layer_in`` / ``layer_out`` — *source-frame* indices (within
      the underlying ``sequence``). Default to the sequence's full
      range. ``layer_out`` is **inclusive**.
    * ``offset`` — *master-frame* index where ``layer_in`` lands.
      ``offset = 0`` puts the layer's first trimmed frame at master
      frame 0. Negative offsets are allowed (the master timeline's
      lower bound shifts to accommodate).

    So a layer with ``sequence`` 1001-1100, ``layer_in=1010``,
    ``layer_out=1090``, ``offset=50`` covers master frames 50..130
    (= 81 trimmed frames placed starting at master 50).
    """

    sequence: SequenceInfo
    layer_in: int
    layer_out: int
    offset: int = 0
    visible: bool = True
    name: str = ""
    id: str = field(default_factory=_new_id)

    # ---- Per-layer state (populated as the user touches the layer) ----
    channel_selection: ChannelSelection | None = None
    channel_layout_mode: str = "Auto"
    channel_labels_visible: bool = True
    source_colorspace: str | None = None
    exposure: float = 0.0
    gamma: float = 1.0
    # Alpha compositing mode (per-layer "T toggle"). When ``True`` this
    # layer alpha-blends over what's beneath in the stack; when
    # ``False`` it acts as an opaque "floor" and masks every layer
    # below entirely. Default ``True`` because the common review case
    # is "see what my matte looks like" — opaque sources just have
    # alpha=1 everywhere, so the composite path no-ops on them and
    # there's no penalty.
    alpha_composite: bool = True
    # Alpha encoding convention of the source pixels. Auto-detected
    # from the file extension at ``from_sequence`` time (PNG / TGA /
    # JPG → straight, EXR / DPX / TIFF → premult); the user can
    # still flip the αS button to override per-layer if needed.
    alpha_is_straight: bool = False

    # Sidecar paths for annotations + comments. Resolved at
    # construction time relative to ``sequence.directory`` so a layer
    # carries enough info to save its notes without consulting the
    # owning app. ``None`` for layers built before the sidecar layout
    # was decided (defensive — every code path that creates layers
    # should populate these).
    annotations_path: Path | None = None
    comments_path: Path | None = None

    # Video-backed layers populate this from the PyAV probe; image
    # sequences leave it ``None``. Renderer code branches on
    # ``is_video`` to send video layers down the VideoSource decode
    # path instead of the per-frame OIIO file loader. The synthetic
    # ``sequence`` field still describes the virtual frame range so
    # all timeline / trim / offset arithmetic stays uniform.
    video_metadata: VideoMetadata | None = None
    # Per-layer audio policy for video layers. ``solo``: only this
    # layer's audio plays even when other video layers are visible;
    # ``mute``: never play this layer's audio. Defaults match what
    # the user expects on import — no solo, no mute, the active
    # layer policy decides which track is audible.
    audio_solo: bool = False
    audio_mute: bool = False
    # Volume in linear gain (1.0 = unity, 0.0 = silent). Applied to
    # the audio mix only; no effect on image-sequence layers.
    audio_gain: float = 1.0

    # Still-image layer: a single file held visible for ``still_hold_frames``
    # master frames. Built by :meth:`from_still` (drop / open of a single
    # image file). The underlying :attr:`sequence` carries exactly one
    # :class:`FrameInfo`; ``still_hold_frames`` controls how many master
    # frames the layer "covers" without growing the disk-side frame count.
    # The cache short-circuits decode for stills (one OIIO read shared
    # across all hold frames via ndarray ref aliasing).
    is_still: bool = False
    still_hold_frames: int = 1

    @classmethod
    def from_sequence(
        cls,
        sequence: SequenceInfo,
        offset: int = 0,
        name: str | None = None,
    ) -> Layer:
        """Build a fresh layer from a sequence, defaulting trim to
        the full source range and ``name`` to the display pattern.

        ``alpha_is_straight`` defaults to ``False`` (premultiplied —
        the VFX rendering standard). Per-format auto-detection used
        to live here but turned out to be wrong as often as right —
        EXR can be straight, PNG can come out of compositors with
        premult baked in. The user toggles αS per-layer when the
        default is off.
        """
        return cls(
            sequence=sequence,
            layer_in=sequence.first_frame,
            layer_out=sequence.last_frame,
            offset=offset,
            name=name or sequence.display_pattern(),
        )

    @classmethod
    def from_image(
        cls,
        sequence: SequenceInfo,
        *,
        default_still_hold: int = 100,
        offset: int = 0,
        name: str | None = None,
    ) -> Layer:
        """Smart image-layer factory: routes to still vs sequence.

        Builds a still layer when the scanner returned exactly one
        :class:`FrameInfo` (the user dropped a single image file —
        slate, lookdev ref, matte). Otherwise falls through to
        :meth:`from_sequence`. The ``default_still_hold`` argument is
        only consulted on the still path; callers compute it from the
        existing layer stack so a still dropped into a session full of
        100-frame sequences inherits a 100-frame hold (no manual config
        for the common slate case).
        """
        if sequence.frame_count == 1:
            return cls.from_still(
                sequence,
                hold_frames=default_still_hold,
                offset=offset,
                name=name,
            )
        return cls.from_sequence(sequence, offset=offset, name=name)

    @classmethod
    def from_video(
        cls,
        metadata: VideoMetadata,
        offset: int = 0,
        name: str | None = None,
    ) -> Layer:
        """Build a video-backed layer from a PyAV probe result.

        The ``SequenceInfo`` we attach is **synthetic** — one virtual
        :class:`FrameInfo` per video frame, all pointing at the same
        container file. This keeps every existing geometry method
        (``covers``, ``source_frame_at``, master-range math) working
        without a parallel "video Layer" class. The renderer detects
        :attr:`is_video` and routes pixel fetches through
        :class:`img_player.media.video_source.VideoSource` instead of
        the per-file OIIO loader.

        ``frame_count`` falls back to ``duration × fps`` when the
        container header didn't report a clean count (common with
        streamed mp4s). At minimum we synthesise one frame so the
        sequence invariant ("at least one frame") holds.
        """
        if not metadata.has_video:
            raise ValueError(f"Cannot build Layer from non-video file: {metadata.path}")
        # Resolve a frame count from whatever the probe gave us.
        # Container ``frame_count`` first; ``duration × fps`` second;
        # absolute fallback = 1 (so the dataclass invariant holds).
        n_frames = metadata.frame_count
        if n_frames is None and metadata.duration_seconds and metadata.fps:
            n_frames = max(1, int(round(
                metadata.duration_seconds * float(metadata.fps),
            )))
        if n_frames is None or n_frames < 1:
            n_frames = 1
        # mtime captured once — the per-frame mtime exists for the
        # sequence path's "did the file change" reload semantic. For a
        # single video file there's just one mtime; we copy it to
        # every virtual frame so callers can keep a uniform interface.
        try:
            mtime = metadata.path.stat().st_mtime
        except OSError:
            mtime = 0.0
        frames = tuple(
            FrameInfo(path=metadata.path, frame_number=i, mtime=mtime)
            for i in range(n_frames)
        )
        sequence = SequenceInfo(
            base_name=metadata.path.stem,
            extension=metadata.path.suffix,
            directory=metadata.path.parent,
            padding=0,
            frames=frames,
            fps_default=float(metadata.fps) if metadata.fps else 24.0,
            width=metadata.width,
            height=metadata.height,
        )
        return cls(
            sequence=sequence,
            layer_in=sequence.first_frame,
            layer_out=sequence.last_frame,
            offset=offset,
            name=name or metadata.path.name,
            video_metadata=metadata,
            # Video frames coming out of FFmpeg are already display-
            # range RGB (no alpha). The premult / straight distinction
            # only matters for compositing layers with alpha. Mark
            # straight so the alpha-composite path no-ops cleanly.
            alpha_is_straight=True,
        )

    @classmethod
    def from_still(
        cls,
        sequence: SequenceInfo,
        hold_frames: int,
        offset: int = 0,
        name: str | None = None,
    ) -> Layer:
        """Build a still-image layer from a single-file sequence.

        ``sequence`` must hold exactly one :class:`FrameInfo` — the
        scanner returns this when a single image file is dropped.
        ``hold_frames`` is the number of master frames the still
        will be visible for; a sensible default is the duration of
        the longest existing layer (or a project-wide constant when
        loaded into an empty session). The caller picks; the
        constructor only validates ``hold_frames >= 1``.

        Trim semantics: ``layer_in == layer_out == sequence.first_frame``
        always — the trim window collapses to the single source frame
        because there's only one to point at. The "logical" duration
        is carried by :attr:`still_hold_frames` instead, and
        :meth:`trim_length` / :meth:`source_frame_at` branch on
        :attr:`is_still` so the rest of the code keeps treating
        ``trim_length`` and ``master_end`` uniformly.
        """
        if sequence.frame_count != 1:
            raise ValueError(
                f"Layer.from_still requires a 1-frame SequenceInfo, "
                f"got {sequence.frame_count}",
            )
        hold = max(1, int(hold_frames))
        return cls(
            sequence=sequence,
            layer_in=sequence.first_frame,
            layer_out=sequence.first_frame,
            offset=offset,
            name=name or sequence.frames[0].path.name,
            is_still=True,
            still_hold_frames=hold,
            # Stills are usually JPG / PNG / single EXR refs — the
            # straight/premult flag follows whatever ``from_sequence``
            # would pick, but stills aren't typically composited as
            # alpha mattes. Default ``False`` (premult) matches the
            # rest of the app.
            alpha_is_straight=False,
            # Treat stills as opaque "floor" layers by default. This
            # routes their cache requests through the simple
            # single-decode path (``_decode_and_store``) instead of
            # the multi-layer composite path — the simple path
            # carries the still-fan-out optimisation, and a slate /
            # ref doesn't need alpha blending against the void
            # underneath. The user can still flip the T toggle in
            # the layer panel if they intentionally want to use a
            # still as an alpha matte over another layer.
            alpha_composite=False,
        )

    # ---- Derived properties --------------------------------------------

    @property
    def is_video(self) -> bool:
        """``True`` when this layer is backed by a video container.

        Used by renderer / cache code to dispatch to the VideoSource
        decode path instead of the per-frame OIIO file loader.
        """
        return self.video_metadata is not None

    @property
    def trim_length(self) -> int:
        """Number of *master* frames this layer covers.

        Sequences and video layers: ``layer_out - layer_in + 1`` (the
        trimmed source range, inclusive on both ends).

        Still layers: :attr:`still_hold_frames` — the trim window is
        collapsed to one source frame (``layer_in == layer_out``) and
        the user-facing duration is decoupled from disk content via
        the hold field.
        """
        if self.is_still:
            return max(1, self.still_hold_frames)
        return max(0, self.layer_out - self.layer_in + 1)

    @property
    def master_start(self) -> int:
        """First master frame this layer occupies."""
        return self.offset

    @property
    def master_end(self) -> int:
        """Last master frame this layer occupies (inclusive)."""
        return self.offset + self.trim_length - 1

    def covers(self, master_frame: int) -> bool:
        """``True`` if this layer has a frame at ``master_frame``.

        Independent of visibility — the LayerStack checks ``visible``
        separately so this method stays useful for "what could be
        shown if I unhid this layer" queries.
        """
        return self.master_start <= master_frame <= self.master_end

    def source_frame_at(self, master_frame: int) -> int:
        """Translate ``master_frame`` to this layer's source-frame.

        Caller must check :meth:`covers` first; out-of-range inputs
        return a frame number outside ``[layer_in, layer_out]``,
        which most downstream code treats as "no decode".

        Stills always return :attr:`layer_in` (the single source
        frame on disk) — every master frame in the hold range maps
        to the same file, which lets the cache alias one decoded
        ndarray across the entire hold without re-reading the file.
        """
        if self.is_still:
            return self.layer_in
        return self.layer_in + (master_frame - self.master_start)

    # ---- Validation -----------------------------------------------------

    def is_trim_valid(self) -> bool:
        """Whether ``layer_in`` / ``layer_out`` are within the
        underlying sequence's range and well-ordered. The model
        accepts inconsistent values (the UI sometimes mid-edits a
        spinbox); the renderer checks this before issuing a decode.

        Stills are always trim-valid: the trim window is collapsed
        to the single source frame and the duration is carried by
        ``still_hold_frames`` instead.
        """
        if self.is_still:
            return True
        return (
            self.sequence.first_frame <= self.layer_in <= self.layer_out
            <= self.sequence.last_frame
        )
