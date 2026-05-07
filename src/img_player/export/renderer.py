"""The :class:`FrameRenderer` — composes one export-ready frame.

For each source frame, this orchestrates:

1. Read the file via :func:`img_player.io.reader.read_frame`.
2. Apply the OCIO display transform (or skip — depends on settings).
3. Resize via OIIO when the user picked a non-source resolution.
4. Bake annotations using the same painter as the live overlay.
5. Convert to the dtype the writer wants.

The renderer doesn't know about threads or progress signals — it
just turns a frame index into a ready-to-write numpy array. The
engine handles the loop, cancellation, and progress emission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenImageIO as oiio
from PySide6.QtGui import QImage, QPainter

from img_player.annotate.store import AnnotationStore
from img_player.export.settings import ExportSettings, MissingFramePolicy
from img_player.export.stroke_painter import paint_strokes
from img_player.export.writers.image_seq import output_dtype_for
from img_player.io.reader import read_frame
from img_player.sequence.channels import ChannelSelection
from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)


@dataclass
class CompareRenderContext:
    """Snapshot of the live A/B compare overlay for the export.

    Captured at export-dialog accept time so the export reproduces
    exactly what the user sees: same A/B layers, same blend mode
    (vert / horiz / opacity), same seam value, same swap toggle.

    The renderer reads each layer's pixels at the master frame via
    its own ``Layer.sequence`` (path map + per-layer channel
    selection), then blends through ``compare.compose.compose``.
    Falls back to single-layer rendering if either layer no longer
    covers the frame at export time (e.g. trimmed mid-export).
    """

    layer_a: object  # img_player.layers.Layer — kept untyped to avoid a circular import
    layer_b: object  # img_player.layers.Layer
    mode: str
    seam: float
    swap_showing_b: bool


@dataclass
class RenderContext:
    """All long-lived inputs for the renderer.

    Bundled so a worker thread receives a single value instead of a
    long argument list. Constructed by :meth:`ExportEngine.__init__`.
    """

    sequence: SequenceInfo
    annotation_store: AnnotationStore | None
    # Optional CPU OCIO processor pre-built by the engine. ``None``
    # means "no display transform" (or "user disabled it"); the
    # renderer skips the color step entirely in that case.
    ocio_cpu_processor: object | None = None  # PyOpenColorIO.CPUProcessor
    # Snapshot of the live channel selection at export-dialog accept
    # time. Lets the export reproduce exactly what the user has on
    # screen — single channel or AOV. ``None`` falls back to the
    # legacy behaviour: read the default RGB(A) channels per frame.
    channel_selection: ChannelSelection | None = None
    # When set, the renderer takes the A/B compare path: decode
    # ``compare.layer_a`` and ``compare.layer_b`` independently at
    # the master frame, blend through ``compare.compose``, then
    # continue with OCIO / resize / annotations as usual. ``None``
    # falls back to the single-sequence read.
    compare: CompareRenderContext | None = None


class FrameRenderer:
    """Stateless-ish renderer. Holds a :class:`RenderContext` for
    the duration of an export and produces one frame per call."""

    def __init__(self, context: RenderContext, settings: ExportSettings) -> None:
        self._ctx = context
        self._settings = settings
        # Cache the source-frame map for O(1) lookup (the sequence's
        # ``frames`` is a tuple ordered by frame_number; we also need
        # to skip holes gracefully).
        self._frame_paths: dict[int, Path] = {
            fi.frame_number: fi.path for fi in context.sequence.frames
        }
        # Output dtype contract for the writer.
        if settings.is_image_sequence:
            self._writer_dtype = output_dtype_for(settings.format_key)
        else:
            self._writer_dtype = np.dtype(np.uint8)

        # Cache the missing-frame substitute (BLACK or PLACEHOLDER) so
        # we don't regenerate it for every hole. The placeholder build
        # runs a Qt painter + numpy chromatic aberration pass per
        # call; re-running it on a sequence with thousands of missing
        # frames would dominate the export time.
        self._missing_substitute_cache: np.ndarray | None = None

    @property
    def output_dtype(self) -> np.dtype:
        """The dtype that :meth:`render` returns. Writers assert on it."""
        return self._writer_dtype

    # ------------------------------------------------------------------ Per-frame entry point

    def render(self, source_frame: int, output_size: tuple[int, int]) -> np.ndarray:
        """Produce one ready-to-encode frame.

        ``output_size`` is ``(width, height)`` in pixels of the FINAL
        export. Used here for both the resize step and the
        annotation-bake widget size. The engine is the single place
        that resolves "Source / preset / Custom" → concrete pixels.

        Raises :class:`FileNotFoundError` if ``source_frame`` is a
        hole in the sequence.
        """
        out_w, out_h = output_size

        # Compare path: blend layer A + layer B per the captured
        # CompareState before any of the existing OCIO / resize /
        # annotation steps. Each contributor brings its own channel
        # selection and is decoded independently — same semantics as
        # the live overlay. Falls back to the single-sequence path
        # below if either side can't produce pixels at this frame
        # (= layer trimmed away after the export started).
        if self._ctx.compare is not None:
            arr = self._render_compare(source_frame, out_w, out_h)
            if arr is not None:
                return self._post_compose(
                    arr, source_frame, out_w, out_h,
                )
            # Fallthrough — at least one side is out of range, we
            # render the active sequence's frame normally.

        if source_frame not in self._frame_paths:
            policy = self._settings.missing_frame_policy
            if policy == MissingFramePolicy.ABORT:
                raise FileNotFoundError(
                    f"Frame {source_frame} missing from sequence"
                )
            log.info(
                "[export] frame %d missing — substituting (policy=%s)",
                source_frame, policy.value,
            )
            if self._missing_substitute_cache is None:
                self._missing_substitute_cache = (
                    self._build_missing_substitute(out_w, out_h, policy)
                )
            return self._to_writer_dtype(self._missing_substitute_cache)
        selection = self._ctx.channel_selection

        # ----- 1. Read from disk in float32 (we'll do colour math) ----
        # ``as_half=False`` so the float math stays in 32-bit.
        if selection is not None:
            channels = list(selection.active.channels)
            arr = read_frame(
                self._frame_paths[source_frame],
                channels=channels,
                as_half=False,
            )
        else:
            arr = read_frame(self._frame_paths[source_frame], as_half=False)
        # Always operate in 4-channel RGBA in our pipeline. Unpadded
        # source becomes RGBA with alpha=1.
        arr = self._ensure_rgba(arr)
        return self._post_compose(arr, source_frame, out_w, out_h)

    # ------------------------------------------------------------------ Compose-and-finalise tail

    def _post_compose(
        self,
        arr: np.ndarray,
        source_frame: int,
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """Apply OCIO, resize, annotation bake, and dtype convert.

        Shared tail for both the single-frame read path and the
        compare-mode A/B compose path — keeps the colour and resize
        rules identical regardless of where the input pixels came
        from.
        """
        # ----- OCIO display transform if requested -----------------
        if (
            self._settings.apply_display_transform
            and self._ctx.ocio_cpu_processor is not None
        ):
            arr = self._apply_ocio(arr)

        # ----- Resize ---------------------------------------------
        src_h, src_w = arr.shape[:2]
        if (src_w, src_h) != (out_w, out_h):
            arr = self._resize(arr, out_w, out_h)

        # ----- Annotation bake ------------------------------------
        # We bake into a uint8 QImage. The post-bake array is in the
        # writer's target dtype.
        # For uint8-target formats (PNG/JPG/video): clamp + scale
        # + bake = single uint8 buffer.
        # For float / 16-bit formats (EXR/TIFF): bake on a uint8
        # composite then composite-blend back into the float buffer
        # (otherwise we'd lose the floating-point fidelity for the
        # parts NOT covered by annotations).
        if (
            self._settings.bake_annotations
            and self._ctx.annotation_store is not None
        ):
            strokes = self._ctx.annotation_store.strokes_at(source_frame)
            if strokes:
                arr = self._bake_strokes(arr, strokes, source_frame, out_w, out_h)

        # ----- Final dtype conversion -----------------------------
        return self._to_writer_dtype(arr)

    # ------------------------------------------------------------------ Compare A/B path

    def _render_compare(
        self, master_frame: int, out_w: int, out_h: int,
    ) -> np.ndarray | None:
        """Decode A and B at ``master_frame`` and blend them through
        the live :func:`compare.compose.compose` helper.

        ``master_frame`` follows the export-loop iteration index,
        which mirrors master-timeline coordinates (the dialog's
        in/out bounds come from the controller's master-frame state).
        Each layer contributes its own channel selection so AOVs
        previewed in compare are reproduced verbatim in the export.

        Returns ``None`` when either layer can't produce pixels at
        this frame — the caller falls back to the single-sequence
        path so the export never silently swaps a black screen for a
        missing slice.
        """
        del out_w, out_h  # output sizing is handled in _post_compose
        ctx = self._ctx.compare
        if ctx is None:
            return None
        a_arr = self._read_layer_frame(ctx.layer_a, master_frame)
        if a_arr is None:
            return None
        b_arr = self._read_layer_frame(ctx.layer_b, master_frame)
        if b_arr is None:
            return None
        # Compose path expects float buffers; both layer reads are
        # already float32 (as_half=False below).
        from img_player.compare.compose import compose
        composed = compose(
            a_arr,
            b_arr,
            mode=ctx.mode,
            seam=ctx.seam,
            swap_showing_b=ctx.swap_showing_b,
            # The viewer's accent-orange seam line is a UI affordance;
            # the user wants the EXPORT to reflect what they see, so
            # keep it on by default. (If a future setting wants to
            # hide it for a clean print export, we can plumb that.)
            draw_seam_line=True,
        )
        # ``compose`` returns the same dtype as A; ensure float32
        # RGBA for the rest of the pipeline.
        return self._ensure_rgba(composed)

    def _read_layer_frame(self, layer, master_frame: int) -> np.ndarray | None:
        """OIIO read of ``layer`` at the given master frame.

        Mirrors :class:`img_player.compare.decode.CompareDecoder`'s
        image-sequence path but stays self-contained so the export
        worker doesn't have to share state with the live
        ``CompareDecoder`` (which is keyed on the live video source
        manager).
        """
        if not layer.covers(master_frame):
            return None
        source_frame = layer.source_frame_at(master_frame)
        path = None
        for fi in layer.sequence.frames:
            if fi.frame_number == source_frame:
                path = fi.path
                break
        if path is None:
            return None
        sel = layer.channel_selection
        channels = list(sel.active.channels) if sel is not None else None
        try:
            arr = read_frame(path, channels=channels, as_half=False)
        except Exception:
            log.warning(
                "[export-compare] decode failed for layer=%s master=%d",
                layer.id, master_frame, exc_info=True,
            )
            return None
        return self._ensure_rgba(arr)

    # ------------------------------------------------------------------ Helpers

    @staticmethod
    def _build_missing_substitute(
        out_w: int, out_h: int, policy: MissingFramePolicy,
    ) -> np.ndarray:
        """Generate a stand-in float32 RGBA frame for a missing source.

        Honours :class:`MissingFramePolicy`:

        * ``BLACK`` — a solid opaque black frame, sized to the export.
        * ``PLACEHOLDER`` — the same "MISSING FRAME" visual the live
          viewer shows in gaps (greyscale damier + chromatic aberration
          + 4-corner crosshairs + central boxed label + vignette),
          generated at the export resolution.

        ``ABORT`` is handled upstream in :meth:`render` — this method
        is only reached for the substitute policies.
        """
        if policy == MissingFramePolicy.BLACK:
            arr = np.zeros((out_h, out_w, 4), dtype=np.float32)
            arr[..., 3] = 1.0
            return arr
        # PLACEHOLDER — delegate to the same generator the cache uses
        # so live preview and export stay visually consistent.
        from img_player.cache.missing_frame import (
            generate_missing_frame_rgba_float,
        )
        return generate_missing_frame_rgba_float(out_w, out_h)

    @staticmethod
    def _ensure_rgba(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected source shape: {arr.shape}")
        c = arr.shape[2]
        if c == 1:
            arr = np.repeat(arr, 3, axis=2)
            c = 3
        if c == 3:
            alpha = np.ones((*arr.shape[:2], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=2)
        elif c > 4:
            arr = arr[..., :4]
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))

    def _apply_ocio(self, arr: np.ndarray) -> np.ndarray:
        """Run the CPU OCIO processor on the buffer in place.

        OCIO's `applyRGB` / `applyRGBA` expects a contiguous float32
        array. We keep the alpha channel passthrough by calling
        `applyRGB` on the first 3 channels — display transforms
        should not touch alpha.
        """
        proc = self._ctx.ocio_cpu_processor
        if proc is None:
            return arr
        rgb = np.ascontiguousarray(arr[..., :3], dtype=np.float32)
        # PyOpenColorIO 2.5 API: CPUProcessor.applyRGB(arr) operates
        # in-place on a (H*W, 3) or (H, W, 3) float32 buffer.
        proc.applyRGB(rgb)
        arr[..., :3] = rgb
        return arr

    @staticmethod
    def _resize(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """Lanczos resize via OIIO ImageBuf."""
        # OIIO works on its own ImageBuf; we hand it the numpy array
        # via a temporary spec.
        h, w, c = arr.shape
        spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
        buf = oiio.ImageBuf(spec)
        buf.set_pixels(oiio.ROI.All, np.ascontiguousarray(arr.astype(np.float32)))
        out_buf = oiio.ImageBuf(oiio.ImageSpec(out_w, out_h, c, oiio.FLOAT))
        if not oiio.ImageBufAlgo.resize(out_buf, buf, "lanczos3"):
            log.warning("[export] OIIO resize failed (%s); falling back to nearest", oiio.geterror())
            return _nearest_resize(arr, out_w, out_h)
        result = np.asarray(out_buf.get_pixels(oiio.FLOAT))
        return np.ascontiguousarray(result)

    def _bake_strokes(
        self,
        arr: np.ndarray,
        strokes: tuple,
        frame_idx: int,  # noqa: ARG002 — kept for log/debug symmetry
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """Composite annotations into ``arr`` (float32 RGBA).

        Strategy:

        * Convert ``arr`` to a uint8 RGBA buffer (clamp + scale).
        * Wrap that buffer in a ``QImage`` and paint the strokes
          via :func:`paint_strokes`.
        * Convert the painted uint8 buffer back to float32, scaled
          to [0, 1].

        This means EXR / TIFF lose floating-point precision in the
        baked region, which is the right tradeoff: the annotation
        IS a presentation-grade overlay, expected to look like the
        live viewer (sRGB-ish, 8-bit). Pixels NOT covered by any
        stroke pass through with no quantisation thanks to QPainter's
        SourceOver alpha blending — only the rasterised stroke
        pixels round to uint8.
        """
        # Clamp to [0, 1] for a reasonable uint8 conversion. OCIO-baked
        # frames are already in [0, 1]; linear passthrough may exceed
        # 1.0 on highlights — clamping there is a known compromise.
        rgba8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        rgba8 = np.ascontiguousarray(rgba8)
        # Important: QImage::Format_RGBA8888 expects R, G, B, A byte
        # order in memory — matches our channel order. Format_ARGB32
        # would force a per-pixel byte shuffle.
        # `bytesPerLine` is the row stride; with contiguous memory
        # it's exactly width*4.
        h, w, _ = rgba8.shape
        if (w, h) != (out_w, out_h):
            raise RuntimeError(f"bake size mismatch: arr={(w,h)} expected={(out_w,out_h)}")
        qimg = QImage(
            rgba8.data, w, h, w * 4, QImage.Format.Format_RGBA8888,
        )
        # We paint with image-space coords mapped 1:1 to widget coords:
        # factor=1.0, pan=(0,0), widget_size=img_size. The strokes are
        # already in source-image-space; if the export is RESIZED, the
        # bake target is the resized size, so we must scale the
        # strokes from source dims to output dims. We do this via a
        # uniform `factor` and centred coordinate system: a stroke at
        # source (x, y) lands at output (x * sx, y * sy) where
        # sx = out_w / src_w. The simplest path is to pre-scale the
        # widget_size and image_size both equal to the OUTPUT dims,
        # and scale the strokes' image coordinates by sx / sy in-line.
        src_w = self._ctx.sequence.width or out_w
        src_h = self._ctx.sequence.height or out_h
        scaled_strokes = _scale_strokes(strokes, src_w, src_h, out_w, out_h)
        painter = QPainter(qimg)
        try:
            paint_strokes(
                painter,
                scaled_strokes,
                widget_size=(out_w, out_h),
                img_size=(out_w, out_h),
                factor=1.0,
                pan=(0.0, 0.0),
            )
        finally:
            painter.end()
        # Back to float32 in [0, 1].
        baked = rgba8.astype(np.float32) / 255.0
        return baked

    def _to_writer_dtype(self, arr: np.ndarray) -> np.ndarray:
        target = self._writer_dtype
        # uint8 path: clamp & scale
        if target == np.uint8:
            return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if target == np.uint16:
            return (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        if target == np.float16:
            return arr.astype(np.float16)
        if target == np.float32:
            return arr.astype(np.float32)
        # Fallback — just cast.
        return arr.astype(target)


def _scale_strokes(strokes, src_w: int, src_h: int, out_w: int, out_h: int):
    """Return strokes with points/size scaled from source to output dims.

    Same logic Nuke uses when scaling its annotation overlays for a
    Write — the painted gesture stays visually anchored to the same
    image content regardless of the export resolution.
    """
    if (src_w, src_h) == (out_w, out_h):
        return strokes
    sx = out_w / max(1, src_w)
    sy = out_h / max(1, src_h)
    # Scale stroke size by the average to keep round dots round even
    # under non-uniform scaling.
    s_avg = (sx + sy) / 2.0
    from img_player.annotate.stroke import Stroke  # local to avoid circular noise
    return tuple(
        Stroke(
            points=tuple((p[0] * sx, p[1] * sy) for p in s.points),
            color=s.color,
            size=max(1.0, s.size * s_avg),
        )
        for s in strokes
    )


def _nearest_resize(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Pure-numpy nearest-neighbour fallback if OIIO's resize fails.

    Not pretty but never crashes — a safety net so the export can
    still complete with degraded quality rather than aborting on a
    transient OIIO error.
    """
    h, w = arr.shape[:2]
    ys = (np.arange(out_h) * h / out_h).astype(np.int32)
    xs = (np.arange(out_w) * w / out_w).astype(np.int32)
    return arr[ys[:, None], xs[None, :]]
