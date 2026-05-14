"""A QOpenGLWidget that displays a numpy frame through an OCIO GPU shader.

The viewport is fully independent of the player / cache / controller layers.
Consumers push frames via :meth:`set_frame` and tune the color pipeline via
:meth:`set_color_params`.

Two upload paths coexist behind a runtime flag:

* **Synchronous** (default at boot, mandatory on iGPU): a direct
  ``glTexSubImage2D`` blocks the main thread until the DMA finishes.
  Cheap, predictable, what we ship as the safe baseline.
* **PBO async** (gated on ``use_pbo`` from the perf tune): a 3-buffer
  ring orphans → maps → ``memcpy`` → unmaps → ``glTexSubImage2D(NULL)``,
  letting the driver dispatch the DMA out-of-band. The main thread
  pays only the memcpy + dispatch cost (~5-7 ms for a 4K float16
  frame), not the full DMA. Only worth it on a discrete GPU with
  PCIe DMA — see ``perf/PBO_NOTES.md`` for the iGPU regression.

The PBO path is enabled at runtime by ``app.py`` after the first
``initializeGL()`` reveals the real ``GL_RENDERER`` (the late-bind
flow described in spec §4 / slice 4).
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass

import numpy as np
from OpenGL import GL
from OpenGL.error import GLError
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from img_player.bench import recorder
from img_player.color.gpu_processor import ShaderBundle, Texture1D, Texture3D

log = logging.getLogger(__name__)


# ============================================================================
# PBO ring helper — see spec §4
# ============================================================================


_PBO_RING_SIZE = 3  # spec §4: 3 buffers, gives a 1-paint headroom over ping-pong


class _PboRing:
    """Three-PBO ring for asynchronous texture upload on discrete GPUs.

    Lazily allocated on the first call to :meth:`upload`. Re-allocates
    its three buffers when the requested upload size changes (e.g. a
    new sequence at a different resolution). Holds one ``glFenceSync``
    per buffer so we can attribute a *GPU-side wall-clock* to each
    upload — read at the *next* paint that touches the same slot.

    All GL calls happen on the GL thread (the Qt main thread for a
    ``QOpenGLWidget``). Callers must already have made the relevant
    texture current before invoking :meth:`upload`.

    Failure recovery: any GL exception inside :meth:`upload` is
    captured by the caller, which flips the viewport back to the
    synchronous path for the rest of the session. The ring itself
    doesn't try to retry — see spec §4 fallback rules.
    """

    def __init__(self) -> None:
        # Lazy: GL ids are zeroed until ensure_allocated() runs once.
        self._pbo_ids: list[int] = []
        # Parallel list of fences. ``None`` means "no upload yet on
        # this slot" (start of session) or "fence was already consumed
        # at the previous wrap-around".
        self._fences: list[object | None] = [None] * _PBO_RING_SIZE
        # ``perf_counter`` value at the moment we placed each fence.
        # Used to compute ``upload_gpu_us`` when the fence becomes
        # signalled three paints later.
        self._fence_dispatch_t: list[float] = [0.0] * _PBO_RING_SIZE
        self._capacity_bytes: int = 0
        # Wraps modulo _PBO_RING_SIZE on each upload.
        self._idx: int = 0

    # -- allocation ---------------------------------------------------------

    def ensure_allocated(self, nbytes: int) -> None:
        """Allocate (or reallocate) the three PBOs to hold ``nbytes`` each.

        On a 4K UHD float16 RGBA frame, ``nbytes`` is ~63 MB so the
        whole ring is ~189 MB on an 8 GB card — negligible. On a 6K
        sequence at the same format we'd want ~140 MB × 3 = 420 MB,
        still fine. We don't bother to track high-water marks: if a
        frame asks for less than current capacity, we keep the larger
        allocation rather than re-buffering on each shrink.
        """
        if not self._pbo_ids:
            # First call: generate the GL buffer ids.
            ids = GL.glGenBuffers(_PBO_RING_SIZE)
            # PyOpenGL returns either an int (n=1) or a numpy array.
            self._pbo_ids = list(ids) if hasattr(ids, "__iter__") else [int(ids)]
        if nbytes <= self._capacity_bytes:
            return
        for pbo in self._pbo_ids:
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
            GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        self._capacity_bytes = nbytes

    # -- upload -------------------------------------------------------------

    def upload(
        self,
        pixels: np.ndarray,
        *,
        gl_format: int,
        gl_type: int,
        width: int,
        height: int,
    ) -> tuple[float, float | None, bool]:
        """Run one async upload through the next ring slot.

        Returns ``(upload_cpu_us, upload_gpu_us, gpu_pending)``:

        * ``upload_cpu_us`` — wall-clock the main thread spent here
          (orphan + map + memcpy + unmap + dispatch). What costs us fps.
        * ``upload_gpu_us`` — wall-clock of the *previous* fence on
          this slot (= 3 paints ago) if it's now signalled, else
          ``None``. The first three uploads of a session always
          return ``None`` here because no fence has been placed yet
          on the slot we're about to overwrite.
        * ``gpu_pending`` — ``True`` when a fence existed but was
          not yet signalled by this paint (= the GPU can't keep up
          with the dispatch rate, a real warning sign on dGPU).
          The fence is kept around — we'll try again next time the
          ring wraps to this slot. With three buffers and < 60 fps
          this should never trigger on healthy hardware.

        The ``GL_MAP_UNSYNCHRONIZED_BIT`` is paired with ``glBufferData(NULL)``
        (orphaning) and ``GL_MAP_INVALIDATE_BUFFER_BIT``: together they
        tell the driver "I'm overwriting this whole buffer; don't sync
        on the previous DMA — give me a fresh region or copy-on-write
        the old one out". This is the only way to avoid the implicit
        stall the previous PBO experiment triggered (cf PBO_NOTES.md).
        """
        cpu_t0 = time.perf_counter()

        slot = self._idx
        # First, see if the previous fence on this slot is signalled —
        # if so, we can attribute a GPU-side timing to *that previous
        # upload*, not to ours.
        upload_gpu_us: float | None = None
        gpu_pending = False
        prev_fence = self._fences[slot]
        if prev_fence is not None:
            wait_result = GL.glClientWaitSync(prev_fence, 0, 0)
            if wait_result in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
                upload_gpu_us = (time.perf_counter() - self._fence_dispatch_t[slot]) * 1e6
                GL.glDeleteSync(prev_fence)
                self._fences[slot] = None
            elif wait_result == GL.GL_TIMEOUT_EXPIRED:
                # Fence still pending. We'll keep it and try again at
                # the next wrap. The upload below uses
                # GL_MAP_INVALIDATE_BUFFER_BIT, so the driver is free
                # to allocate a fresh backing store and let the in-
                # flight DMA finish on the old one — no stall.
                gpu_pending = True
            # GL_WAIT_FAILED is in theory possible but a healthy
            # context shouldn't hit it; treat it as "drop the fence".
            else:
                GL.glDeleteSync(prev_fence)
                self._fences[slot] = None

        nbytes = pixels.nbytes
        self.ensure_allocated(nbytes)

        pbo = self._pbo_ids[slot]
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
        # Orphan: tells the driver we don't care about the old contents,
        # which lets it return a fresh backing store immediately even
        # if a previous DMA is in flight.
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL.GL_STREAM_DRAW)

        # Map → memcpy → unmap. The three flags together are the
        # "fast path": invalidate the buffer (driver is free to give
        # us a fresh region), unsynchronized (don't wait on the
        # previous DMA — orphan + ring guarantee correctness), write-
        # only (skip the read-back path).
        ptr = GL.glMapBufferRange(
            GL.GL_PIXEL_UNPACK_BUFFER,
            0,
            nbytes,
            GL.GL_MAP_WRITE_BIT
            | GL.GL_MAP_INVALIDATE_BUFFER_BIT
            | GL.GL_MAP_UNSYNCHRONIZED_BIT,
        )
        if not ptr:
            # Map failed — bubble up to the caller, which will fall
            # back to the sync path for the rest of the session.
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            raise RuntimeError("glMapBufferRange returned NULL on PBO upload")

        # ctypes.memmove with the numpy data pointer is the cheapest
        # CPU-side memcpy we can do. ``np.ascontiguousarray`` upstream
        # guarantees the layout is dense.
        ctypes.memmove(int(ptr), pixels.ctypes.data, nbytes)
        GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)

        # Source = bound PBO, hence ``None`` for the data pointer.
        # The driver dispatches a DMA copy from the PBO to the texture
        # and returns immediately on a healthy dGPU.
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            gl_format,
            gl_type,
            None,
        )

        # Place a fence right after the dispatch so we can time the
        # DMA at the next wrap.
        new_fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        self._fences[slot] = new_fence
        self._fence_dispatch_t[slot] = time.perf_counter()

        # Restore the unbound state so a subsequent legacy (non-PBO)
        # call site doesn't accidentally read from our buffer.
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        self._idx = (self._idx + 1) % _PBO_RING_SIZE
        upload_cpu_us = (time.perf_counter() - cpu_t0) * 1e6
        return upload_cpu_us, upload_gpu_us, gpu_pending

    # -- teardown -----------------------------------------------------------

    def cleanup(self) -> None:
        """Release fences + PBOs. Idempotent."""
        for fence in self._fences:
            if fence is not None:
                try:
                    GL.glDeleteSync(fence)
                except (GLError, RuntimeError, TypeError):  # pragma: no cover — best effort
                    # GL context may already be torn down (RuntimeError),
                    # the fence pointer may be stale (GLError), or
                    # PyOpenGL may have unbound the call (TypeError on
                    # shutdown). Best-effort cleanup either way.
                    pass
        self._fences = [None] * _PBO_RING_SIZE
        if self._pbo_ids:
            try:
                GL.glDeleteBuffers(len(self._pbo_ids), self._pbo_ids)
            except (GLError, RuntimeError, TypeError):  # pragma: no cover
                pass
            self._pbo_ids = []
        self._capacity_bytes = 0
        self._idx = 0


@dataclass
class _ColorParams:
    exposure: float = 0.0
    gamma: float = 1.0
    # Channel mask — (R, G, B, A) booleans flattened to floats for the
    # GLSL uniform. All-on by default. When the user disables a
    # channel, we multiply that component by 0.0 in the fragment
    # shader (after OCIO) — cheap, no texture re-upload, no cache
    # invalidation.
    channel_mask: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    # Whether single-channel isolation should render as luminance
    # (Nuke behaviour) vs the original colour. True by default
    # because that's what artists expect when they click "R" alone.
    isolate_as_luminance: bool = True
    # Checker-pattern cell size in pixels. Drawn unconditionally
    # where the cached buffer's alpha is < 1 (per-layer alpha-
    # composite controls whether the buffer has alpha at all).
    checker_scale: float = 8.0
    # Transparency-background mode. 0 = checker (default), 1 = black,
    # 2 = mid-grey, 3 = white. The user picks via the BG button in
    # the menu-bar's right corner; the choice persists across launches
    # via :class:`Preferences`. Only matters where the cached buffer
    # has alpha < 1; opaque content paints over it regardless.
    transparency_bg_mode: int = 0


class GLViewport(QOpenGLWidget):  # type: ignore[misc]
    """OpenGL 3.3 Core viewport that color-transforms a float32 frame.

    Upload a frame with :meth:`set_frame` (shape HxWx3 or HxWx4, float32).
    Reconfigure the color pipeline with :meth:`set_color_params`.
    Click + drag horizontally to scrub through the timeline; the
    viewport emits :attr:`frame_requested` with the absolute target
    frame, same contract as the timeline scrubber so the controller
    handler is shared.
    """

    DEFAULT_BG = (0.0, 0.0, 0.0, 1.0)
    # Click-and-drag sensitivity: how many pixels of horizontal motion
    # advance the playhead by one frame. The user found 1 px / frame
    # too nervous on the AMD APU + standard mouse — 6 px / frame keeps
    # the gesture coarse enough that a flick of the wrist doesn't
    # overshoot, while still being responsive. The timeline scrubber
    # remains the absolute random-access path for big jumps.
    DRAG_PIXELS_PER_FRAME = 6
    # Wheel-zoom step. One notch of the wheel multiplies the zoom
    # factor by this; deltas accumulate in 120-unit increments per
    # notch (Qt convention). 1.10 gives ~10 % per notch, smooth.
    WHEEL_ZOOM_STEP = 1.10
    # Hard limits on the zoom factor. The combo box exposes 0.5 …
    # 2.0; the wheel can go a bit beyond for fine inspection.
    MIN_ZOOM = 0.10
    MAX_ZOOM = 8.0

    frame_requested = Signal(int)
    # Drag-scrub lifecycle inside the viewport (left-button horizontal
    # drag). Same signal contract as the Timeline's so the app can
    # route both through the same fast-seek toggle on video decoders.
    scrub_started = Signal()
    scrub_finished = Signal()
    # Emitted when the wheel changes the zoom — lets the transport's
    # zoom combo follow. Carries the new zoom factor (1.0 = 100%) or
    # ``None`` when fit-to-window mode is engaged.
    zoom_changed = Signal(object)
    # Emitted exactly once per session, on the first ``initializeGL``,
    # carrying the raw ``glGetString(GL_RENDERER)`` string. ``app.py``
    # uses it to re-run the perf tune now that the real GPU
    # classification is known (slice 4 late-bind, spec §4).
    gpu_renderer_detected = Signal(str)
    # Emitted whenever the image-to-widget transform changes — i.e.
    # zoom factor, pan offset, or widget size. The annotation overlay
    # listens to repaint its strokes in sync with the image. Plain
    # ``Signal()`` (no args): consumers ask for the current state via
    # :meth:`current_transform` / :meth:`image_size` when they need it.
    transform_changed = Signal()
    # Contact-sheet per-tile scrub. Carries ``(tile_idx, delta_frames)``
    # where ``tile_idx`` is the 0-based index of the tile under the
    # initial mouse-press (row-major: ``row * cols + col``) and
    # ``delta_frames`` is the cumulative horizontal drag distance
    # since the press, in frames. Emitted only when the viewport's
    # contact-sheet grid layout has been set via
    # :meth:`set_contact_sheet_grid` (which routes the press through
    # the per-tile path instead of the regular master-timeline scrub).
    contact_sheet_tile_scrub_requested = Signal(int, int)
    # Lifecycle bookends for the per-tile drag gesture. The app side
    # uses ``started(tile_idx)`` to snapshot the per-layer offset
    # *at press time* so move events can compute delta-from-anchor
    # rather than cumulative-across-gestures. ``finished`` lets the
    # app clear any per-gesture state.
    contact_sheet_tile_scrub_started = Signal(int)
    contact_sheet_tile_scrub_finished = Signal()

    # ------------------------------------------------------------------ Lifecycle

    def __init__(self, parent: QOpenGLWidget | None = None) -> None:
        super().__init__(parent)

        # Ask for GL 3.3 Core. On Windows this is what we get by default
        # with a modern driver, but being explicit makes the shader version
        # directive (`#version 410 core`) match the context.
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(0)
        self.setFormat(fmt)

        self._pending_bundle: ShaderBundle | None = None
        self._pending_frame: np.ndarray | None = None
        self._color_params = _ColorParams()

        # GL state (populated in initializeGL)
        self._program = 0
        self._vao = 0
        self._vbo = 0
        self._image_tex = 0
        self._lut_1d_ids: dict[str, int] = {}
        self._lut_3d_ids: dict[str, int] = {}
        # Cache of ``glGetUniformLocation`` lookups, keyed by uniform
        # name. paintGL used to call ``glGetUniformLocation`` 15+ times
        # every frame on names that don't change between paints — each
        # call is a GL round-trip. Reset to ``{}`` in ``_apply_bundle``
        # (the program id is the only thing that invalidates locations).
        self._uniform_locs: dict[str, int] = {}
        # Once a bundle has been applied we bind its LUT textures to
        # fixed units (1, 2, …) and push the sampler uniforms ONCE —
        # paintGL doesn't need to rebind every frame because GL texture
        # unit bindings are persistent. ``_apply_bundle`` flips this
        # to ``True``; paintGL short-circuits the LUT bind loop while
        # set.
        self._lut_bindings_set: bool = False
        # Number of LUT samplers (1D + 3D); paintGL needs it to know
        # which texture unit to use for ``uImageB``.
        self._compare_b_unit: int = 1
        self._image_size: tuple[int, int] = (0, 0)
        self._image_channels = 4
        # Tracks the most recent texture allocation so same-sized frames
        # can use the much cheaper glTexSubImage2D upload.
        self._tex_alloc: tuple[int, int, int] = (0, 0, 0)  # (w, h, channels)

        # Async PBO upload state (slice 4). ``None`` = synchronous
        # path active. Populated by ``set_pbo_enabled(True)``, which
        # ``app.py`` calls after the late-bind perf tune detects a
        # discrete GPU. Diagnostic: ``upload_gpu_us`` from the latest
        # PBO upload is captured here so ``paintGL`` can attach it
        # to the bench sample without keeping yet another field on
        # the recorder API.
        self._pbo_ring: _PboRing | None = None
        self._last_upload_gpu_us: float | None = None
        self._last_upload_gpu_pending: bool = False

        # Compare-overlay state (v1.2). When ``_compare_mode != 0``
        # the fragment shader picks pixels from ``uImage`` (= layer
        # A, uploaded via the existing ``set_frame``) AND ``uImageB``
        # (= layer B, uploaded via ``set_compare_b``) according to
        # the wipe / opacity mode + the ``_compare_seam`` value.
        # Mouse-drag updates ``_compare_seam`` only — no numpy
        # compose, no per-event GL upload, just a uniform write +
        # repaint. ``_pending_compare_b`` is the deferred upload
        # target (mirrors ``_pending_frame``); ``_compare_b_alloc``
        # tracks size for the texSubImage fast path.
        self._compare_tex_b: int = 0
        self._pending_compare_b: np.ndarray | None = None
        self._compare_b_alloc: tuple[int, int, int] = (0, 0, 0)
        # 0=off, 1=vert wipe, 2=horiz wipe, 3=opacity, 4=solo B.
        self._compare_mode: int = 0
        self._compare_seam: float = 0.5
        # Seam line tint alpha (0 = no line). Only meaningful in
        # wipe modes 1 and 2; the shader no-ops elsewhere.
        self._compare_seam_line_alpha: float = 0.55

        # Drag-to-scrub state. The viewport doesn't own the playback
        # state — the controller does — so we just remember "where the
        # drag started, and what frame we were on then" and emit
        # absolute target frames from there.
        self._current_frame = 0
        self._drag_base_frame: int | None = None
        self._drag_start_x: float = 0.0
        # Contact-sheet per-tile scrub state. ``_cs_grid`` is the
        # ``(cols, rows)`` layout the GL viewport thinks is active —
        # set via :meth:`set_contact_sheet_grid` whenever the app
        # toggles contact-sheet mode or changes the grid. ``None``
        # means "not in contact-sheet mode", and the drag-to-scrub
        # path falls back to its usual master-timeline behaviour.
        # ``_cs_drag_tile`` and ``_cs_drag_start_x`` track the
        # current per-tile drag gesture. The viewport doesn't own
        # the per-tile offsets themselves — it just emits delta
        # frames; the app side accumulates them into
        # ``ContactSheetState.per_layer_offsets``.
        self._cs_grid: tuple[int, int] | None = None
        self._cs_drag_tile: int | None = None
        self._cs_drag_start_x: float = 0.0
        # Navigable frame bounds — set by the app via
        # :meth:`set_navigable_range`. ``None`` = no clamp (used at
        # init and after :meth:`detach`-ing the sequence).
        self._nav_first: int | None = None
        self._nav_last: int | None = None
        # Hover cursor: the user gets a visual hint that the viewer is
        # scrubbable. Switching to OpenHandCursor / ClosedHandCursor on
        # press would also work — SizeHor is the more conventional
        # "you can drag me horizontally" affordance.
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        # Take keyboard focus on click — without this, clicking the
        # image leaves focus on whatever was previously focused (e.g.
        # the frame-display QLineEdit), which makes Space type a
        # literal space instead of triggering play/pause. ClickFocus
        # keeps Tab navigation skipping the viewport (the viewport has
        # no keyboard interactions of its own; the shortcuts live on
        # the main window).
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        # Zoom state. ``None`` = fit-to-window (the legacy behaviour);
        # any float = user-set zoom factor where 1.0 means "1 image
        # pixel = 1 widget pixel" (= "Actual size" / 100 %).
        self._zoom_factor: float | None = None

        # Pan offset in widget pixels (0, 0 = image centred). Applied
        # as a translation to the view matrix, scaled into NDC. Reset
        # on fit toggle since panning a fit-to-window image makes no
        # sense.
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        # Middle-button drag tracking: the position when the press
        # event fired and the pan offset at that moment, so move
        # events compute the new pan from a stable origin.
        self._pan_drag_start: tuple[float, float] | None = None
        self._pan_base_offset: tuple[float, float] = (0.0, 0.0)

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    # ------------------------------------------------------------------ Public API

    def set_frame(self, pixels: np.ndarray) -> None:
        """Upload a new frame. Non-blocking in the Qt main thread."""
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3 or HxWx4, got shape {pixels.shape}")
        if pixels.dtype not in (np.float16, np.float32):
            pixels = pixels.astype(np.float32, copy=False)
        self._pending_frame = np.ascontiguousarray(pixels)
        self.update()

    # ------------------------------------------------------------------ Compare overlay

    def set_compare_b(self, pixels: np.ndarray) -> None:
        """Upload the layer-B image used by the compare shader.

        Same constraints as :meth:`set_frame` (HxWx3 or HxWx4,
        float16/32). The actual GL upload runs deferred inside
        ``paintGL`` so callers don't need to be on the GL thread.

        Compare-mode keeps two textures alive at once: ``uImage``
        (= layer A, set via :meth:`set_frame`) and ``uImageB`` (this
        one). The fragment shader picks per-fragment between A and
        B based on :meth:`set_compare_state`'s mode + seam, so a
        seam drag is just a uniform update and a repaint.
        """
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3 or HxWx4, got shape {pixels.shape}")
        if pixels.dtype not in (np.float16, np.float32):
            pixels = pixels.astype(np.float32, copy=False)
        self._pending_compare_b = np.ascontiguousarray(pixels)
        self.update()

    def set_compare_state(
        self,
        mode: int,
        seam: float,
        seam_line_alpha: float = 0.55,
    ) -> None:
        """Configure the compare shader uniforms.

        ``mode`` is one of:
          0 — off (= ignore B, use A as before)
          1 — vertical wipe (left = A, right = B at ``seam``)
          2 — horizontal wipe (top = A, bottom = B at ``seam``)
          3 — opacity blend (linear mix(A, B, seam))
          4 — solo B (= show only B regardless of seam)
        ``seam`` is the wipe / opacity position in [0, 1].
        ``seam_line_alpha`` controls the visible seam stripe in
        wipe modes — 0 hides it.

        All three values are pure uniforms — no upload, no compose.
        Calling this on every mouse-move during a drag is safe.
        """
        self._compare_mode = int(mode)
        self._compare_seam = max(0.0, min(1.0, float(seam)))
        self._compare_seam_line_alpha = max(0.0, min(1.0, float(seam_line_alpha)))
        self.update()

    def clear_compare(self) -> None:
        """Disable compare overlay — sets mode to 0 + drops any
        pending B upload. Doesn't free the underlying GL texture
        (cheap to keep allocated for the next compare entry)."""
        self._compare_mode = 0
        self._pending_compare_b = None
        self.update()

    def clear_image(self) -> None:
        """Drop the current image without uploading a new one.

        Restores the "no sequence loaded" look the user sees at
        first launch — the viewport's ``paintGL`` early-returns
        when ``_image_size == (0, 0)``, leaving the GL clear color
        as the only thing on screen. Used by File → New so the
        viewport doesn't keep showing the last frame of the old
        sequence (and doesn't fake a "missing frame" placeholder
        that would suggest something went wrong).
        """
        self._pending_frame = None
        self._image_size = (0, 0)
        self.update()

    def set_color_params(
        self,
        bundle: ShaderBundle | None = None,
        *,
        exposure: float | None = None,
        gamma: float | None = None,
        channel_mask: tuple[float, float, float, float] | None = None,
        transparency_bg_mode: int | None = None,
    ) -> None:
        """Swap the OCIO shader bundle and/or tweak exposure / gamma /
        channel mask / transparency background. Any argument left
        ``None`` keeps its current value."""
        if bundle is not None:
            self._pending_bundle = bundle
        if exposure is not None:
            self._color_params.exposure = exposure
        if gamma is not None:
            self._color_params.gamma = max(0.01, gamma)
        if channel_mask is not None:
            self._color_params.channel_mask = channel_mask
        if transparency_bg_mode is not None:
            mode = int(transparency_bg_mode)
            if 0 <= mode <= 3:
                self._color_params.transparency_bg_mode = mode
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Tell the viewport which frame is currently displayed.

        Used as the base when the user starts a drag-scrub: the first
        emitted target is ``base + dx / DRAG_PIXELS_PER_FRAME``.
        """
        self._current_frame = frame

    def set_navigable_range(self, first: int, last: int) -> None:
        """Set the inclusive frame bounds for drag-scrub clamping.

        Without this the viewport happily emits frame numbers past
        the timeline's end while the user keeps dragging — the
        downstream cache miss falls back to the nearest cached
        neighbour, which makes adjacent frames flash on/off as the
        cursor pushes beyond the boundary. Clamping here at the
        source keeps the playhead pinned cleanly at the last (or
        first) frame regardless of how far the cursor wanders.

        Pass any reversed pair (e.g. ``(0, 0)``) to disable clamping.
        """
        if last < first:
            self._nav_first = None
            self._nav_last = None
        else:
            self._nav_first = int(first)
            self._nav_last = int(last)

    def _cs_tile_at(self, x: float, y: float) -> int | None:
        """Map widget-space cursor coords to a 0-based tile index.

        Returns ``None`` when contact-sheet mode is off or the
        cursor falls outside the widget bounds. Assumes the
        composite fills the widget — see the
        :meth:`set_contact_sheet_grid` docstring for why this is a
        safe simplification in practice.
        """
        if self._cs_grid is None:
            return None
        cols, rows = self._cs_grid
        w = max(1, self.width())
        h = max(1, self.height())
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        col = int(x / w * cols)
        row = int(y / h * rows)
        col = max(0, min(col, cols - 1))
        row = max(0, min(row, rows - 1))
        return row * cols + col

    def set_contact_sheet_grid(self, grid: tuple[int, int] | None) -> None:
        """Tell the viewport about the active contact-sheet grid.

        ``grid = (cols, rows)`` activates per-tile drag-to-scrub:
        the next ``mousePressEvent`` inside the widget maps to a
        tile index via the cursor's (x, y) position and the press
        starts a per-tile scrub gesture (emitting
        ``contact_sheet_tile_scrub_requested(tile_idx, delta_frames)``
        on each move). ``None`` reverts to the regular master-
        timeline scrub used outside contact-sheet mode.

        The viewport assumes the composite fills the widget. The
        smart-grid path in ``ImgPlayerApp._render_contact_sheet``
        picks the grid that matches the viewport aspect, so the
        small residual letterbox at the composite edges is at most
        a few percent and doesn't break tile mapping in practice.
        """
        if grid is None or grid[0] <= 0 or grid[1] <= 0:
            self._cs_grid = None
        else:
            self._cs_grid = (int(grid[0]), int(grid[1]))
        # Any in-progress drag should be discarded — switching mode
        # mid-gesture would emit the wrong signal type otherwise.
        self._cs_drag_tile = None
        self._drag_base_frame = None

    def set_pbo_enabled(self, enabled: bool) -> None:
        """Toggle the async PBO upload path on or off.

        Called by ``app.py`` after the first ``initializeGL`` reveals
        the real ``GL_RENDERER`` and the perf tune decides whether
        we're on a discrete GPU (``use_pbo=True``) or anywhere else
        (``use_pbo=False``). Idempotent — calling twice with the same
        value is a no-op.

        When toggling off, we release the ring's GL resources
        immediately so we don't keep ~190 MB of VRAM tied up. The
        synchronous path then resumes on the next ``paintGL``.
        """
        if enabled and self._pbo_ring is None:
            self._pbo_ring = _PboRing()
            log.info("[gl] async PBO upload path enabled (3-buffer ring)")
        elif not enabled and self._pbo_ring is not None:
            try:
                self._pbo_ring.cleanup()
            except Exception:  # pragma: no cover — best-effort GL teardown
                log.warning("[gl] PBO ring cleanup raised; continuing on sync path")
            self._pbo_ring = None
            log.info("[gl] async PBO upload path disabled, falling back to sync")

    def set_zoom(self, factor: float | None) -> None:
        """Set the zoom factor, or pass ``None`` for fit-to-window.

        ``factor`` is interpreted as image-pixels-per-widget-pixel:
        ``1.0`` is "Actual size" (100 %), ``0.5`` is half size,
        ``2.0`` is 2× zoom. Clamped to :attr:`MIN_ZOOM` /
        :attr:`MAX_ZOOM`. Switching back to fit (None) clears the
        pan offset because a centred fit is the only sensible
        baseline.

        The combo box in the transport bar drives this; the wheel
        also calls it with the new factor and emits ``zoom_changed``
        so the combo can stay in sync.
        """
        if factor is None:
            self._zoom_factor = None
            self._pan_x = 0.0
            self._pan_y = 0.0
        else:
            self._zoom_factor = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(factor)))
        self.update()
        self.transform_changed.emit()

    def current_transform(self) -> tuple[float, float, float]:
        """The image→widget transform parameters as ``(factor, pan_x, pan_y)``.

        ``factor`` is image-pixels-per-widget-pixel: in fit mode (when
        ``_zoom_factor is None``) we return the computed fit ratio, so
        callers always get a numeric factor. ``pan_x`` and ``pan_y``
        are widget-pixel offsets from the centred position (0, 0).

        Used by the annotation overlay to map image-space stroke coords
        to widget pixels for rendering, and the inverse for capture.
        """
        factor = (
            self._compute_fit_factor()
            if self._zoom_factor is None
            else self._zoom_factor
        )
        return (factor, self._pan_x, self._pan_y)

    def image_size(self) -> tuple[int, int]:
        """Image dimensions in pixels as ``(width, height)``.

        Returns ``(0, 0)`` until the first frame has been uploaded.
        The annotation overlay treats ``(0, 0)`` as "no transform yet,
        skip rendering" — so a still-loading viewport renders nothing
        instead of dividing by zero.
        """
        return self._image_size

    # ------------------------------------------------------------------ Mouse — drag-to-scrub

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Contact-sheet mode: identify which tile the cursor
            # sits in and start a per-tile scrub. The regular
            # master-timeline scrub stays disabled until the user
            # leaves contact sheet — global scrubbing lives on the
            # dedicated timeline widget below the viewer.
            if self._cs_grid is not None:
                tile_idx = self._cs_tile_at(
                    event.position().x(), event.position().y(),
                )
                if tile_idx is not None:
                    self._cs_drag_tile = tile_idx
                    self._cs_drag_start_x = event.position().x()
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                    self.contact_sheet_tile_scrub_started.emit(tile_idx)
                    event.accept()
                    return
            self._drag_base_frame = self._current_frame
            self._drag_start_x = event.position().x()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.scrub_started.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            # Middle-button drag = pan the image inside the viewport.
            # We capture the start position + the current pan offset
            # so subsequent move events compute deltas from a stable
            # origin (no drift if the user wiggles back and forth).
            self._pan_drag_start = (event.position().x(), event.position().y())
            self._pan_base_offset = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._cs_drag_tile is not None:
            # Per-tile scrub: horizontal pixels → frame delta. Same
            # ``DRAG_PIXELS_PER_FRAME`` ratio as the master scrub so
            # the gesture feels consistent.
            delta_px = event.position().x() - self._cs_drag_start_x
            delta_frames = int(delta_px / self.DRAG_PIXELS_PER_FRAME)
            self.contact_sheet_tile_scrub_requested.emit(
                self._cs_drag_tile, delta_frames,
            )
            event.accept()
            return
        if self._drag_base_frame is not None:
            delta_px = event.position().x() - self._drag_start_x
            delta_frames = int(delta_px / self.DRAG_PIXELS_PER_FRAME)
            target = self._drag_base_frame + delta_frames
            # Clamp to the navigable range so dragging past the
            # timeline's edges doesn't emit out-of-range frames —
            # those would round-trip through the cache as misses,
            # the fallback path would flash adjacent cached frames,
            # and the transport's frame readout would happily display
            # numbers past the actual end.
            if self._nav_first is not None and self._nav_last is not None:
                if target < self._nav_first:
                    target = self._nav_first
                elif target > self._nav_last:
                    target = self._nav_last
            if target != self._current_frame:
                # We don't update _current_frame ourselves — the controller
                # will push it back via set_current_frame when the seek
                # actually lands. That avoids drift if a seek is rejected
                # (e.g. clamped against in/out range).
                self.frame_requested.emit(target)
            event.accept()
            return
        if self._pan_drag_start is not None:
            dx = event.position().x() - self._pan_drag_start[0]
            dy = event.position().y() - self._pan_drag_start[1]
            self._pan_x = self._pan_base_offset[0] + dx
            self._pan_y = self._pan_base_offset[1] + dy
            self.update()
            self.transform_changed.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._cs_drag_tile is not None
        ):
            self._cs_drag_tile = None
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.contact_sheet_tile_scrub_finished.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_base_frame is not None:
            self._drag_base_frame = None
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.scrub_finished.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_drag_start is not None:
            self._pan_drag_start = None
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Reset any in-progress drag-scrub on a double-click so the
        scrub doesn't keep tracking the cursor while the user is
        gesturing for something else.

        Historical: this used to emit ``tile_isolate_requested`` for
        contact-sheet tile isolation, retired with the contact-sheet
        feature in v1.2.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            was_dragging = self._drag_base_frame is not None
            self._drag_base_frame = None
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            if was_dragging:
                # End the scrub gesture cleanly so video decoders flip
                # back to precise seeks — otherwise a double-click in
                # mid-drag would leave them stuck in fast-seek mode.
                self.scrub_finished.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Mouse-wheel zoom, anchored at the cursor. Scroll up zooms
        in, down zooms out.

        Qt's ``angleDelta()`` reports 120 units per "notch" of a
        traditional wheel. We use that as our quantum: each notch
        multiplies the zoom by :attr:`WHEEL_ZOOM_STEP`. Smooth-scroll
        wheels (mac trackpads) report fractional deltas — the same
        formula Just Works because we use a power.

        Starting point matters: in *fit* mode the image isn't at 100 %
        — it's at whatever ratio makes it fill the widget (often
        30-40 % for a 4K image in a 1280-wide window). Wheeling from
        Fit must continue *from that ratio*, not jump back to 100 %.
        We therefore base the multiplication on the current effective
        zoom, which is the fit factor when ``_zoom_factor is None``.

        Cursor anchor: when the factor changes, we adjust ``_pan_x`` /
        ``_pan_y`` so the image-space pixel under the cursor stays
        under the cursor — the only zoom feel that doesn't make the
        user immediately re-pan to find what they were looking at.
        Math lives in :func:`_anchored_pan_for_zoom`.
        """
        delta_steps = event.angleDelta().y() / 120.0
        if delta_steps == 0:
            super().wheelEvent(event)
            return
        base = (
            self._compute_fit_factor()
            if self._zoom_factor is None
            else self._zoom_factor
        )
        new_zoom = base * (self.WHEEL_ZOOM_STEP ** delta_steps)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))

        # Compute the pan that keeps the cursor's image-space pixel in
        # place. Note: in fit mode (_zoom_factor is None), the current
        # pan is forced to (0, 0) by set_zoom(None) — so old_pan here
        # is the right starting point regardless of mode.
        cursor = event.position()
        new_pan_x, new_pan_y = _anchored_pan_for_zoom(
            cursor_widget_xy=(cursor.x(), cursor.y()),
            widget_size=(self.width(), self.height()),
            old_factor=base,
            new_factor=new_zoom,
            old_pan=(self._pan_x, self._pan_y),
        )

        # Atomic update: zoom + pan in one shot, single repaint. We
        # don't go through set_zoom() because it doesn't know about
        # pan; calling it would issue a redundant update() and then
        # we'd update() again after the pan assignment.
        self._zoom_factor = new_zoom
        self._pan_x = new_pan_x
        self._pan_y = new_pan_y
        self.update()

        # Tell the rest of the UI (= the transport's zoom combo) so
        # it can reflect the new value without us setting it from
        # here. Also notify the annotation overlay that the transform
        # changed so it can repaint its strokes in sync.
        self.zoom_changed.emit(new_zoom)
        self.transform_changed.emit()
        event.accept()

    # ------------------------------------------------------------------ QOpenGLWidget overrides

    def initializeGL(self) -> None:
        renderer = GL.glGetString(GL.GL_RENDERER).decode("utf-8", errors="replace")
        log.info(
            "GL context: %s | %s | GLSL %s",
            GL.glGetString(GL.GL_VERSION).decode("utf-8", errors="replace"),
            renderer,
            GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode("utf-8", errors="replace"),
        )
        GL.glClearColor(*self.DEFAULT_BG)
        self._make_fullscreen_quad()
        self._make_image_texture()
        self._make_compare_texture()
        # Late-bind hook: now that the GL context exists, app.py can
        # re-run the perf tune with the real GPU classification (slice 4).
        # Connecting handlers were attached at app boot; the signal is
        # safe to emit even if no one's listening.
        self.gpu_renderer_detected.emit(renderer)

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, max(1, w), max(1, h))
        # In fit mode, the effective zoom factor is computed from the
        # widget size — so a resize changes the transform even if
        # _zoom_factor and pan are untouched. The annotation overlay
        # needs to know.
        self.transform_changed.emit()

    def paintGL(self) -> None:
        # Bench hook: time the whole paintGL body. Cheap (one branch, one
        # time.monotonic) when disabled.
        bench_enabled = recorder.is_enabled()
        paint_t0 = time.monotonic() if bench_enabled else 0.0
        upload_us = 0.0

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Apply deferred uploads inside a valid GL context.
        if self._pending_bundle is not None:
            self._apply_bundle(self._pending_bundle)
            self._pending_bundle = None

        had_upload = False
        if self._pending_frame is not None:
            if bench_enabled:
                up_t0 = time.monotonic()
                self._upload_image(self._pending_frame)
                upload_us = (time.monotonic() - up_t0) * 1e6
            else:
                self._upload_image(self._pending_frame)
            self._pending_frame = None
            had_upload = True

        if self._pending_compare_b is not None:
            self._upload_compare_b(self._pending_compare_b)
            self._pending_compare_b = None

        if self._program == 0 or self._image_size == (0, 0):
            if bench_enabled and had_upload:
                width, height = self._image_size
                recorder.record_paint(
                    displayed_frame=-1,
                    upload_us=upload_us,
                    paint_us=(time.monotonic() - paint_t0) * 1e6,
                    width=width, height=height, channels=self._image_channels,
                    upload_gpu_us=self._last_upload_gpu_us,
                    upload_gpu_pending=self._last_upload_gpu_pending,
                )
            return

        GL.glUseProgram(self._program)

        # Bind the input image to texture unit 0. The unit binding
        # persists across paints (texture units are global GL state),
        # but rebinding here is cheap (one driver call) and removes
        # us from worrying about other widgets that might steal
        # unit 0 in a future multi-viewport scenario.
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._image_tex)
        self._set_uniform_int("uImage", 0)

        # LUTs are bound once in ``_apply_bundle`` to fixed units
        # (1, 2, …) — paintGL doesn't need to walk the dicts every
        # frame. The compare-B texture sits at ``_compare_b_unit``
        # right after the last LUT.
        compare_unit = self._compare_b_unit
        GL.glActiveTexture(GL.GL_TEXTURE0 + compare_unit)
        GL.glBindTexture(
            GL.GL_TEXTURE_2D,
            self._compare_tex_b if self._compare_tex_b else self._image_tex,
        )
        # The sampler uniform's value (the texture unit number) only
        # changes when a new bundle is applied → set once there. We
        # still write the per-paint uniforms below.
        self._set_uniform_int("uCompareMode", self._compare_mode)
        self._set_uniform_float("uCompareSeam", self._compare_seam)
        self._set_uniform_float(
            "uCompareSeamLineAlpha", self._compare_seam_line_alpha,
        )

        self._set_uniform_float("uExposure", self._color_params.exposure)
        self._set_uniform_float("uGamma", self._color_params.gamma)
        # Per-channel mask + isolation flag (cf. fragment_template.glsl).
        mask = self._color_params.channel_mask
        self._set_uniform_vec4(
            "uChannelMask", mask[0], mask[1], mask[2], mask[3],
        )
        self._set_uniform_float(
            "uChannelIsolateLuminance",
            1.0 if self._color_params.isolate_as_luminance else 0.0,
        )
        self._set_uniform_float(
            "uCheckerScale", float(self._color_params.checker_scale),
        )
        self._set_uniform_int(
            "uTransparencyBgMode",
            int(self._color_params.transparency_bg_mode),
        )
        self._set_uniform_matrix4("uTransform", self._fit_matrix())

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        if bench_enabled:
            width, height = self._image_size
            recorder.record_paint(
                displayed_frame=-1,
                upload_us=upload_us,
                paint_us=(time.monotonic() - paint_t0) * 1e6,
                width=width, height=height, channels=self._image_channels,
                upload_gpu_us=self._last_upload_gpu_us,
                upload_gpu_pending=self._last_upload_gpu_pending,
            )

    # ------------------------------------------------------------------ Quad / texture helpers

    def _make_fullscreen_quad(self) -> None:
        # (x, y, u, v) — triangle strip: bottom-left, bottom-right, top-left, top-right
        # fmt: off
        vertices = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype=np.float32)
        # fmt: on
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        stride = 4 * 4  # 4 floats per vertex
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        GL.glBindVertexArray(0)

    def _make_image_texture(self) -> None:
        self._image_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._image_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    def _make_compare_texture(self) -> None:
        """Allocate the layer-B texture used by the compare shader.

        Same filter / wrap settings as the main image texture so the
        wipe doesn't show a seam-edge artefact between the two
        sources at the boundary. Storage is allocated lazily on the
        first ``_upload_compare_b`` call (we don't know the image
        size yet here).
        """
        self._compare_tex_b = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._compare_tex_b)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    def _upload_image(self, pixels: np.ndarray) -> None:
        """Push a frame to the GPU. Routes to sync or async PBO path.

        The sync branch below is **bit-for-bit identical** to the
        previous implementation — that's what protects the iGPU
        non-regression bench (slice 4 plan, bench C). The PBO
        branch is taken only when ``set_pbo_enabled(True)`` has been
        called earlier (post late-bind, on a discrete GPU).

        See also ``perf/PBO_NOTES.md`` for the experiment that
        revealed the iGPU regression and motivates keeping the sync
        path as the default.
        """
        height, width, channels = pixels.shape
        size_changed = (width, height) != self._image_size
        self._image_size = (width, height)
        self._image_channels = channels
        if size_changed:
            # Image dimensions feed into the fit-factor formula and the
            # image→widget centring offset — the overlay needs to know.
            self.transform_changed.emit()
        gl_format = GL.GL_RGBA if channels == 4 else GL.GL_RGB
        gl_type = GL.GL_HALF_FLOAT if pixels.dtype == np.float16 else GL.GL_FLOAT
        # Always use 16F internal storage — plenty of precision for display,
        # halves VRAM compared to RGBA32F.
        internal = GL.GL_RGBA16F if channels == 4 else GL.GL_RGB16F

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._image_tex)

        first_alloc = (width, height, channels) != self._tex_alloc

        if first_alloc:
            # Texture storage must be (re)allocated for a new size / format.
            # ``glTexImage2D`` is slow because it allocates on the GPU. We
            # always go through the synchronous path here regardless of
            # PBO state — the storage allocation step has no async
            # equivalent. Subsequent paints at the same size will pick
            # the PBO path if enabled.
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                internal,
                width,
                height,
                0,
                gl_format,
                gl_type,
                pixels,
            )
            self._tex_alloc = (width, height, channels)
            # Drop any stale GPU-side timing left over from the previous
            # texture size. Fences from the old layout aren't
            # interpretable anymore.
            self._last_upload_gpu_us = None
            self._last_upload_gpu_pending = False
            return

        # Same-sized frame: reuse the texture storage. Either go through
        # the PBO ring (if enabled) or fall through to the legacy sync
        # path. Any failure on the PBO path falls back to sync for the
        # rest of the session — never crashes the viewport.
        if self._pbo_ring is not None:
            try:
                _, gpu_us, gpu_pending = self._pbo_ring.upload(
                    pixels,
                    gl_format=gl_format,
                    gl_type=gl_type,
                    width=width,
                    height=height,
                )
                self._last_upload_gpu_us = gpu_us
                self._last_upload_gpu_pending = gpu_pending
                return
            except Exception as err:
                log.warning(
                    "[gl] PBO upload failed (%s); falling back to sync path for the rest of the session",
                    err,
                )
                # Tear down the ring and never touch it again this
                # session. The user can restart the app to retry.
                try:
                    self._pbo_ring.cleanup()
                except (GLError, RuntimeError, TypeError):  # pragma: no cover
                    # Cleanup is best-effort: the ring is already being
                    # abandoned, we just don't want a tertiary GL error
                    # to mask the original PBO failure that triggered
                    # this fallback.
                    pass
                self._pbo_ring = None
                self._last_upload_gpu_us = None
                self._last_upload_gpu_pending = False
                # Fall through to the sync path below.

        # Synchronous path — bit-for-bit identical to the pre-slice-4 code.
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            gl_format,
            gl_type,
            pixels,
        )
        self._last_upload_gpu_us = None
        self._last_upload_gpu_pending = False

    def _upload_compare_b(self, pixels: np.ndarray) -> None:
        """Push the layer-B image to ``_compare_tex_b``.

        Stripped-down version of :meth:`_upload_image` — same dtype
        and format handling, but no PBO ring (compare-mode entry is a
        once-per-frame_changed event, the bulk of paints during a
        seam drag are uniform-only and don't touch the texture). The
        first allocation goes through ``glTexImage2D``; subsequent
        same-sized uploads use ``glTexSubImage2D``.
        """
        height, width, channels = pixels.shape
        gl_format = GL.GL_RGBA if channels == 4 else GL.GL_RGB
        gl_type = GL.GL_HALF_FLOAT if pixels.dtype == np.float16 else GL.GL_FLOAT
        internal = GL.GL_RGBA16F if channels == 4 else GL.GL_RGB16F

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._compare_tex_b)
        first_alloc = (width, height, channels) != self._compare_b_alloc
        if first_alloc:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                internal,
                width,
                height,
                0,
                gl_format,
                gl_type,
                pixels,
            )
            self._compare_b_alloc = (width, height, channels)
            return
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0,
            width, height, gl_format, gl_type, pixels,
        )

    # ------------------------------------------------------------------ Shader / LUT setup

    def _apply_bundle(self, bundle: ShaderBundle) -> None:
        new_program = _compile_program(bundle.vertex_source, bundle.fragment_source)
        if new_program == 0:
            return
        if self._program:
            GL.glDeleteProgram(self._program)
        self._program = new_program
        # New program → invalidate every cached uniform location
        # (locations are program-scoped).
        self._uniform_locs.clear()
        self._lut_bindings_set = False

        self._release_luts()
        for tex1d in bundle.textures_1d:
            self._lut_1d_ids[tex1d.name] = _upload_lut_1d(tex1d)
        for tex3d in bundle.textures_3d:
            self._lut_3d_ids[tex3d.name] = _upload_lut_3d(tex3d)

        # Bind each LUT to a fixed texture unit and push the sampler
        # uniform value ONCE. GL texture-unit bindings persist across
        # paints, so the per-paint rebind loop in ``paintGL`` is dead
        # weight — eliminated below by the ``_lut_bindings_set`` flag.
        GL.glUseProgram(self._program)
        unit = 1
        for name, tex_id in self._lut_1d_ids.items():
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_1D, tex_id)
            self._set_uniform_int(name, unit)
            unit += 1
        for name, tex_id in self._lut_3d_ids.items():
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_3D, tex_id)
            self._set_uniform_int(name, unit)
            unit += 1
        # paintGL uses this unit for the compare-B texture. Stash
        # the slot so paintGL doesn't recompute it from
        # ``len(_lut_1d_ids) + len(_lut_3d_ids) + 1`` every frame.
        self._compare_b_unit = unit
        # The compare-B sampler value is fixed for the lifetime of
        # this program — set it once here too so paintGL only has
        # to rebind the texture object, not push the sampler uniform.
        self._set_uniform_int("uImageB", unit)
        self._lut_bindings_set = True

    def _release_luts(self) -> None:
        for tex_id in self._lut_1d_ids.values():
            GL.glDeleteTextures(1, [tex_id])
        for tex_id in self._lut_3d_ids.values():
            GL.glDeleteTextures(1, [tex_id])
        self._lut_1d_ids.clear()
        self._lut_3d_ids.clear()

    # ------------------------------------------------------------------ Uniform helpers

    def _uniform_location(self, name: str) -> int:
        """Cached ``glGetUniformLocation`` lookup.

        Locations are program-scoped — ``_apply_bundle`` clears the
        cache on every program swap. Returns ``-1`` (the GL sentinel
        meaning "no such active uniform") for unknown names; the
        helpers below all check for it.
        """
        loc = self._uniform_locs.get(name)
        if loc is None:
            loc = GL.glGetUniformLocation(self._program, name)
            self._uniform_locs[name] = loc
        return loc

    def _set_uniform_int(self, name: str, value: int) -> None:
        loc = self._uniform_location(name)
        if loc != -1:
            GL.glUniform1i(loc, value)

    def _set_uniform_float(self, name: str, value: float) -> None:
        loc = self._uniform_location(name)
        if loc != -1:
            GL.glUniform1f(loc, value)

    def _set_uniform_vec4(self, name: str, v0: float, v1: float, v2: float, v3: float) -> None:
        loc = self._uniform_location(name)
        if loc != -1:
            GL.glUniform4f(loc, v0, v1, v2, v3)

    def _set_uniform_matrix4(self, name: str, matrix: np.ndarray) -> None:
        loc = self._uniform_location(name)
        if loc != -1:
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, matrix)

    def _compute_fit_factor(self) -> float:
        """Return the zoom factor that makes the image fit the widget.

        Used both by the fit-mode view matrix and by the wheel-event
        handler when transitioning out of fit (so the first wheel
        notch zooms relative to "what's currently on screen", not
        from a hard-coded 100 %). Falls back to ``1.0`` when no
        image has been loaded yet.
        """
        win_w = max(1, self.width())
        win_h = max(1, self.height())
        img_w, img_h = self._image_size
        if img_w == 0 or img_h == 0:
            return 1.0
        return min(win_w / img_w, win_h / img_h)

    def fit_factor(self) -> float:
        """Public accessor for the fit-mode scale factor.

        The annotation overlay uses ``current_factor / fit_factor``
        as a "display scale" so brush widths stay invariant when
        the image size changes (e.g. switching to contact-sheet
        mode inflates the composite to N× the source) — but still
        grow when the user zooms in. Without this normalisation,
        strokes shrink as soon as the composite gets bigger than
        the source, which felt wrong because the source content
        on screen is still the same physical size.
        """
        return self._compute_fit_factor()

    def _fit_matrix(self) -> np.ndarray:
        """View matrix combining aspect-ratio fit + user zoom.

        Two regimes:

        * ``self._zoom_factor is None`` — fit mode: scale the
          fullscreen quad so the image fits the widget while
          preserving aspect ratio (letterbox / pillarbox).
        * ``self._zoom_factor`` is a float — user-set zoom, where the
          image is rendered at ``factor`` widget-pixels per
          image-pixel (centred). The image can extend beyond the
          widget at large zooms; that's the user's intent (inspect
          a region pixel-by-pixel).

        Both regimes share the same underlying formula
        ``s = (img_size × factor) / win_size`` — fit just substitutes
        ``factor = _compute_fit_factor()``.
        """
        win_w = max(1, self.width())
        win_h = max(1, self.height())
        img_w, img_h = self._image_size
        if img_w == 0 or img_h == 0:
            return np.identity(4, dtype=np.float32)

        factor = self._zoom_factor if self._zoom_factor is not None else self._compute_fit_factor()
        sx = (img_w * factor) / win_w
        sy = (img_h * factor) / win_h

        # Pan: convert widget-pixel offsets to NDC ([-1, 1] across
        # the widget = 2 NDC units total). We negate Y because Qt's
        # mouse coordinates have Y growing down while NDC has Y
        # growing up.
        tx = (self._pan_x / max(1, win_w)) * 2.0
        ty = -(self._pan_y / max(1, win_h)) * 2.0

        m = np.identity(4, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        # Translation lives on column 3 (rows 0, 1) for an OpenGL
        # column-major matrix. PySide6 uses row-major numpy arrays
        # for upload via glUniformMatrix4fv with transpose=GL_FALSE,
        # but we transposed the whole codebase to put translation
        # at m[3, 0/1] — keep that convention.
        m[3, 0] = tx
        m[3, 1] = ty
        return m


# ---------------------------------------------------------------------- Module-level helpers


def _anchored_pan_for_zoom(
    *,
    cursor_widget_xy: tuple[float, float],
    widget_size: tuple[int, int],
    old_factor: float,
    new_factor: float,
    old_pan: tuple[float, float],
) -> tuple[float, float]:
    """Pan offset that keeps the image-space pixel under the cursor in
    place when the zoom factor changes.

    The viewport's transform is::

        widget_xy = (win_size / 2) + pan + (image_xy - img_size/2) * factor

    For the cursor at ``c`` the image-space pixel under it is::

        i = ((c - win_size/2) - old_pan) / old_factor

    Solving for the new pan such that the same ``i`` lands at the same
    cursor position after we swap ``old_factor`` for ``new_factor``::

        new_pan = (c - win_size/2) - (c - win_size/2 - old_pan) * (new_factor / old_factor)

    Pure function: kept module-level so the math has unit-test coverage
    without spinning up a GL context. ``old_factor == 0`` is treated
    as a no-op (defensive — the viewport never produces zero, but
    callers passing arbitrary fits shouldn't divide-by-zero us).
    """
    if old_factor == 0.0:
        return old_pan
    cx, cy = cursor_widget_xy
    win_w, win_h = widget_size
    px1, py1 = old_pan
    u = cx - win_w / 2.0
    v = cy - win_h / 2.0
    ratio = new_factor / old_factor
    px2 = u - (u - px1) * ratio
    py2 = v - (v - py1) * ratio
    return (px2, py2)


def _compile_program(vertex_src: str, fragment_src: str) -> int:
    vs = _compile_shader(GL.GL_VERTEX_SHADER, vertex_src)
    fs = _compile_shader(GL.GL_FRAGMENT_SHADER, fragment_src)
    if vs == 0 or fs == 0:
        return 0
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        info = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        log.error("shader link failed: %s", info)
        GL.glDeleteProgram(program)
        return 0
    return int(program)


def _compile_shader(kind: int, source: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        info = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        stage = "vertex" if kind == GL.GL_VERTEX_SHADER else "fragment"
        log.error("%s shader compile failed: %s", stage, info)
        GL.glDeleteShader(shader)
        return 0
    return int(shader)


def _upload_lut_1d(tex: Texture1D) -> int:
    tex_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_1D, tex_id)
    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)

    data = np.ascontiguousarray(tex.values, dtype=np.float32)
    if tex.channel == "RED":
        GL.glTexImage1D(
            GL.GL_TEXTURE_1D, 0, GL.GL_R32F, data.shape[0], 0, GL.GL_RED, GL.GL_FLOAT, data
        )
    else:
        GL.glTexImage1D(
            GL.GL_TEXTURE_1D, 0, GL.GL_RGB32F, data.shape[0], 0, GL.GL_RGB, GL.GL_FLOAT, data
        )
    return int(tex_id)


def _upload_lut_3d(tex: Texture3D) -> int:
    tex_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, tex_id)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

    data = np.ascontiguousarray(tex.values, dtype=np.float32)
    GL.glTexImage3D(
        GL.GL_TEXTURE_3D,
        0,
        GL.GL_RGB32F,
        tex.edge,
        tex.edge,
        tex.edge,
        0,
        GL.GL_RGB,
        GL.GL_FLOAT,
        data,
    )
    return int(tex_id)
