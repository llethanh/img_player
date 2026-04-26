"""A QOpenGLWidget that displays a numpy frame through an OCIO GPU shader.

The viewport is fully independent of the player / cache / controller layers.
Consumers push frames via :meth:`set_frame` and tune the color pipeline via
:meth:`set_color_params`.
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from img_player.bench import recorder
from img_player.color.gpu_processor import ShaderBundle, Texture1D, Texture3D

log = logging.getLogger(__name__)


@dataclass
class _ColorParams:
    exposure: float = 0.0
    gamma: float = 1.0


class GLViewport(QOpenGLWidget):  # type: ignore[misc]
    """OpenGL 3.3 Core viewport that color-transforms a float32 frame.

    Upload a frame with :meth:`set_frame` (shape HxWx3 or HxWx4, float32).
    Reconfigure the color pipeline with :meth:`set_color_params`.
    Click + drag horizontally to scrub through the timeline; the
    viewport emits :attr:`frame_requested` with the absolute target
    frame, same contract as the timeline scrubber so the controller
    handler is shared.
    """

    DEFAULT_BG = (0.05, 0.05, 0.05, 1.0)
    # Click-and-drag sensitivity: 1 pixel of horizontal motion =
    # 1 frame of timeline scrub. Feels right on HD/4K screens for the
    # typical sequence sizes (90 - 1000 frames). Bigger sequences end
    # up needing a couple of drags to traverse — that's intentional;
    # the timeline scrubber is the absolute random-access path.
    DRAG_PIXELS_PER_FRAME = 1

    frame_requested = Signal(int)

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
        self._image_size: tuple[int, int] = (0, 0)
        self._image_channels = 4
        # Tracks the most recent texture allocation so same-sized frames
        # can use the much cheaper glTexSubImage2D upload.
        self._tex_alloc: tuple[int, int, int] = (0, 0, 0)  # (w, h, channels)

        # Drag-to-scrub state. The viewport doesn't own the playback
        # state — the controller does — so we just remember "where the
        # drag started, and what frame we were on then" and emit
        # absolute target frames from there.
        self._current_frame = 0
        self._drag_base_frame: int | None = None
        self._drag_start_x: float = 0.0
        # Hover cursor: the user gets a visual hint that the viewer is
        # scrubbable. Switching to OpenHandCursor / ClosedHandCursor on
        # press would also work — SizeHor is the more conventional
        # "you can drag me horizontally" affordance.
        self.setCursor(Qt.CursorShape.SizeHorCursor)

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

    def set_color_params(
        self,
        bundle: ShaderBundle | None = None,
        *,
        exposure: float | None = None,
        gamma: float | None = None,
    ) -> None:
        """Swap the OCIO shader bundle and/or tweak exposure/gamma."""
        if bundle is not None:
            self._pending_bundle = bundle
        if exposure is not None:
            self._color_params.exposure = exposure
        if gamma is not None:
            self._color_params.gamma = max(0.01, gamma)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Tell the viewport which frame is currently displayed.

        Used as the base when the user starts a drag-scrub: the first
        emitted target is ``base + dx / DRAG_PIXELS_PER_FRAME``.
        """
        self._current_frame = frame

    # ------------------------------------------------------------------ Mouse — drag-to-scrub

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_base_frame = self._current_frame
            self._drag_start_x = event.position().x()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_base_frame is None:
            super().mouseMoveEvent(event)
            return
        delta_px = event.position().x() - self._drag_start_x
        delta_frames = int(delta_px / self.DRAG_PIXELS_PER_FRAME)
        target = self._drag_base_frame + delta_frames
        if target != self._current_frame:
            # We don't update _current_frame ourselves — the controller
            # will push it back via set_current_frame when the seek
            # actually lands. That avoids drift if a seek is rejected
            # (e.g. clamped against in/out range).
            self.frame_requested.emit(target)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_base_frame is not None:
            self._drag_base_frame = None
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ QOpenGLWidget overrides

    def initializeGL(self) -> None:
        log.info(
            "GL context: %s | %s | GLSL %s",
            GL.glGetString(GL.GL_VERSION).decode("utf-8", errors="replace"),
            GL.glGetString(GL.GL_RENDERER).decode("utf-8", errors="replace"),
            GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode("utf-8", errors="replace"),
        )
        GL.glClearColor(*self.DEFAULT_BG)
        self._make_fullscreen_quad()
        self._make_image_texture()

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, max(1, w), max(1, h))

    def paintGL(self) -> None:
        # Bench hook: time the whole paintGL body. Cheap (one branch, one
        # time.monotonic) when disabled.
        bench_enabled = recorder.is_enabled()
        paint_t0 = time.monotonic() if bench_enabled else 0.0
        upload_us = 0.0
        displayed_frame = -1  # filled in below if we actually upload this paint

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

        if self._program == 0 or self._image_size == (0, 0):
            if bench_enabled and had_upload:
                width, height = self._image_size
                recorder.record_paint(
                    displayed_frame=-1,
                    upload_us=upload_us,
                    paint_us=(time.monotonic() - paint_t0) * 1e6,
                    width=width, height=height, channels=self._image_channels,
                )
            return

        GL.glUseProgram(self._program)

        # Bind the input image to texture unit 0.
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._image_tex)
        self._set_uniform_int("uImage", 0)

        # Bind OCIO LUTs to texture units starting at 1.
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

        self._set_uniform_float("uExposure", self._color_params.exposure)
        self._set_uniform_float("uGamma", self._color_params.gamma)
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

    def _upload_image(self, pixels: np.ndarray) -> None:
        # NOTE on PBO async upload (tested 2026-04-26, see perf/PBO_NOTES.md):
        # On a unified-memory iGPU (AMD Radeon 780M / Ryzen APU) the PBO
        # ping-pong path measured *worse* than the direct glTexSubImage2D
        # below — there's no PCIe DMA to pipeline, so the extra
        # memcpy + glMapBufferRange overhead just stacks on top. Sticking
        # with the direct path until we benchmark on a discrete GPU.
        height, width, channels = pixels.shape
        self._image_size = (width, height)
        self._image_channels = channels
        gl_format = GL.GL_RGBA if channels == 4 else GL.GL_RGB
        gl_type = GL.GL_HALF_FLOAT if pixels.dtype == np.float16 else GL.GL_FLOAT
        # Always use 16F internal storage — plenty of precision for display,
        # halves VRAM compared to RGBA32F.
        internal = GL.GL_RGBA16F if channels == 4 else GL.GL_RGB16F

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._image_tex)

        if (width, height, channels) != self._tex_alloc:
            # Texture storage must be (re)allocated for a new size / format.
            # glTexImage2D is slow because it allocates on the GPU.
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
        else:
            # Same-sized frame: reuse the texture storage and only push the
            # pixels.
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

    # ------------------------------------------------------------------ Shader / LUT setup

    def _apply_bundle(self, bundle: ShaderBundle) -> None:
        new_program = _compile_program(bundle.vertex_source, bundle.fragment_source)
        if new_program == 0:
            return
        if self._program:
            GL.glDeleteProgram(self._program)
        self._program = new_program

        self._release_luts()
        for tex1d in bundle.textures_1d:
            self._lut_1d_ids[tex1d.name] = _upload_lut_1d(tex1d)
        for tex3d in bundle.textures_3d:
            self._lut_3d_ids[tex3d.name] = _upload_lut_3d(tex3d)

    def _release_luts(self) -> None:
        for tex_id in self._lut_1d_ids.values():
            GL.glDeleteTextures(1, [tex_id])
        for tex_id in self._lut_3d_ids.values():
            GL.glDeleteTextures(1, [tex_id])
        self._lut_1d_ids.clear()
        self._lut_3d_ids.clear()

    # ------------------------------------------------------------------ Uniform helpers

    def _set_uniform_int(self, name: str, value: int) -> None:
        loc = GL.glGetUniformLocation(self._program, name)
        if loc != -1:
            GL.glUniform1i(loc, value)

    def _set_uniform_float(self, name: str, value: float) -> None:
        loc = GL.glGetUniformLocation(self._program, name)
        if loc != -1:
            GL.glUniform1f(loc, value)

    def _set_uniform_matrix4(self, name: str, matrix: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(self._program, name)
        if loc != -1:
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, matrix)

    def _fit_matrix(self) -> np.ndarray:
        """Letterbox matrix: scale the fullscreen quad so the image aspect
        ratio is preserved inside the widget's aspect ratio."""
        win_w = max(1, self.width())
        win_h = max(1, self.height())
        img_w, img_h = self._image_size
        if img_w == 0 or img_h == 0:
            return np.identity(4, dtype=np.float32)

        win_aspect = win_w / win_h
        img_aspect = img_w / img_h
        sx, sy = 1.0, 1.0
        if img_aspect > win_aspect:
            # image is wider relative to window: fit width, shrink height
            sy = win_aspect / img_aspect
        else:
            sx = img_aspect / win_aspect

        m = np.identity(4, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        return m


# ---------------------------------------------------------------------- Module-level helpers


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
