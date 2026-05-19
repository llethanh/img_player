"""Nuke-inspired custom-painted timeline with tick marks, in/out markers, cache bar."""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from img_player.ui.theme import C, F, G

DisplayMode = Literal["frames", "tc"]


def frame_to_timecode(frame: int, fps: float) -> str:
    """Non-drop-frame timecode ``HH:MM:SS:FF`` for a frame index."""
    fps_int = max(1, round(fps))
    frame = max(0, frame)
    total_seconds = frame // fps_int
    ff = frame % fps_int
    hours = total_seconds // 3600
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{ff:02d}"


class Timeline(QWidget):  # type: ignore[misc]
    """Custom timeline: ticks + labels + range + in/out + playhead triangle + cache bar.

    Scrubbing emits ``frame_requested`` continuously during the drag.
    """

    frame_requested = Signal(int)
    # Scrub gesture lifecycle. Emitted at the start and end of a left-
    # button drag on the timeline (i.e. when ``_scrubbing`` flips).
    # Used by the app to toggle video decoders into a keyframe-only
    # fast-seek mode during drag — long-GOP H.264 / H.265 seeks
    # are expensive enough that a per-tick precise decode lags.
    scrub_started = Signal()
    scrub_finished = Signal()
    # Ctrl + click on the timeline: places (and lets the user drag)
    # an in or out point depending on which side of the playhead
    # the click landed. Same gesture as Nuke's timeline. Emits the
    # frame number under the cursor; the controller decides what
    # to do with the previously-set in/out value.
    set_in_at_requested = Signal(int)
    set_out_at_requested = Signal(int)

    # ---- Geometry (from charter) -------------------------------------
    # MARGIN_X used to be 8 to give edge labels overflow room. With
    # the :class:`MasterTimelinePanel` composite the timeline widget
    # shares its x-range with the layer bars (whose ``PADDING_X`` is
    # 0) — any non-zero MARGIN_X would push the timeline's drawable
    # half a slot off vs the bars, recreating the misalignment we
    # just fixed. Edge labels are clamped to widget bounds in
    # ``_draw_ticks_and_labels`` instead of relying on overflow space.
    MARGIN_X      = 0
    LABEL_H       = 14
    TICK_TOP      = 14
    TICK_MINOR_H  = 5
    TICK_MAJOR_H  = 9
    RANGE_Y       = 28
    RANGE_H       = 3
    CACHE_TOP     = 41
    CACHE_H       = 6
    TOTAL_H       = G.TIMELINE_H  # 52

    # Minimum visible frame count when no real sequence is loaded
    # (empty app) OR when the loaded sequence is degenerate
    # (single still: last == first). Keeps a 100-frame skeleton
    # painted so the user always has a visual timeline track —
    # otherwise the widget collapses to a blank rectangle and the
    # affordance disappears.
    _DEFAULT_LENGTH = 100

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(self.TOTAL_H)
        self.setFixedHeight(self.TOTAL_H)
        self.setMouseTracking(True)

        # Empty-state default range: 100 frames worth of empty
        # ticks. Without this, an app freshly opened (or one that
        # holds only a single still / no footage at all) would
        # paint a blank rectangle where the timeline track should
        # be — the user has no visual anchor that "a timeline
        # exists here". ``set_range`` widens to the same 100-frame
        # minimum when the underlying sequence is degenerate
        # (single still: last == first; new session: 0, 0).
        self._first = 0
        self._last = self._DEFAULT_LENGTH - 1
        self._current = 0
        self._in_frame: int | None = None
        self._out_frame: int | None = None
        self._fps = 24.0
        self._display_mode: DisplayMode = "frames"
        self._cached_frames: frozenset[int] = frozenset()
        # Missing source frames — drawn red on the cache bar so the
        # user spots holes in the sequence at a glance. The cache
        # still marks them as "playable" (it serves a checkerboard
        # placeholder) so playback continues; only the colour
        # changes here.
        self._missing_frames: frozenset[int] = frozenset()
        # Master frames not covered by any visible layer (multi-layer
        # gaps). Painted distinctly from cached / missing so the user
        # reads them as "no content" rather than "slow decode".
        self._gap_frames: frozenset[int] = frozenset()
        # Master frames whose blob exists in the on-disk cache (= a
        # previous session decoded them). Drawn as a dimmer orange
        # behind the bright "in RAM" cached run so the user sees at a
        # glance which frames are "warm on disk, not yet promoted to
        # RAM" — usually means an instant scrub away from a hit. Set
        # difference vs ``_cached_frames`` removes overlap so the
        # dim-orange only shows where the bright orange isn't already.
        self._disk_available_frames: frozenset[int] = frozenset()
        self._annotated_frames: frozenset[int] = frozenset()
        self._commented_frames: frozenset[int] = frozenset()
        self._scrubbing = False
        # Minimal mode: skip every decoration (cache bar, annotation
        # / comment markers, in-out flags, tick labels) and render
        # just the range track + playhead. Used by ``MainWindow``'s
        # fullscreen mode where the user wants a clean review
        # surface without the review-tool chrome on top.
        self._minimal_mode: bool = False
        # Dimmed mode: paints a semi-transparent grey film over the
        # whole widget and ignores mouse input. Used by contact-sheet
        # mode where the per-tile scrub-drag on the viewport is the
        # primary interaction and the global timeline is showing the
        # master playhead as a *read-only* reference (per-tile offsets
        # stack on top of the master frame, so changing master via the
        # timeline mid-contact-sheet was a foot-gun for users learning
        # the mode). The "set / unset in/out via Ctrl-click" gesture
        # is blocked too — same reason: keep the contact-sheet UI focused.
        self._dimmed: bool = False
        # Drag mode for Ctrl-modified mouse interactions:
        #   None         → normal scrub
        #   "drag_in"    → moving the in-point
        #   "drag_out"   → moving the out-point
        # Decided at press time based on which side of the playhead
        # the click landed; held until mouseRelease.
        self._drag_mode: str | None = None

        self._label_font: QFont = F.mono(F.SIZE_XS)

    # ------------------------------------------------------------------ Public API

    def set_range(self, first: int, last: int) -> None:
        """Set the visible frame range. Degenerate ranges (empty
        session, single still) are pre-widened upstream so the
        timeline always receives at least ``_DEFAULT_LENGTH``
        frames. The local ``max(first, last)`` here is just a
        defensive guard against swapped arguments — without it,
        ``_last < _first`` would slip through into the paintEvent
        early-return and the skeleton would disappear.

        Widening lives in :meth:`LayerPanel.broad_master_range` so
        the same range is pushed to both the timeline and the
        layer-bar rows below — without that shared axis, the two
        playhead cursors would land at different x-positions
        whenever the underlying sequence is shorter than the
        widened skeleton.
        """
        self._first = first
        self._last = max(first, last)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        if frame == self._current:
            return
        self._current = frame
        self.update()

    def set_in_out(self, in_frame: int | None, out_frame: int | None) -> None:
        self._in_frame = in_frame
        self._out_frame = out_frame
        self.update()

    def set_fps(self, fps: float) -> None:
        if abs(fps - self._fps) < 1e-6:
            return
        self._fps = max(0.1, fps)
        if self._display_mode == "tc":
            self.update()

    def set_display_mode(self, mode: DisplayMode) -> None:
        if mode == self._display_mode:
            return
        self._display_mode = mode
        self.update()

    def set_cached_frames(self, frames: frozenset[int]) -> None:
        if frames == self._cached_frames:
            return
        self._cached_frames = frames
        self.update()

    def set_missing_frames(self, frames: frozenset[int]) -> None:
        """Frames whose source file was missing / unreadable. Painted
        red on the cache bar (overrides the orange "cached" run for
        the same slots so the user sees holes immediately)."""
        if frames == self._missing_frames:
            return
        self._missing_frames = frames
        self.update()

    def set_gap_frames(self, frames: frozenset[int]) -> None:
        """Master frames not covered by any visible layer. Painted in
        a neutral light grey on the cache bar — distinct from
        ``cached`` (orange) and ``missing`` (red) so the user reads
        them as "expected void" rather than a decode failure."""
        if frames == self._gap_frames:
            return
        self._gap_frames = frames
        self.update()

    def set_disk_available_frames(self, frames: frozenset[int]) -> None:
        """Frames known to live in the on-disk cache. Painted with a
        dim orange wash behind the bright "in RAM" cached run, so
        the user can tell at a glance that a session reopen is
        already warm on disk before any frame is actually scrubbed
        back into RAM. Idempotent: same set → no repaint."""
        if frames == self._disk_available_frames:
            return
        self._disk_available_frames = frames
        self.update()

    def set_annotated_frames(self, frames: frozenset[int]) -> None:
        """Frames with at least one annotation stroke. Drives the
        accent-coloured triangle markers above the cache bar.

        Idempotent: same set → no repaint.
        """
        if frames == self._annotated_frames:
            return
        self._annotated_frames = frames
        self.update()

    def set_dimmed(self, enabled: bool) -> None:
        """Toggle the read-only dim-overlay used when contact-sheet
        mode is active. Idempotent — same value → no repaint."""
        flag = bool(enabled)
        if flag == self._dimmed:
            return
        self._dimmed = flag
        # When dimming on, also drop any in-progress drag / scrub so
        # the next click in normal mode starts clean.
        if flag:
            self._scrubbing = False
            self._drag_mode = None
        self.update()

    def set_minimal_mode(self, enabled: bool) -> None:
        """Toggle the stripped-down rendering used by the fullscreen
        bottom bar — track + playhead only, no labels / cache bar /
        annotation markers / etc."""
        if bool(enabled) == self._minimal_mode:
            return
        self._minimal_mode = bool(enabled)
        self.update()

    def set_commented_frames(self, frames: frozenset[int]) -> None:
        """Frames with at least one textual comment. Drives the small
        blue dot markers above the annotation triangles, so the user
        sees at a glance what kind of note lives on each frame
        (annotation only / comment only / both).

        Idempotent: same set → no repaint.
        """
        if frames == self._commented_frames:
            return
        self._commented_frames = frames
        self.update()

    # ------------------------------------------------------------------ Painting

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Brief §3 — timeline track sits on the deepest "ruler" surface
        # (BG_TRACK = #0F1013). One step deeper than BG_DEEP so the
        # warm cache-bar fills above read as "on top of" the track,
        # not flush with the panel chrome.
        painter.fillRect(self.rect(), C.BG_TRACK)

        if self._last <= self._first:
            return

        painter.setFont(self._label_font)
        if self._minimal_mode:
            # Fullscreen rendering: the user wants frame ticks + the
            # cache bar (so they can still see what's loaded), plus
            # the range bar and playhead. The chatty per-frame
            # markers (annotation triangles, comment dots, in / out
            # flags) stay off — they add visual noise on top of the
            # full-frame image content.
            self._draw_ticks_and_labels(painter)
            self._draw_range_bar(painter)
            self._draw_playhead(painter)
            self._draw_cache_bar(painter)
            return
        self._draw_ticks_and_labels(painter)
        self._draw_range_bar(painter)
        self._draw_in_out_markers(painter)
        self._draw_playhead(painter)
        self._draw_cache_bar(painter)
        self._draw_annotation_markers(painter)
        self._draw_comment_markers(painter)

        if self._dimmed:
            # Semi-transparent grey film matching the panel background
            # — desaturates ticks, cache bar, in/out flags and playhead
            # in one stroke so the user reads the timeline as "context
            # only, not a control" while contact-sheet mode is active.
            # Alpha tuned so the playhead is still legible (the user
            # needs to track playback position) but the surface clearly
            # looks disabled vs the bright normal state.
            painter.fillRect(self.rect(), QColor(20, 20, 22, 160))


    def _usable_width(self) -> int:
        return max(1, int(self.width() - 2 * self.MARGIN_X))

    def _total_frames(self) -> int:
        # "Frame as point" — see BarGeometry for the rationale. Length
        # is the number of intervals between first and last, not the
        # count of frames, so ``frame_to_x(first) == MARGIN_X`` and
        # ``frame_to_x(last) == MARGIN_X + usable_width`` exactly,
        # aligning the timeline's tick endpoints with the layer bar's
        # fill endpoints below.
        return max(1, self._last - self._first)

    def _frame_to_x(self, frame: int) -> float:
        ppf = self._usable_width() / self._total_frames()
        return self.MARGIN_X + (frame - self._first) * ppf

    def _x_to_frame(self, x: float) -> int:
        ppf = self._usable_width() / self._total_frames()
        raw = round((x - self.MARGIN_X) / ppf + self._first)
        return max(self._first, min(self._last, raw))

    def _tick_spacings(self) -> tuple[int, int]:
        ppf = self._usable_width() / self._total_frames()
        if ppf >= 8:
            return 1, 5
        if ppf >= 2:
            return 5, 25
        if ppf >= 0.5:
            return 25, 100
        if ppf >= 0.1:
            return 100, 500
        return 500, 2500

    def _format_label(self, frame: int) -> str:
        if self._display_mode == "tc":
            return frame_to_timecode(frame, self._fps)
        return str(frame)

    def _draw_ticks_and_labels(self, painter: QPainter) -> None:
        minor, major = self._tick_spacings()
        tick_baseline = self.TICK_TOP
        metrics = QFontMetrics(self._label_font)
        label_y = self.LABEL_H - 2

        minor_pen = QPen(C.TICK_MINOR, 1)
        major_pen = QPen(C.TICK_MAJOR, 1)

        last_label_x = -9999
        for frame in range(self._first, self._last + 1):
            is_major = (frame - self._first) % major == 0 or frame == self._last
            is_minor = (frame - self._first) % minor == 0
            if not (is_minor or is_major):
                continue
            x = round(self._frame_to_x(frame))
            if is_major:
                painter.setPen(major_pen)
                painter.drawLine(x, tick_baseline, x, tick_baseline + self.TICK_MAJOR_H)
                label = self._format_label(frame)
                label_w = metrics.horizontalAdvance(label)
                lx = x - label_w // 2
                # Clamp into widget bounds so edge labels (frame_first
                # / frame_last with MARGIN_X==0) don't get clipped.
                lx = max(0, min(int(self.width()) - label_w, lx))
                if lx - last_label_x > label_w + 8:
                    painter.setPen(C.TICK_LABEL)
                    painter.drawText(lx, label_y, label)
                    last_label_x = lx
            else:
                painter.setPen(minor_pen)
                painter.drawLine(x, tick_baseline, x, tick_baseline + self.TICK_MINOR_H)

    def _draw_range_bar(self, painter: QPainter) -> None:
        in_frame = self._in_frame if self._in_frame is not None else self._first
        out_frame = self._out_frame if self._out_frame is not None else self._last
        in_x = self._frame_to_x(in_frame)
        out_x = self._frame_to_x(out_frame)
        if out_x <= in_x:
            return
        rect = QRectF(in_x, self.RANGE_Y, out_x - in_x, self.RANGE_H)
        painter.fillRect(rect, C.RANGE_BAR)

    def _draw_in_out_markers(self, painter: QPainter) -> None:
        painter.setPen(QPen(C.MARKER_IO, 1.5))
        painter.setBrush(C.MARKER_IO)
        y_top = self.TICK_TOP
        y_bot = self.RANGE_Y + self.RANGE_H + 4

        if self._in_frame is not None:
            x = self._frame_to_x(self._in_frame)
            painter.drawLine(QPointF(x, y_top), QPointF(x, y_bot))
            flag = QPolygonF(
                [
                    QPointF(x, y_top),
                    QPointF(x + 6, y_top + 3),
                    QPointF(x, y_top + 6),
                ]
            )
            painter.drawPolygon(flag)
        if self._out_frame is not None:
            x = self._frame_to_x(self._out_frame)
            painter.drawLine(QPointF(x, y_top), QPointF(x, y_bot))
            flag = QPolygonF(
                [
                    QPointF(x, y_top),
                    QPointF(x - 6, y_top + 3),
                    QPointF(x, y_top + 6),
                ]
            )
            painter.drawPolygon(flag)

    def _draw_playhead(self, painter: QPainter) -> None:
        x = self._frame_to_x(self._current)
        painter.setPen(QPen(C.PLAYHEAD, 1))
        painter.drawLine(QPointF(x, self.TICK_TOP), QPointF(x, self.RANGE_Y + self.RANGE_H))
        triangle = QPolygonF(
            [
                QPointF(x - 5, self.LABEL_H - 1),
                QPointF(x + 5, self.LABEL_H - 1),
                QPointF(x, self.TICK_TOP + 6),
            ]
        )
        painter.setPen(QPen(C.PLAYHEAD_OUTLINE, 1))
        painter.setBrush(C.PLAYHEAD)
        painter.drawPolygon(triangle)

    def _draw_cache_bar(self, painter: QPainter) -> None:
        # Slot background: solid deep black so empty / not-yet-cached
        # frames are clearly readable as "nothing here yet". No outer
        # border anymore — the border belongs to each cached run, not
        # to the slot itself.
        bar_rect = QRectF(self.MARGIN_X, self.CACHE_TOP, self._usable_width(), self.CACHE_H)
        painter.fillRect(bar_rect, C.CACHE_BAR_BG)

        # Pass 0 — light-grey runs for multi-layer gaps. Drawn first
        # so the orange / red runs paint over them if a frame ever
        # ended up in both sets (shouldn't happen — a gap can't be
        # cached — but the order keeps the colour priority obvious).
        if self._gap_frames:
            painter.setBrush(QColor(160, 160, 160, 110))
            painter.setPen(QPen(QColor("#909090"), 1))
            self._draw_runs(painter, self._gap_frames)

        # Pass 0.5 — dim-orange wash for frames known to be on disk
        # (but not yet promoted to RAM). Drawn before the bright
        # "in-RAM" pass so the bright orange overpaints when a frame
        # ends up in both sets. Set difference removes the RAM /
        # missing / gap overlap so this pass only paints "disk-only"
        # frames. Brush is a 50%-saturation accent + 60% alpha → reads
        # as a subdued version of the cache bar rather than a separate
        # colour.
        disk_only = (
            self._disk_available_frames
            - self._cached_frames
            - self._missing_frames
            - self._gap_frames
        )
        if disk_only:
            painter.setBrush(QColor(232, 144, 28, 60))
            painter.setPen(QPen(QColor(232, 144, 28, 120), 1))
            self._draw_runs(painter, disk_only)

        # Pass 1 — orange runs of *real* cached frames (cached minus
        # missing). Drawn first so the red "missing" run lands on top
        # if both states ever overlap.
        ok_cached = self._cached_frames - self._missing_frames - self._gap_frames
        if ok_cached:
            painter.setBrush(C.CACHE_BAR)
            painter.setPen(QPen(C.CACHE_BAR_BORDER, 1))
            self._draw_runs(painter, ok_cached)

        # Pass 2 — red runs for missing frames. Same shape as the
        # cached runs (translucent fill + opaque border) so the
        # visual language stays consistent — only the colour
        # changes. Cohérent with the v0.4.1 review-tool palette
        # (red = "problem").
        if self._missing_frames:
            painter.setBrush(QColor(232, 74, 74, 128))   # #E84A4A @ 50%
            painter.setPen(QPen(QColor("#E84A4A"), 1))
            self._draw_runs(painter, self._missing_frames)

    def _draw_runs(self, painter: QPainter, frames: frozenset[int] | set[int]) -> None:
        """Coalesce ``frames`` into contiguous runs and paint them.

        Extracted so the same run-detection code is shared by the
        cached and missing passes — only the brush + pen differ.
        """
        in_range = sorted(f for f in frames if self._first <= f <= self._last)
        if not in_range:
            return
        run_start = in_range[0]
        prev = run_start
        for f in in_range[1:]:
            if f == prev + 1:
                prev = f
                continue
            self._draw_cache_run(painter, run_start, prev)
            run_start = f
            prev = f
        self._draw_cache_run(painter, run_start, prev)

    def _draw_cache_run(self, painter: QPainter, start: int, end: int) -> None:
        x1 = self._frame_to_x(start) - self._half_frame_width()
        x2 = self._frame_to_x(end)   + self._half_frame_width()
        painter.drawRect(QRectF(x1, self.CACHE_TOP, max(1.0, x2 - x1), self.CACHE_H))

    def _half_frame_width(self) -> float:
        return 0.5 * self._usable_width() / self._total_frames()

    def _draw_annotation_markers(self, painter: QPainter) -> None:
        """Big green triangle pointing UP toward the timeline track.

        Lives in the slack between the range bar (ends y=31) and
        the cache bar (starts y=41). Points up so the tip "lands"
        on the range bar — visually anchoring the marker to the
        frame it refers to.

        Boundary clamping: with ``MARGIN_X = 0`` (kept for layer-bar
        alignment), a marker at ``self._first`` lands at ``x=0`` and
        a marker at ``self._last`` at ``x=width`` — half the
        symmetric triangle would then sit outside the widget bounds
        and get clipped by Qt. We keep the **tip** at the exact
        frame x (so the visual anchor to the right frame is
        preserved) and clamp the **base corners** independently to
        the widget bounds. At the boundary frames the triangle
        becomes a right-triangle shape (vertical edge on the inside,
        slanted edge toward the centre) — slightly asymmetric but
        the tip still points unambiguously at the correct frame,
        which is the property that matters for review feedback.
        """
        if not self._annotated_frames:
            return

        marker_h = 9.0
        marker_w = 8.0
        marker_half = marker_w / 2.0
        # Tip flush against the range bar's bottom edge, base near
        # the cache bar — fills the 10 px slack.
        tip_y = self.RANGE_Y + self.RANGE_H  # 31
        base_y = tip_y + marker_h            # 40

        painter.setPen(Qt.PenStyle.NoPen)
        # Same green as the palette's "ok / approved" swatch —
        # familiar review-tool colour.
        painter.setBrush(QColor("#5DC46C"))

        in_range = sorted(
            f for f in self._annotated_frames if self._first <= f <= self._last
        )
        widget_w = float(self.width())
        for frame in in_range:
            x = self._frame_to_x(frame)
            # Tip stays on the exact frame x. Base corners clamp to
            # the widget edges independently — at the first frame
            # the left corner snaps to x=0 (= tip x), making a
            # right-triangle with the vertical edge on the left;
            # at the last frame the right corner snaps to widget_w
            # symmetrically.
            left_base = max(0.0, x - marker_half)
            right_base = min(widget_w, x + marker_half)
            tri = QPolygonF([
                QPointF(left_base, base_y),
                QPointF(right_base, base_y),
                QPointF(x, tip_y),
            ])
            painter.drawPolygon(tri)

    def _draw_comment_markers(self, painter: QPainter) -> None:
        """Blue triangle pointing DOWN toward the timeline track.

        Lives in the 5 px gap between the major-tick row bottom
        (y=23) and the range bar top (y=28). Mirror of the
        annotation triangle below — both point at the range bar
        for clear "this frame" anchoring.

        * green ▲ below alone               = annotation only
        * blue ▼ above alone                = comment only
        * blue ▼ above + green ▲ below      = both

        Different shape (triangle pointing different way) AND
        different colour AND different position — readable at a
        glance even when a heavy review LUT flattens colours.
        """
        if not self._commented_frames:
            return

        marker_h = 9.0
        marker_w = 8.0
        marker_half = marker_w / 2.0
        # Tip flush against the range bar top (y=28); base reaches
        # up into the major-tick row. Same dimensions as the
        # annotation triangle below — so a frame with both notes
        # shows two SYMMETRIC mirrored triangles. Slight overlap
        # with major tick lines at frames that happen to align is
        # acceptable: ticks are thin vertical bars, the triangle
        # is a bold filled shape, and the comment marker is the
        # more important info on a noted frame.
        tip_y = float(self.RANGE_Y)        # 28
        base_y = tip_y - marker_h          # 19

        painter.setPen(Qt.PenStyle.NoPen)
        # Same blue as the palette's "note" swatch — familiar
        # review-tool colour for "comment".
        painter.setBrush(QColor("#4A8DE8"))

        in_range = sorted(
            f for f in self._commented_frames if self._first <= f <= self._last
        )
        widget_w = float(self.width())
        for frame in in_range:
            x = self._frame_to_x(frame)
            # Same boundary-clamping rationale as
            # ``_draw_annotation_markers`` — tip stays exactly on
            # the frame, base corners clamp to widget bounds. At a
            # boundary frame the triangle becomes a right-triangle
            # but the tip's frame anchoring is preserved.
            left_base = max(0.0, x - marker_half)
            right_base = min(widget_w, x + marker_half)
            tri = QPolygonF([
                QPointF(left_base, base_y),
                QPointF(right_base, base_y),
                QPointF(x, tip_y),
            ])
            painter.drawPolygon(tri)

    # ------------------------------------------------------------------ Mouse

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        # Dimmed = contact-sheet mode is active. The timeline is purely
        # a read-only playhead indicator; per-tile scrub on the viewport
        # is the user's only direct frame-control gesture.
        if self._dimmed:
            return
        x = event.position().x()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl and self._last > self._first:
            # Ctrl + click → in/out drag mode. Choose IN or OUT
            # based on which side of the playhead the click
            # landed: left of cursor = in, right = out. The user
            # can then drag (still holding Ctrl) to fine-tune the
            # marker until release.
            click_frame = self._x_to_frame(x)
            if click_frame <= self._current:
                self._drag_mode = "drag_in"
                self.set_in_at_requested.emit(click_frame)
            else:
                self._drag_mode = "drag_out"
                self.set_out_at_requested.emit(click_frame)
            self.update()
            return
        self._scrubbing = True
        self.scrub_started.emit()
        self._emit_for_x(x)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dimmed:
            return
        x = event.position().x()
        if self._drag_mode is not None:
            if self._last <= self._first:
                return
            frame = self._x_to_frame(x)
            if self._drag_mode == "drag_in":
                self.set_in_at_requested.emit(frame)
            else:
                self.set_out_at_requested.emit(frame)
            self.update()
            return
        if self._scrubbing:
            self._emit_for_x(x)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        was_scrubbing = self._scrubbing
        self._scrubbing = False
        self._drag_mode = None
        if was_scrubbing:
            # Tells the app to switch video decoders back to precise
            # seeks and re-request the current frame at full accuracy.
            self.scrub_finished.emit()

    def _emit_for_x(self, x: float) -> None:
        if self._last <= self._first:
            return
        frame = self._x_to_frame(x)
        if frame != self._current:
            self._current = frame
            self.update()
        self.frame_requested.emit(frame)
