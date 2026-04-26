# UI re-skin — applying the Studio Dark charter

*Spec — 2026-04-26 · author: img_player team · status: draft awaiting user review*

## Context

img_player v0.1.0 ships a working VFX-grade image-sequence player with
solid playback performance (23.8 fps on 4K UHD multichannel EXR on the
target workstation, indistinguishable from OpenRV for human perception
on this content). The Phase 1 perf work is done.

The project also already has:

* `src/img_player/ui/theme.py` — a **complete** design-system module
  (palette, typography, spacing, geometry, global QSS). Source of truth
  for every styling decision.
* `ui_mockup.html` at the repo root — a **fully detailed visual mockup**
  of the target UI ("Studio Dark v1") authored by the user.
* `app.py` already calls `app.setStyleSheet(build_stylesheet())` at
  boot, so every Qt widget inherits the dark charter for free.
* The dock tabs `Color` / `Channels` are already implemented in
  `MainWindow` (`QTabWidget` inside a `QDockWidget`).

What's **missing** vs the mockup:

* Several widgets (`color_panel`, `channel_panel`, `main_window`,
  `viewer_widget`, `shortcuts_dialog`) only inherit the global QSS —
  they don't use the explicit tokens (`C`, `F`, `S`, `G`) and may have
  pétouilles (padding, hard-coded colours, group-box titles) that
  diverge slightly from the charter.
* The viewer has **no decorative brackets** in its corners.
* Transport buttons use **Qt-native icons** (`SP_MediaPlay`, etc.)
  instead of the custom SVG set shown in the mockup.
* The status bar is a **single monolithic `showMessage()` line** with
  no visual hierarchy — not the split layout (left = contextual
  message, right = perf indicators with coloured dots) shown in the
  mockup.
* No **effective playback FPS metric** is exposed at runtime — the
  player only knows its target fps from the combo box, not what it's
  actually delivering.

This spec captures the work to close those gaps.

## Goals

1. Bring img_player's UI in **fidelity** with `ui_mockup.html`.
2. Surface the **effective playback fps** as a live indicator the user
   can monitor without running the benchmark mode.
3. Keep all styling decisions in `theme.py` — never hard-code hex /
   pixel values in widget code.
4. Ship in **4 independently testable slices**, each delivering
   visible value when you launch the app.

## Non-goals

Explicitly **out of scope** for this spec — these are reported features
that will get their own design later:

* **HUD overlay** on the viewport (semi-transparent badges with
  colour-management info, RGB pixel-pick under the cursor, frame /
  timecode badge). Reported because it touches functional behaviour
  (toggle, pixel sampling) that deserves a focused design pass.
* **Giant frame number** background in the viewport when no sequence
  is loaded. Reported with the HUD because it's the same visual layer.
* **Layout dockable advanced** (saved presets, multi-screen layouts).
  The current `QDockWidget` already supports floating/moving.
* **Annotation review** (drawing, markers with notes).
* **A/B comparison** (wipe, side-by-side).
* **Loupe / zoom / pan** in the viewer.

## Architecture overall

Four slices. Each is one commit. Each can be reverted independently
without breaking the others. Each leaves the app in a runnable, visibly
better state than before.

| # | Slice | What you see when you launch | Key files |
|---|---|---|---|
| 1 | Effective FPS + status bar | Two-panel status bar, live fps indicator with coloured dot | `controller.py`, `app.py`, `main_window.py`, new `ui/status_format.py` |
| 2 | Custom SVG icons | Transport buttons with the mockup's stylised icons | new `ui/icons.py`, `transport.py` |
| 3 | Brackets viewer | Four discrete corner brackets in the image area | `viewer_widget.py`, new `ui/brackets_overlay.py` |
| 4 | Panel migration polish | Cohesive look across every panel | `color_panel.py`, `channel_panel.py`, `main_window.py`, `shortcuts_dialog.py` |

Shared invariants:

* All hex / px values come from `theme.py` (`C` / `H` / `F` / `S` / `G`
  classes). No exception.
* Every slice ships at least one pytest-qt smoke test that verifies
  the touched widget can be constructed without crash.
* Slices have **no inter-dependencies** — they can be reordered. The
  default order above is what gives the most visible payoff first.

## Slice 1 — Effective FPS metric + status bar

### Mesure (in `PlayerController`)

A new bounded deque holds tick timestamps:

```python
from collections import deque
import time

PREFETCH_AHEAD = 64        # existing
PREFETCH_BEHIND = 8        # existing
_TICK_WINDOW = 24          # ~1 s at 24 fps

class PlayerController(QObject):
    def __init__(...):
        ...
        self._tick_timestamps: deque[float] = deque(maxlen=_TICK_WINDOW)

    def _tick(self):
        self._tick_timestamps.append(time.monotonic())
        ...  # existing body unchanged

    def effective_fps(self) -> float | None:
        """Rolling-average effective playback fps, or None if not playing
        or insufficient samples."""
        if not self._state.is_playing or len(self._tick_timestamps) < 2:
            return None
        span = self._tick_timestamps[-1] - self._tick_timestamps[0]
        if span <= 0:
            return None
        return (len(self._tick_timestamps) - 1) / span

    def play(self):
        ...
        self._tick_timestamps.clear()  # discard pre-pause samples

    def pause(self):
        ...
        self._tick_timestamps.clear()

    def seek(self, frame: int):
        ...
        self._tick_timestamps.clear()
```

Rationale for clearing on `play/pause/seek`: keeps the metric reflecting
*current* playback, not stale data from a previous run. We do **not**
clear on `step()` (manual single-frame advance) because step happens
*alongside* an active playback intent and clearing would make the
metric flicker every time the user nudges the playhead. Step doesn't
fire the QTimer tick anyway, so the deque is naturally untouched.

### Refonte (in `MainWindow`)

The `QStatusBar` keeps existing widgets (no breaking change for callers
of `set_status()`) but gains two named labels added as permanent widgets:

```python
self._status_left  = QLabel()
self._status_left.setSizePolicy(Expanding, Preferred)
self._status_left.setStyleSheet(f"color: {H.TEXT_SECONDARY};"
                                f"font-size: {F.SIZE_XS}px;")

self._status_right = QLabel()
self._status_right.setAlignment(Qt.AlignmentFlag.AlignRight)
self._status_right.setTextFormat(Qt.TextFormat.RichText)  # for coloured dots
self._status_right.setFont(F.mono(F.SIZE_XS))

self.statusBar().addWidget(self._status_left, 1)        # stretch
self.statusBar().addPermanentWidget(self._status_right) # right-anchored
```

The existing `set_status(msg: str)` method is kept and re-routed to
`_status_left.setText(msg)` for backwards compatibility — no caller
changes needed.

### Logique (in `app._refresh_status`)

```python
def _refresh_status(self) -> None:
    if self._controller.sequence is None:
        return
    stats = self._cache.stats()
    state = self._controller.state

    # Left: contextual narrative — replaces the old monolithic line
    self._window.set_status(
        f"{state_message(state, self._controller.sequence)}"
    )

    # Right: perf indicators (rich text)
    eff = self._controller.effective_fps()
    cache_n = stats.frames_cached
    cache_total = self._controller.sequence.frame_count
    cache_ratio = stats.bytes_used / max(1, stats.bytes_budget)
    # "RAM" in the indicator = cache RAM occupancy (stats.bytes_used),
    # NOT process RSS. The user cares about how full *their* cache is —
    # the rest of the process is a constant ~few hundred MB and adding
    # it would just confuse the signal. Same convention as OpenRV's
    # status display.
    ram_gb = stats.bytes_used / 1024**3

    self._window.status_right.setText(
        format_perf_html(
            cache_n=cache_n, cache_total=cache_total, cache_ratio=cache_ratio,
            fps_effective=eff, fps_target=state.fps,
            ram_gb=ram_gb,
        )
    )
```

### Helpers (new `ui/status_format.py`)

Pure functions, no Qt dependency, easily unit-testable:

```python
from img_player.ui.theme import H

# Thresholds chosen so "ok" == green, "watch out" == amber, "broken" == red
_FPS_OK     = 0.95
_FPS_WARN   = 0.80
_CACHE_FULL = 0.80

def fps_dot_color(effective: float | None, target: float) -> str | None:
    """Returns the hex colour for the fps dot, or None when no dot at all
    (e.g. paused). Behaviour:
      - paused/unknown  → None (no dot, '— fps' rendered)
      - >= 95% target   → green
      - 80-95% target   → amber
      - <  80% target   → red
    """

def cache_dot_color(ratio: float) -> str | None:
    """ Returns green when ratio >= 0.8, None otherwise (mockup B). """

def format_perf_html(*, cache_n: int, cache_total: int, cache_ratio: float,
                     fps_effective: float | None, fps_target: float,
                     ram_gb: float) -> str:
    """Builds the rich-text HTML rendered in the right status label.
    Coloured dots are inline <span style='color:HEX'>●</span>."""
```

Default text when paused: `— fps` (em dash) instead of a number.

### Tests (slice 1)

* `tests/test_controller_fps.py` — instantiate `PlayerController`, force
  24 `_tick()` calls at simulated 41.67 ms intervals (monkeypatch
  `time.monotonic`), assert `effective_fps()` returns ~24 ± 0.5.
  Assert `pause()` resets it to None.
* `tests/test_status_format.py` — pure-function tests for
  `fps_dot_color`, `cache_dot_color`, `format_perf_html`. No Qt needed.
* `tests/test_main_window.py::test_status_bar_widgets_present` — pytest-qt
  smoke that verifies `status_left` and `status_right` exist on the
  freshly built window.

## Slice 2 — Custom SVG icons

### Module layout

New file `src/img_player/ui/icons.py`:

```python
"""Inline SVG icon factory — single source for transport / panel icons.

Icons are stored as XML strings with a ``{color}`` placeholder. We use
QSvgRenderer to paint them into a QPixmap of any size, then wrap as
QIcon. Cached via lru_cache on (name, color, size).
"""

from functools import lru_cache
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import Qt, QByteArray

from img_player.ui.theme import H

_TEMPLATES: dict[str, str] = {
    "play":  '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
             '<polygon points="4,2 14,8 4,14" fill="{color}"/></svg>',
    "pause": '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
             '<rect x="4" y="2" width="3" height="12" fill="{color}"/>'
             '<rect x="9" y="2" width="3" height="12" fill="{color}"/></svg>',
    "stop":  '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
             '<rect x="3" y="3" width="10" height="10" rx="1" fill="{color}"/></svg>',
    "prev":  '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
             '<polygon points="12,2 4,8 12,14" fill="{color}"/>'
             '<polygon points="6,2 4,8 6,14" fill="{color}" opacity="0.5"/></svg>',
    "next":  '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
             '<polygon points="4,2 12,8 4,14" fill="{color}"/>'
             '<polygon points="10,2 12,8 10,14" fill="{color}" opacity="0.5"/></svg>',
    "first": '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
             '<rect x="2" y="2" width="2" height="12" rx="1" fill="{color}"/>'
             '<polygon points="14,2 6,8 14,14" fill="{color}"/></svg>',
    "last":  '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
             '<rect x="12" y="2" width="2" height="12" rx="1" fill="{color}"/>'
             '<polygon points="2,2 10,8 2,14" fill="{color}"/></svg>',
}

@lru_cache(maxsize=64)
def make_icon(name: str, color: str = H.TEXT_PRIMARY, size: int = 18) -> QIcon:
    xml = _TEMPLATES[name].format(color=color)
    renderer = QSvgRenderer(QByteArray(xml.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)
```

### Wiring (in `transport.py`)

```python
# Before (Qt-native, ugly):
self._play_btn = _icon_button(
    style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
    "Play / Pause (Space)",
)

# After (charter-compliant):
from img_player.ui.icons import make_icon
from img_player.ui.theme import H

self._play_btn = _icon_button(make_icon("play", color=H.ACCENT), "Play / Pause (Space)")
self._stop_btn = _icon_button(make_icon("stop"),  "Stop")
...
```

The play/pause toggle in `update_from_state` swaps between
`make_icon("play", color=H.ACCENT)` and `make_icon("pause")` — same
pattern as today, just with our icons.

### Tests (slice 2)

* `tests/test_icons.py::test_make_icon_returns_non_null` — call
  `make_icon("play")` for every name in `_TEMPLATES`, assert
  `not icon.isNull()` and `icon.actualSize(QSize(18, 18))` is non-zero.
* `tests/test_icons.py::test_lru_cache_returns_same_instance` —
  asserts caching works (`make_icon("play") is make_icon("play")`).

## Slice 3 — Brackets viewer

### Approach

Adding the brackets directly to the `GLViewport` would require fiddling
with OpenGL, which is out-of-proportion for four straight lines. The
clean Qt way is to **stack a transparent overlay widget** on top of
the GL viewport using `QStackedLayout` in `StackingMode.StackAll`:

```
viewer_widget.py
└── QStackedLayout(StackAll)
    ├── GLViewport          (bottom — receives the image / drag&drop)
    └── BracketsOverlay     (top — paints 4 corner brackets)
```

`BracketsOverlay` has `setAttribute(WA_TransparentForMouseEvents)` so
clicks/drag&drop go through to the GL viewport.

### New file `src/img_player/ui/brackets_overlay.py`

```python
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor


class BracketsOverlay(QWidget):
    """Decorative L-shaped brackets in the four corners of the viewport.

    Painted directly so we don't need stylesheets or images. Transparent
    to mouse events — clicks fall through to the widget below.
    """

    BRACKET_SIZE   = 20      # px length of each bracket arm
    BRACKET_INSET  = 20      # px offset from the widget edge
    BRACKET_COLOR  = QColor(255, 255, 255, 30)   # rgba 0.12 alpha
    BRACKET_WIDTH  = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        ...  # 4 × (vertical line + horizontal line) at each corner
```

### Wiring (in `viewer_widget.py`)

The existing `ViewerWidget` currently nests `GLViewport` directly. We
swap its layout to `QStackedLayout(StackAll)` and add `BracketsOverlay`
on top.

### Tests (slice 3)

* `tests/test_brackets_overlay.py::test_overlay_is_transparent_to_mouse`
  — instantiate the overlay, post a `QMouseEvent`, assert it's not
  consumed.
* `tests/test_brackets_overlay.py::test_paints_at_correct_geometry` —
  call `paintEvent` on a sized overlay, render into a `QImage`,
  inspect pixels at the four corners to confirm the brackets are
  visible (alpha > 0) and the centre is fully transparent.

## Slice 4 — Panel migration polish

### Approach

These widgets already inherit the global stylesheet — the visible
rendering should be 90 % correct after slices 1-3. This slice is an
**audit + fix**, not a rewrite:

1. Launch the app, screenshot every panel and dialog.
2. Compare to `ui_mockup.html` side-by-side.
3. List divergences: paddings off, hard-coded colours, group-box
   titles, label colours, etc.
4. Fix each one by importing the right token from `theme.py`.

### Targeted fixes (likely)

* `color_panel.py` — replace any hard-coded hex with `H.*`. Ensure
  group-box titles use the charter's section-label style (uppercase,
  letter-spacing 1.5px, TEXT_DISABLED).
* `channel_panel.py` — list rows should match the mockup's coloured-
  dot style (`R` red, `G` green, `B` blue, `A` neutral, AOVs greyed).
* `main_window.py` — verify menu padding and the scan-state status
  message use TEXT_SECONDARY.
* `shortcuts_dialog.py` — keys are styled with a custom `background`
  inline today; replace with QSS class for cohesion.

### Tests (slice 4)

* No new test logic — this is visual polish. We rely on existing
  smoke tests still passing after token migration. Optional manual
  visual diff via screenshot before/after.

## Tests strategy (overall)

* All new code lands with a pytest-qt test that constructs the widget
  and verifies it doesn't crash — minimum bar.
* Pure-function modules (`status_format.py`, `icons.py`) get
  full-coverage unit tests because they have no Qt dependency.
* The bench harness (`python -m img_player --benchmark`) keeps working
  identically — none of these changes touch the playback path. We
  re-run it once after slice 1 lands to verify the fps metric is
  consistent with what the bench reports.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Some Qt versions render rich-text dots inconsistently in `QStatusBar` | low | Smoke-test on PySide 6.6 (CI-equivalent locally). Fall back to plain text indicators if dots break. |
| `QSvgRenderer` quirk: small icons get blurry on hi-DPI | medium | Render at 2× requested size when `devicePixelRatio() > 1`, then downscale via `QPixmap.setDevicePixelRatio`. |
| `QStackedLayout(StackAll)` swallows drag&drop events on some platforms | low | Confirm `WA_TransparentForMouseEvents` works on Windows 11 first; fall back to manual `eventFilter` if not. |
| Effective fps reads stale on long pauses | trivial | We `clear()` the deque on `play/pause/seek` — nothing to leak. |
| Slice 4 audit finds way more than expected | medium | Slice 4 is bounded to 1 day — anything beyond goes to a follow-up "polish 2" spec. |

## Estimation

| Slice | Effort |
|---|---|
| 1 — Effective FPS + status bar | 1 day (½ code, ½ tests + polish) |
| 2 — Custom SVG icons | ½ day |
| 3 — Brackets viewer | ½ day |
| 4 — Panel migration polish | 1 day (audit-bound) |
| **Total** | **~3 days** |

## Definition of done

* All four slices merged on `main`, each as a separate commit.
* Smoke tests in `tests/` cover every new public API
  (`controller.effective_fps`, `make_icon`, `BracketsOverlay`,
  `format_perf_html`).
* `python -m img_player` launches, plays a sequence, and visually
  matches `ui_mockup.html` (modulo HUD overlay which is reported).
* `python -m img_player --benchmark` still produces the same numbers
  it does today.
* `perf/BASELINE.md` re-run on the workstation post-slice-1 confirms
  the live fps reading agrees (±0.5) with the bench's reported
  effective fps.
* No hex / pixel value hard-coded outside `theme.py`.
