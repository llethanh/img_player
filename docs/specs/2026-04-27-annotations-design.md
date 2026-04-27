# Annotations — drawing tool for per-frame review notes (v1)

*Spec — 2026-04-27 · author: img_player team · status: draft awaiting user review*

## Context

The user needs a way to annotate sequences during review: draw a circle
around a problem area, write a remark on a specific frame, navigate
between frames that have notes. Two distinct workflows surfaced during
brainstorming:

* **Persistent review** — annotate frame 42 with "fix the timing here",
  close the app, hand the dossier to the director, they re-open and see
  the same notes on the same frames. The notes are part of the review
  document.
* **Ephemeral live show-and-tell** — during a screen-share, draw on the
  image to point at things, the strokes fade out by themselves so you
  don't have to manually erase between frames. Like Google Meet's
  screen-share annotation.

This spec covers **v1: persistent review only**. The ephemeral mode is
explicitly deferred to a v2 iteration once the persistent fundamentals
are solid; the toolbar / palette / stroke engine built here are
reused at zero rebuild cost.

## Decisions captured during brainstorming

The following architectural choices were ratified by the user before
this spec was written. They are load-bearing — re-opening any of them
ripples through the design.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Slice into v1 (persistent) + v2 (ephemeral)** | Persistent is the higher-value workflow (notes consulted weeks later). Ephemeral adds a temporal dimension to rendering and the data model — keep it isolated. |
| 2 | **Strokes live in image-space pixel coordinates** | Standard VFX review convention. Annotations zoom and pan with the image, stay anchored to the pixel they reference. |
| 3 | **Persistence = sidecar JSON next to the sequence** | Notes follow the dossier when copied/archived. Discoverable in the file explorer. Robust to renames as long as the sidecar stays in the same dir. |
| 4 | **Annotations hidden during playback by default, toggle to override** | Clean playback for animation review. Toggle (`A` key) for the case where you want to verify motion through an existing note. |
| 5 | **Per-frame undo stacks** | "Ctrl+Z undoes my last gesture *here*" mental model. No surprise frame jumps. Stacks are not persisted — fresh each session. |
| 6 | **Toolbar is hybrid: float overlay OR right-side dock** | User switches via a pin icon. Float for quick markup sessions; dock for longer drawing sessions where you want viewport real estate preserved. Mode persisted in prefs. |
| 7 | **Fixed palette of 7 high-contrast colors** | Vélocity for review workflows (palette pick = 0.3 s vs picker = 5 s). Curated to stay legible on any background. |
| 8 | **Eraser is per-stroke, click-only** | Vector data model — a stroke is an atomic unit. Cleaner UX than pixel eraser. Forgiving 8 px-image hit radius. |
| 9 | **Freeform pen only at v1** | Covers ~95 % of review needs. Arrows / lines / rectangles deferred to v1.x if the usage demands it. |

## User-facing summary

When the feature ships, the user can:

1. Press `D` (or click the ✏ button on the transport bar) to reveal the
   annotation toolbar. It appears in float mode at the top-left of the
   viewport, semi-transparent.
2. Click the pen icon and one of seven palette swatches (red, yellow,
   green, blue, orange, white, black). Drag the size slider for trait
   thickness (1–50 px image-space, default 5).
3. Drag on the image — a stroke is recorded at the current frame. The
   timeline above grows a small accent-colored marker at that frame.
4. Navigate to another frame, draw more strokes — each frame holds
   its own list. Press `[` / `]` to jump to the previous / next
   annotated frame.
5. Click the eraser, click on a stroke — it disappears. `Ctrl+Z`
   undoes the last action on the current frame; `Ctrl+Y` redoes.
6. Click the pin icon on the toolbar — it docks to the right of the
   viewport. Click again — back to floating. The mode is remembered.
7. Close the app — annotations are saved to a sidecar
   `.img_player_annotations.json` next to the sequence. Re-open later,
   they're back.

## Architecture

### New module tree

```
src/img_player/annotate/
├── __init__.py
├── stroke.py            # Stroke dataclass (frozen, immutable)
├── store.py             # AnnotationStore: per-frame state + Qt signals
├── persistence.py       # Sidecar JSON read/write (atomic)
├── overlay.py           # AnnotationOverlay QWidget — capture + render
└── toolbar.py           # AnnotationToolbar — composite widget, float/dock
```

### Modified modules

* `app.py` — instantiates `AnnotationStore`, wires shortcuts, calls
  `save_annotations()` at shutdown.
* `ui/main_window.py` — hosts the toolbar (float child of viewport, or
  child of the right-side `QDockWidget`).
* `ui/timeline.py` — paints accent-colored markers for annotated
  frames.
* `ui/transport.py` — adds three buttons: ✏ (toggle toolbar), ⏮ /
  ⏭ (prev / next annotated frame).
* `preferences.py` — persists `annotation_toolbar_mode`,
  `annotation_toolbar_pos`, `annotation_toolbar_visible`.

### Data flow

```
1. open_path(seq)
   → AnnotationStore.load(seq.dir / ".img_player_annotations.json")
   → emits annotated_frames_changed → Timeline repaints markers

2. user navigates to frame 42, presses D, presses pen
   → AnnotationOverlay.setEnabled(True) — captures mouse over GLViewport
   → mouse press / move / release → strokes appended to AnnotationStore[42]
   → emits frame_annotated(42) → Timeline grows a marker

3. paint loop
   → GLViewport paints the image (textured fullscreen quad)
   → AnnotationOverlay (parent of viewport, raise_()) paints the strokes
     in image-space, transformed via the same _fit_matrix (zoom + pan)

4. _shutdown
   → save_annotations(seq.dir / ".img_player_annotations.json", store)
   → atomic .tmp + rename, mirroring profile.json
```

### Key architectural points

* **Strict MVC.** `AnnotationStore` knows nothing about Qt widgets (only
  `QObject` for signals). `AnnotationOverlay` reads the store, paints,
  writes via the toolbar's tool/color/size state. Toolbar mutates the
  store. Timeline observes signals.
* **Overlay sits above the viewport.** A `QWidget` parented to the
  `GLViewport`, `setAttribute(WA_TransparentForMouseEvents, False)` only
  when the pen or eraser is active — otherwise the mouse passes through
  to the viewport's existing drag-scrub / pan / zoom handlers.
* **Coords in image-space throughout.** `Stroke.points: tuple[tuple[float, float], ...]`
  in image pixels. Rendering applies the same transform as the image,
  reusing the helper math from PR #34's zoom-anchor fix.
* **No coupling with the OCIO shader.** Strokes are painted with stock
  `QPainter` on the overlay above the `QOpenGLWidget`. The color shader
  stays untouched.

## Drawing engine & data model

### Stroke

```python
@dataclass(frozen=True)
class Stroke:
    """One pen-down, drag, pen-up gesture as a polyline in image-space."""
    points: tuple[tuple[float, float], ...]  # (x, y) in image pixels
    color: str                                # "#E84A4A" — hex from the palette
    size: float                               # diameter in image pixels
```

* Tuples (not lists): the object is immutable, so the undo stack can
  reference it without risk of mutation.
* No ID / timestamp at v1: the index in the per-frame list is the
  stroke's identity. Sufficient for delete-by-index on the eraser.
* `size` in image-pixels, not widget-pixels: a 5 px-image stroke stays
  visually proportional to the image at any zoom.

### AnnotationStore

```python
class AnnotationStore(QObject):
    annotated_frames_changed = Signal()   # set of annotated frames mutated
    frame_annotated = Signal(int)         # specific frame got a new stroke

    _strokes: dict[int, list[Stroke]]     # frame_index -> ordered strokes
    _undo:    dict[int, list[Action]]     # per-frame undo stack
    _redo:    dict[int, list[Action]]     # per-frame redo stack

    def add_stroke(frame: int, stroke: Stroke) -> None: ...
    def remove_stroke(frame: int, idx: int) -> None: ...
    def undo(frame: int) -> bool: ...     # returns True if anything changed
    def redo(frame: int) -> bool: ...
    def annotated_frames(self) -> frozenset[int]: ...  # for timeline markers
    def strokes_at(self, frame: int) -> tuple[Stroke, ...]: ...
```

`Action` is `("add", frame, idx, stroke)` or `("remove", frame, idx, stroke)` —
symmetric, so undo and redo are trivial mirrors.

### Mouse capture

The overlay implements:

* `mousePressEvent` (pen active, button down) → opens
  `_current_stroke = [(x, y)]` after converting widget→image.
* `mouseMoveEvent` (button held) → appends the new point if it is
  ≥ 1 px-image from the previous one (decimation: avoids sub-pixel
  noise inflating the polyline).
* `mouseReleaseEvent` → finalises the stroke
  (`Stroke(tuple(points), color, size)`), calls
  `store.add_stroke(frame, stroke)`.

For the **eraser**:

* `mousePressEvent` (eraser active) → finds the stroke whose minimum
  point-segment distance to the cursor is ≤ 8 px-image. If found,
  `store.remove_stroke(frame, idx)`. If not, no-op.
* No drag for the eraser at v1 — click-only, per-stroke. Simple, net.

### Coordinate conversion helpers

Module-level pure functions in `overlay.py`, mirrors of PR #34's
`_anchored_pan_for_zoom` for testability:

```python
def widget_to_image(
    widget_xy: tuple[float, float],
    widget_size: tuple[int, int],
    img_size: tuple[int, int],
    factor: float,
    pan: tuple[float, float],
) -> tuple[float, float]: ...

def image_to_widget(
    image_xy: tuple[float, float],
    widget_size: tuple[int, int],
    img_size: tuple[int, int],
    factor: float,
    pan: tuple[float, float],
) -> tuple[float, float]: ...
```

Each tested with parametrised round-trip cases (`widget_to_image` then
`image_to_widget` returns the same point, modulo float rounding).

### Rendering

The overlay's `paintEvent` redraws all strokes for the current frame:

```python
painter = QPainter(self)
painter.setRenderHint(QPainter.RenderHint.Antialiasing)

factor, pan = self._gl_viewport.current_transform()  # (zoom factor, pan offset)
img_size = self._gl_viewport.image_size()

for stroke in self._store.strokes_at(self._current_frame):
    pen = QPen(QColor(stroke.color))
    pen.setWidthF(stroke.size * factor)        # scale with zoom
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    path = QPainterPath()
    p0 = image_to_widget(stroke.points[0], (self.width(), self.height()),
                         img_size, factor, pan)
    path.moveTo(*p0)
    for p in stroke.points[1:]:
        path.lineTo(*image_to_widget(p, ...))
    painter.drawPath(path)
```

`update()` is called when:
* the store mutates (new stroke, eraser, undo/redo),
* the current frame changes,
* the GL viewport's zoom/pan change (signal subscription).

### Visibility during playback

`AnnotationStore` carries a `_show_during_playback: bool = False` flag.
The overlay's `paintEvent` short-circuits when the controller is
playing and the flag is False:

```python
if self._controller.is_playing and not self._store.show_during_playback:
    return  # paint nothing
```

`A` key toggles the flag. The default `False` keeps playback clean for
animation review — most users will never need to flip it.

## Toolbar (hybrid float/dock)

### Visual structure

A vertical composite widget:

```
┌──────────────┐
│   📌 pin     │  ← toggle float ⇄ dock
├──────────────┤
│   ✏ pen      │  ← active button (orange when active)
│   🧽 eraser  │
├──────────────┤
│  ● ● ● ● ●   │  ← 7 color swatches (4-3 grid in float, vertical column in dock)
│  ● ● ●       │
├──────────────┤
│   ▔▔●▔▔     │  ← size slider (1-50 px image-space)
│   "5px"      │  ← live size readout
├──────────────┤
│   ↶ undo     │
│   ↷ redo     │
└──────────────┘
```

Final icon visuals will be designed at implementation time alongside
the rest of the SVG icon set in `ui/icons.py`.

### Mode float vs dock — re-parenting

A single `AnnotationToolbar` instance, two placements:

* **Float** — `setParent(self._gl_viewport)`,
  `setWindowFlags(Qt.WindowType.SubWindow)`, semi-transparent background
  (`rgba(36, 36, 40, 0.92)` to match `BG_RAISED` at ~92 % alpha).
  Position absolute, raised above the `AnnotationOverlay` via `raise_()`.
  Default position: top-left at `(12, 12)`. Draggable by its title bar
  (the strip containing the pin icon).
* **Dock** — embedded in a `QDockWidget` anchored to the right of the
  `MainWindow`. Cohérent with the existing `color_panel` dock pattern.
  Background `BG_RAISED` (opaque). Width pinned to ~80 px.

```python
def set_mode(self, mode: ToolbarMode) -> None:
    """ToolbarMode.FLOAT or ToolbarMode.DOCK. Re-parents the widget."""
    if mode == ToolbarMode.FLOAT:
        self._dock_wrapper.setWidget(None)
        self.setParent(self._gl_viewport)
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.move(self._floating_pos)
        self.raise_()
        self.show()
    else:  # DOCK
        self.setParent(None)
        self.setWindowFlags(Qt.WindowType.Widget)
        self._dock_wrapper.setWidget(self)
        self._dock_wrapper.show()
    self._mode = mode
    self._save_mode_to_prefs()
```

Pin icon click emits `pin_toggled` which calls `set_mode(opposite)`.

### Preferences

Three new fields in the `Preferences` QSettings wrapper:

```python
annotation_toolbar_mode: str = "float"             # "float" or "dock"
annotation_toolbar_pos: tuple[int, int] = (12, 12) # in float mode
annotation_toolbar_visible: bool = False           # show on startup
```

At boot, `App._apply_preferences` sets the mode (float/dock) and
visibility before showing the window.

### Transport bar integration

Three new buttons added to the right of the existing transport row:

| Button | Action |
|---|---|
| ✏ Toggle annotations | Show/hide toolbar (= `D` shortcut). Active visual when pen or eraser is selected. |
| ⏮ Prev annotation | Jump to the highest annotated frame `< current_frame`. Disabled (greyed) when none. |
| ⏭ Next annotation | Jump to the lowest annotated frame `> current_frame`. Disabled when none. |

### Keyboard shortcuts

Registered via `QShortcut(seq, self._window)` in `App._wire`. All
disabled when the focus is on a `QLineEdit` (consistent with
the existing Space/play behaviour).

| Shortcut | Action |
|---|---|
| `D` | Toggle annotation toolbar visibility |
| `P` | Pen tool (only when toolbar visible) |
| `E` | Eraser tool (only when toolbar visible) |
| `Ctrl+Z` | Undo last action on the current frame |
| `Ctrl+Y` | Redo last undone action on the current frame |
| `A` | Toggle annotations during playback |
| `[` | Previous annotated frame |
| `]` | Next annotated frame |

## Persistence (sidecar JSON)

### Location

When the user opens a sequence detected at `<dir>/<basename>.<frame>.<ext>`,
the sidecar lives at:

```
<dir>/.img_player_annotations.json
```

* Dot-prefix: hidden on Linux/macOS, less visible on Windows. Avoids
  cluttering the dossier visually while staying discoverable.
* One sidecar per dossier: if multiple sequences share a folder, the
  schema distinguishes them by basename.
* No global cache at boot: `load_annotations()` runs in
  `App._open_path`, `save_annotations()` runs in `App._shutdown`.

### Schema (v1)

```json
{
  "schema_version": 1,
  "saved_at": "2026-04-27T16:42:00+00:00",
  "img_player_version": "0.3.0",
  "sequences": {
    "render": {
      "frames": {
        "42": [
          {
            "color": "#E84A4A",
            "size": 5.0,
            "points": [[1024.5, 532.0], [1028.1, 535.7], [1031.0, 540.2]]
          }
        ],
        "87": [ ... ]
      }
    }
  }
}
```

* `schema_version`: 1. Loader returns `None` for any other version
  (graceful, no crash).
* `sequences[<basename>]`: keyed by sequence basename so multiple
  sequences in the same dossier do not collide.
* `points`: float-2-tuples, no compression at v1. Re-evaluate if
  files exceed ~10 MB per sequence.
* `saved_at` + `img_player_version`: post-mortem debugging metadata.

### Save & load

```python
def save_annotations(path: Path, store: AnnotationStore) -> None:
    """Atomic JSON write — .tmp then rename. Best-effort: never raises
    user-facing, just logs a warning on permission errors so a read-only
    Drive Stream session doesn't crash the app at shutdown."""
    payload = {
        "schema_version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "img_player_version": __version__,
        "sequences": store.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_annotations(path: Path) -> AnnotationStore | None:
    """Returns None if the file is missing, malformed, wrong schema,
    or the dir is unreadable. Never raises."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            return None
        store = AnnotationStore()
        store.load_from_dict(data["sequences"])
        return store
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
        return None
```

### Error handling

| Failure | Behaviour |
|---|---|
| Read-only dossier (Drive Stream offline, USB write-protect, NAS) | `save_annotations` swallows the OSError, logs a warning. Annotations live in memory until close, then are lost. v1.x: fallback to a global-cache mapping `<sequence_path_hash> → annotations`. |
| Corrupt sidecar at load | `load_annotations` returns `None`, app starts with an empty store. Bad file stays on disk for the user to investigate. |
| Multiple sequences opened in succession | The store resets on each `_open_path`. No cross-contamination. The shutdown saves only the currently open sequence — earlier sequences were already saved when they were closed. |

## Timeline integration

### Markers

When a frame has at least one stroke, the timeline renders a small
indicator above the existing track:

```
[████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] cache fill bar
 ▼     ▼    ▼              ▼          ← annotation markers
[──────────█──────────────────────────] timeline track + playhead
```

* Small triangle pointing down, ~6 px tall, rendered above the timeline
  bar (not on it — keeps the playhead and cache fill bar uncluttered).
* Colour `ACCENT_BRIGHT` (`#F5A830`) for visibility against the dark
  theme.
* No opacity scaling by stroke count — annotated is annotated, period.
* Hover tooltip: `"Frame N — K strokes"`.
* Click → seek to that frame (free side-effect: existing
  `frame_to_x` math handles the position).

Implementation: a new `_paint_annotation_markers` step in
`Timeline.paintEvent`, called after the cache bar and before the
playhead. Iterates over `store.annotated_frames()`.

### Prev/next navigation

```python
def prev_annotated_frame(self) -> int | None:
    candidates = [f for f in self._store.annotated_frames() if f < self._current_frame]
    return max(candidates) if candidates else None

def next_annotated_frame(self) -> int | None:
    candidates = [f for f in self._store.annotated_frames() if f > self._current_frame]
    return min(candidates) if candidates else None
```

* No wrap-around: if the user is on the first annotated frame, prev is
  a no-op. Standard VFX convention.
* Buttons are `setEnabled(False)` (greyed) when the corresponding
  function returns `None`. Tooltip updates to "Aucune annotation
  avant" / "Aucune annotation après".

### Reactivity

`store.annotated_frames_changed` → three consumers:

1. `Timeline.update()` repaints the markers.
2. `TransportBar` updates the prev/next buttons' `enabled` state.
3. The overlay listens to the more specific `frame_annotated(N)` and
   only `update()`s if `N == current_frame`.

## Testing strategy

Project pattern: pure functions are unit-tested, Qt orchestration is
exercised by manual smoke tests. We stay consistent.

### Unit tests (TDD-friendly)

| Module | Coverage |
|---|---|
| `stroke.py` | round-trip serialization, dataclass immutability, `points` is always a tuple |
| `store.py` | add/remove/undo/redo invariants, per-frame stack isolation, `annotated_frames()` set after each mutation, signal emission counts |
| `persistence.py` | round-trip `save → load`, unknown schema version returns None, malformed JSON returns None, missing file returns None, atomic .tmp cleanup, the autouse fixture from PR #36 already isolates the path |
| `overlay.py` math helpers | `widget_to_image` / `image_to_widget` round-trip parametrised across factors and pan offsets, point-segment distance for eraser hit-testing |
| `toolbar.py` `set_mode` | float→dock→float preserves widget state, prefs round-trip on shutdown / boot |

Target: ~40 new tests. Suite total: ~340/340 expected green.

### Not unit-tested

* `paintEvent` of the overlay — visually verified at PR review.
* Drag of the floating toolbar by its title bar — Qt event loop, manual.
* Exact timing of `update()` repaints — observable, not asserted.

### Manual test plan (PR checklist)

* [ ] Draw on frame 42, navigate elsewhere, return → strokes reappear.
* [ ] Quit app, re-open the sequence → strokes are still there.
* [ ] Zoom 2× → strokes scale with the image, stay anchored to the pixel.
* [ ] Eraser → click between two close strokes → only the targeted one disappears.
* [ ] Undo/redo on one frame, navigate, undo on another → frames are independent.
* [ ] Float → click pin → docks right → click pin → returns to float at the same position.
* [ ] Relaunch the app → toolbar comes back in the previous mode/position.
* [ ] Play → annotations hidden by default, `A` forces them visible.
* [ ] `[` `]` jumps to adjacent annotated frames, no wrap.
* [ ] Save fails on a read-only Drive Stream session → warning logged, no crash.

## Out-of-scope (deferred)

### v2 — Ephemeral mode

Strokes with a temporal lifecycle: appear → fade → disappear. Slider
expressed as a global indication (e.g. five steps: Très court / Court /
Moyen / Long / Très long, no second values). No persistence, no
timeline markers. Reuses the toolbar / palette / size slider from v1
with zero rebuild — only adds a fade engine on top of the existing
stroke list.

### v1.x — small features that can wait

* Primitives: arrow first if the usage demands it, then line / rectangle
  / ellipse.
* "More colors..." button opening a `QColorDialog`.
* Export an annotated frame as a flattened PNG (for sharing).
* Pen tablet pressure (Wacom) — possible but adds a dependency.

### Probably never

* Cross-frame annotations ("this stroke visible on frames 40–60").
* Annotation layers / groups.

## Implementation slicing (preview for writing-plans)

The `writing-plans` skill will produce the final plan. Natural
decomposition:

* **Slice 1** — `stroke.py` + `store.py` + `persistence.py`. Pure
  Python, no Qt widget code. ~25 unit tests. 1 PR.
* **Slice 2** — `overlay.py`: drawing engine + rendering, wired to
  GLViewport. Reuses zoom-anchor math. 1 PR.
* **Slice 3** — `toolbar.py`: composite widget with float/dock switch,
  palette, size slider, undo/redo buttons, prefs persistence. 1 PR.
* **Slice 4** — Timeline markers, transport buttons (✏ ⏮ ⏭),
  keyboard shortcuts, final integration. 1 PR.

Estimated ~3–4 days of work across the 4 slices.
