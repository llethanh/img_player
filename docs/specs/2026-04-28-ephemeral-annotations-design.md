# Ephemeral annotations — design (v0.4.1)

**Status :** approved 2026-04-28
**Target release :** v0.4.1 (patch, additive only)
**Predecessor :** [2026-04-27-annotations-design.md](2026-04-27-annotations-design.md) (the persistent annotations system shipped in v0.4.0)

## 1. Intent

Add a second drawing mode to the existing annotation system : **ephemeral strokes** that fade out progressively over a few seconds, like Google Meet's screen-share annotations. The use case is live presentation during a video call — drawing on the image to point at things while talking, without polluting the persistent review notes (sidecar JSON).

A toolbar toggle (`👻`) switches between **Persistent mode** (current behaviour, strokes saved to sidecar) and **Ephemeral mode** (strokes live in memory, fade linearly, never persisted). A 3-preset selector under the toggle picks the fade duration.

This is intentionally a small surface — one new module, edits to overlay/toolbar/app/preferences, ~30 new tests. No sidecar schema bump, no new dependency.

## 2. Decisions (agreed during brainstorming)

| # | Question | Decision |
|---|---|---|
| Q1 | What happens to ephemeral strokes when the user scrubs to a different frame? | **Image-space, frame-agnostic.** Strokes float on the image regardless of frame, until they fade. Presenter model — gestures comment what's currently shown live, not a specific frame. |
| Q2 | UI for mode toggle | **Dedicated `👻` toggle button** at the top of the toolbar (just below the `📌` pin). Toolbar border tints cyan when active. The pen glyph swaps `✏️` → `👻` to confirm "the pen is now ephemeral". Eraser greyed out in ephemeral mode. |
| Q3 | Duration control | **3 compact presets `[● ● ●]`** mapped to 2s / 5s / 10s. Visible only when ephemeral mode is on. Default = preset 1 (5s). Persisted across sessions. |
| Q4a | Eraser in ephemeral mode | Disabled. If active when mode is toggled on → auto-switches to `ToolKind.NONE`. |
| Q4b | Undo in ephemeral mode | `Ctrl+Z` calls `manager.kill_last()` — kills the most recently born live stroke immediately. |
| Q4c | Redo in ephemeral mode | No-op. A faded stroke is gone for good. |
| Q4d | Clear in ephemeral mode | The existing `Clear` button calls `manager.clear_all()` — instant wipe of every live ephemeral stroke. Tooltip changes to reflect the dual role. |
| Q4e | Visibility during playback | Always visible. Independent of `store.show_during_playback`. The whole point is live presentation. |
| Q4f | Keyboard shortcut | `G` (mnemonic "ghost") toggles ephemeral mode on/off. |
| Q4g | Persistence | **Strictly zero.** Sidecar JSON is never touched. Dirty flag never flips. Close prompt never appears for ephemeral strokes. They die with the process. |
| Q4h | Fade curve | **Linear** alpha 1.0 → 0.0 over the configured duration. No hold-then-fade, no expiration flash. |

## 3. Architecture

A single new module : `src/img_player/annotate/ephemeral.py`. Mirrors the conceptual structure of `AnnotationStore` (one store per "type of stroke") but specialised for time-bound, non-persistent state.

### 3.1 New class : `EphemeralStrokeManager(QObject)`

```python
class EphemeralStrokeManager(QObject):
    repaint_needed = Signal()

    def __init__(self, *, parent: QObject | None = None) -> None: ...
    def set_duration(self, seconds: float) -> None: ...
    def add(self, stroke: Stroke) -> None: ...
    def kill_last(self) -> bool: ...
    def clear_all(self) -> int: ...
    def live_strokes_with_alpha(self) -> tuple[tuple[Stroke, float], ...]: ...
    def has_live_strokes(self) -> bool: ...
```

**Internal state**
- `_strokes: list[tuple[Stroke, float]]` — `(stroke, birth_monotonic_ts)`. `time.monotonic()` is used (not `time.time()`) to be robust to system clock adjustments during a video call (NTP, suspend/resume).
- `_duration_s: float` — fade duration in seconds, set by the toolbar via `set_duration()`.
- `_timer: QTimer` — interval **33 ms** (≈30 FPS). Suffices for visually smooth fade without burning CPU.

**Auto start/stop of the timer**
- `add()` starts the timer when the list was previously empty.
- Each tick : sweep expired strokes (alpha == 0), emit `repaint_needed`, and stop the timer if the list is now empty. **No idle polling.**

### 3.2 New module-level pure helper

```python
def alpha_at(birth_ts: float, now_ts: float, duration_s: float) -> float:
    """Linear 1.0 → 0.0 over [birth, birth+duration]. Clamp to [0, 1]."""
    age = max(0.0, now_ts - birth_ts)
    if duration_s <= 0.0:
        return 0.0
    return max(0.0, 1.0 - age / duration_s)
```

Same testability pattern as `widget_to_image()` / `nearest_stroke_index()` in `overlay.py` — pure, no Qt, unit-testable in isolation.

### 3.3 Integration points (no structural changes)

| Existing component | Change |
|---|---|
| `AnnotationStore` | **None.** Ephemeral strokes never touch it. |
| `AnnotationOverlay` | Gains 2 setters (`set_ephemeral_mode`, `set_ephemeral_manager`). `mouseReleaseEvent` routes the finished stroke based on a snapshot taken at `mousePressEvent`. `paintEvent` reads `manager.live_strokes_with_alpha()` after the persistent strokes pass. |
| `AnnotationToolbar` | Gains the `👻` toggle (26×22), the 3-preset bar (visible only in ephemeral mode), and 2 new signals : `ephemeral_mode_changed(bool)`, `ephemeral_duration_changed(float)`. The pen glyph swaps `✏️` ↔ `👻` based on mode. The eraser is disabled in ephemeral mode. |
| `app.py` | Creates the manager. Wires toolbar signals to overlay/manager. Loads/saves preset preference. Calls `manager.clear_all()` on sequence change. |
| `preferences.py` | Adds `ephemeral_duration_preset: int` (0/1/2 → 2s/5s/10s, default 1). |

**Untouched** : sidecar JSON, persistence layer, comment store, timeline markers (green ▲ / blue ▼ remain strictly for persistent annotations + comments).

## 4. Data flow

### Phase A — User clicks `👻`

```
toolbar._on_ephemeral_toggle_clicked
   → toolbar updates internal _ephemeral_mode flag
   → toolbar repaints (border cyan, pen glyph 👻, eraser greyed)
   → toolbar.ephemeral_mode_changed.emit(True)
                ↓
        app.py._on_ephemeral_mode_changed(True)
                ↓
        overlay.set_ephemeral_mode(True)
```

The manager is mode-agnostic — it doesn't care whether mode is on. It simply collects strokes when the overlay calls `add()`.

### Phase B — Drawing an ephemeral stroke

```
mousePress (LeftButton, pen tool)
   → overlay starts in-progress stroke (same code as persistent —
     image-space points, midpoint quadratic smoothing).
   → overlay snapshots self._ephemeral_mode into a per-drag local
     so the routing decision survives a mid-drag mode toggle.
   → in-progress stroke renders at full alpha (the timer doesn't
     see it yet).

mouseMove (samples accumulate, no change vs persistent)

mouseRelease
   → overlay constructs Stroke(...).
   → if drag-snapshot says ephemeral:
         manager.add(stroke)
     else:
         store.add_stroke(current_frame, stroke)
   → manager.add():
         appends (stroke, time.monotonic()) to _strokes
         if list was empty → timer.start(33ms)
         emits repaint_needed
   → overlay.update() → paintEvent → manager.live_strokes_with_alpha()
```

### Phase C — Timer tick & expiration

```
QTimer fires every 33 ms (only while strokes are alive)
   → manager._on_tick():
         now = time.monotonic()
         remove every stroke with alpha_at(birth, now, duration_s) == 0
         emit repaint_needed
         if _strokes is empty → timer.stop()
   → overlay.paintEvent:
         draws persistent strokes at full alpha (current behaviour)
         then draws live ephemeral strokes — for each (stroke, alpha):
             color = QColor(stroke.color); color.setAlphaF(alpha)
             same _draw_stroke() codepath as persistent, with the
             modified pen color.
```

## 5. Edge cases

1. **Mid-drag mode toggle.** The routing decision is bound at `mousePressEvent` (snapshot into `_current_stroke_is_ephemeral`). Releasing always goes to the cible decided at press. Avoids "I started persistent, ended ephemeral because a hotkey fired mid-drag".
2. **Toggle OFF with live ephemeral strokes.** They keep fading. The manager is mode-agnostic. You can draw an ephemeral circle, toggle off, draw a persistent line — both coexist until the first fades.
3. **App close with live strokes.** No prompt, no persistence. Timer dies with the parent widget. Clean.
4. **Toggle ON when `ToolKind.NONE`.** Allowed. Mode is set ahead of any tool. The next click on `✏️` will pick up the ephemeral mode automatically.
5. **Float toolbar dragged with live strokes.** Independent. Toolbar position has nothing to do with the strokes (which are anchored in image-space on the GL viewport).
6. **Window resize / fullscreen toggle.** Image-space → strokes follow zoom/pan/resize for free. No special code.
7. **Sequence change.** `app.py` calls `manager.clear_all()` in the sequence-load callback. Otherwise ghosts from the previous sequence would float over the new one.
8. **Mode ON + show_during_playback OFF + scrub during playback.** Ephemeral strokes still render. The playback gating in `paintEvent` only applies to persistent strokes. Decision Q4e.
9. **Performance.** For a typical 5s duration and ~1 stroke/second drawing rate, simultaneous live count is ~5. Negligible. No cap, no guard.
10. **Color & size in ephemeral mode.** Reuse the active palette swatch and the existing Size slider. No dedicated palette/size for ephemeral. The fade itself signals "this disappears" — adding visual distinctiveness on top would be redundant.
11. **In-progress stroke during a drag.** Renders at alpha 1.0 fixed until release. Timer doesn't see it. At release, `birth_ts = monotonic()` is set and the fade begins.
12. **Duration preset changed mid-life.** The slider updates `_duration_s`. Already-born strokes keep their `birth_ts` but their alpha is recalculated against the new duration. Useful : you can shorten on the fly if the screen is getting cluttered.

## 6. UI specification

### 6.1 Vertical layout (toolbar, top to bottom)

```
┌──────────────┐
│      📌      │  pin (existing)
│  ──────────  │
│      👻      │  NEW : ephemeral toggle, 26×22, checkable
│  [● ● ●]     │  NEW : 3-preset duration, ~80×16, visible only when 👻 active
│  ──────────  │
│   ~~~~~      │  stroke preview (existing)
│  ✏️    🧽    │  pen + eraser (existing ; eraser greyed in ephemeral)
│  ──────────  │
│    Size      │
│  ─────●──    │  size slider (existing)
│   12px       │
│  ──────────  │
│  ●  ●        │  color palette (existing)
│  ●  ●        │
│  ●  ●        │
│   ●          │
│  ──────────  │
│  ↶    ↷      │  undo / redo
│  ──────────  │
│   [Clear]    │  Clear (tooltip changes by mode)
└──────────────┘
```

### 6.2 The `👻` toggle button

- `QToolButton`, checkable, fixed 26×22.
- Glyph : native `👻` emoji (consistent with `📌`, `✏️`, `🧽` already used).
- Tooltip : `"Mode éphémère (G) — les traits s'effacent progressivement, non sauvegardés"`.
- Checked state : the toolbar's outer border switches from the default subtle grey to **cyan accent `#4A8DE8`** (the existing "blue note" color from the palette). Visible in both float and dock modes.

### 6.3 The 3-preset duration bar

- Three `QToolButton`s (checkable, exclusive `QButtonGroup`) in an HBox.
- Each shows a dot of increasing size : `●` 6 px / 9 px / 12 px.
- Internal mapping : `{0: 2.0, 1: 5.0, 2: 10.0}` seconds.
- Default : index 1 (5 s).
- Tooltips : `"Court · ~2s"`, `"Moyen · ~5s"`, `"Long · ~10s"`.
- **Visibility** : entire row hidden (collapse, no blank space) when ephemeral mode is off.
- Persisted in `Preferences.ephemeral_duration_preset`.

### 6.4 Pen glyph swap

When `_ephemeral_mode == True`, the toolbar swaps `_pen_btn.setText("✏️")` → `setText("👻")`. Tooltip swaps to `"Pen éphémère (P) — clic-glisser, le trait s'effacera tout seul"`. The swap is purely visual — `ToolKind` stays `PEN`. Avoids inventing a `ToolKind.GHOST_PEN` that would duplicate all the capture logic.

### 6.5 Eraser handling

- `setEnabled(False)` in ephemeral mode → Qt greys it automatically.
- If eraser was active when ephemeral mode is toggled on : auto-switch the tool to `ToolKind.NONE`. Avoids the inconsistent "greyed but checked" state.

### 6.6 Clear button

- Label `"Clear"` unchanged.
- Tooltip dynamic :
  - persistent mode : `"Supprime toutes les annotations sur la frame courante"` (current text)
  - ephemeral mode : `"Efface les traits éphémères vivants"`
- Click handler in `app.py` routes to `manager.clear_all()` or `store.clear_frame(current)` based on mode.

## 7. Preferences

Single new key in `preferences.py` :

```python
@property
def ephemeral_duration_preset(self) -> int:
    """Index of the active ephemeral fade preset.
    0 = court (~2s), 1 = moyen (~5s, default), 2 = long (~10s).
    """
    try:
        v = int(self._s.value("ephemeral/duration_preset", 1))
    except (TypeError, ValueError):
        return 1
    return v if v in (0, 1, 2) else 1

@ephemeral_duration_preset.setter
def ephemeral_duration_preset(self, value: int) -> None:
    if value not in (0, 1, 2):
        return  # silent reject, consistent with the rest of the file
    self._s.setValue("ephemeral/duration_preset", int(value))
```

Note : we do **not** persist whether ephemeral mode itself is on/off across sessions. It's a transient state during a presentation — re-opening the app should land in the default persistent mode. (The original v0.4.0 spec already chose to persist the toolbar's float position and visibility but not the active tool, for the same reason.)

## 8. Testing strategy

Six layers, ~30-40 new tests, target count 473 → ~510.

### 8.1 Pure helpers (no Qt) — `tests/unit/test_ephemeral_alpha.py`

Tests `alpha_at(birth_ts, now_ts, duration_s)` :
- Birth = full alpha. End-of-life = zero. Mid-life = halfway.
- Beyond duration → clamp 0. Before birth (clock jitter) → clamp 1.
- `duration_s == 0` → instant 0.
- Negative age (clock jump back) → clamp 1. Resilient to jitter.

### 8.2 Manager (pytest-qt + monkeypatched clock) — `tests/unit/test_ephemeral_manager.py`

Uses a `_FakeClock` (same pattern as `test_runtime_monitor.py` for the watchdog) :
- `add()` starts the timer when list was empty, emits `repaint_needed`.
- Multiple consecutive `add()`s keep the timer running.
- Virtual time `+ duration + epsilon` → next tick removes expired stroke.
- `kill_last()` removes the latest, returns `True` ; on empty list, `False`.
- `clear_all()` empties everything, returns count, stops timer.
- `set_duration()` recomputes alphas for already-live strokes on next read.
- `live_strokes_with_alpha()` returns immutable tuple, insertion order preserved.
- Timer auto-stops when last stroke expires (no idle polling).
- `has_live_strokes()` consistent with reality.

### 8.3 Overlay routing (pytest-qt + QWidget) — `tests/unit/test_overlay_ephemeral.py`

Real overlay + real manager + mocked store :
- Persistent mode + release → `store.add_stroke()` called, `manager.add()` not called.
- Ephemeral mode + release → `manager.add()` called, `store.add_stroke()` not called.
- Press persistent + toggle on mid-drag + release → goes to store (press-time snapshot wins).
- Press ephemeral + toggle off mid-drag + release → goes to manager (idem).
- Ephemeral mode + eraser tool → eraser disabled, no capture (auto-switched).
- `paintEvent` reads `manager.live_strokes_with_alpha()` when manager has strokes.

### 8.4 Toolbar UI (pytest-qt + QWidget) — `tests/unit/test_toolbar_ephemeral.py`

- Click `👻` → emits `ephemeral_mode_changed(True)` ; second click → `(False)`.
- Toolbar border styleSheet contains cyan accent in on-state, default in off-state.
- 3-preset row invisible in off-state, visible in on-state.
- Click on preset 2 → emits `ephemeral_duration_changed(5.0)`.
- Default preset = index 1 (5s) at boot if no pref.
- Pen glyph : `_pen_btn.text() == "✏️"` in normal mode, `"👻"` in ephemeral.
- Eraser disabled in ephemeral mode.
- Activating mode while eraser is the active tool → tool auto-switched to `NONE`.

### 8.5 Preferences — extend `tests/unit/test_preferences.py`

- `ephemeral_duration_preset` getter returns 1 by default, range [0, 2].
- Setter validates the 3 values ; any other → silently ignored.
- Round-trip via QSettings : set → new instance → read.

### 8.6 App-level smoke (optional) — `test_app_ephemeral_wiring.py`

- On `App()` creation, manager exists and is connected to overlay + toolbar.
- Toggle `👻` → `overlay._ephemeral_mode` flipped.
- Sequence change → `manager.clear_all()` called.

## 9. Out of scope

The following were considered and explicitly **not** included in v0.4.1 :
- Hold-then-fade alpha curves (linear is enough).
- Expiration flash / animation effect ("about to disappear").
- Per-stroke duration override (all strokes share the global preset).
- Variable curve per stroke (e.g. faster fade for thicker strokes).
- A second palette dedicated to ephemeral (the fade itself signals).
- Persistence of "ephemeral mode was on at last close" (transient state by design).
- Snapshot-to-image feature (export current view + ephemeral overlays as a still). Possible follow-up.
- Pen tablet pressure (already noted as out of scope in v0.4.0, unchanged here).

## 10. Migration & compatibility

- **Sidecar JSON schema** : unchanged (still `schema_version=1`, with the `frames` and `comments` sub-trees from v0.4.0). Ephemeral strokes never touch the sidecar.
- **Sidecar files written by v0.4.0** : load identically in v0.4.1.
- **Sidecar files written by v0.4.1** : load identically in v0.4.0 (no new fields).
- **Preferences file** : the new `ephemeral/duration_preset` key is read only when needed and ignored by older versions. Forward and backward compatible.

## 11. Acceptance checklist

- [ ] `EphemeralStrokeManager` with the 7-method public API documented above.
- [ ] `alpha_at()` pure helper passing all listed test cases.
- [ ] `AnnotationOverlay` routing to manager-or-store with press-time snapshot.
- [ ] `AnnotationToolbar` with `👻` toggle, 3-preset bar, pen glyph swap, eraser disabling.
- [ ] `app.py` wiring : toolbar signals → overlay/manager, sequence-change clear, preset persistence.
- [ ] `preferences.py` with `ephemeral_duration_preset` round-trip.
- [ ] Keyboard `G` shortcut wired and toggling the mode.
- [ ] All listed test files green ; total test count 473 → ~510.
- [ ] No new dependency in `pyproject.toml`.
- [ ] Sidecar JSON files round-trip between v0.4.0 and v0.4.1 unchanged.
- [ ] Version bumped to 0.4.1 in `pyproject.toml` and `__init__.py`.
- [ ] `dist/RELEASE_NOTES_v0.4.1.md` written before release build.
