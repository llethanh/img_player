"""Save / load a :class:`LayerStack` to/from a ``.session`` JSON file.

A session file is a portable snapshot of the multi-layer state:

* every loaded layer (by sequence path on disk)
* its trim (``layer_in`` / ``layer_out``) + offset
* its visibility + name
* its per-layer channel selection / layout / labels-visible
* the focused layer id (so the panel comes back highlighting the
  same row)

The format is plain JSON with a ``"version"`` field so future
schema bumps stay back-compat. Sequences are stored by their
*directory + base_name + extension + padding* so the loader can
re-scan from disk and pick up files that may have moved or grown
(new frames added after the session was saved).

Loading:
1. Parse the JSON.
2. For each layer entry, scan the sequence on disk via the
   existing :func:`scan` helper so the SequenceInfo reflects the
   current file system (mtimes, frame range).
3. Build a :class:`Layer` with the persisted trim / offset /
   visibility / name / channel state, fall back to defaults for
   anything missing.
4. Replace the live LayerStack with the rebuilt layers.
5. Re-establish focus.

Errors during a per-layer scan don't abort the load — the layer
is simply skipped with a status message so the user gets back
*something*.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from img_player.layers.models import Layer
from img_player.layers.stack import LayerStack
from img_player.sequence.channels import ChannelGroup, ChannelSelection
# ``enrich_with_header`` is the canonical impl shared with the live-
# load flow (``ImgPlayerApp._enrich_with_header``) — single source of
# truth so behaviour stays in sync across both paths.
from img_player.sequence.scanner import enrich_with_header as _enrich_with_header
from img_player.sequence.scanner import scan

log = logging.getLogger(__name__)

SESSION_VERSION = 3
SESSION_EXTENSION = ".session"

# Versions we know how to read. v1 lacks the top-level ``color_state``
# block (loader leaves the panel untouched). v2 lacks per-layer still
# fields (``is_still`` / ``still_hold_frames``) — loader treats every
# saved layer as a sequence layer, which is the correct fallback
# (older sessions never had stills). v3 adds the still fields.
_READABLE_VERSIONS = {1, 2, 3}


# ----------------------------------------------------------------- Schema


@dataclass
class _SessionLayer:
    """Serialisable view of a Layer — what we round-trip through JSON."""

    id: str
    name: str
    sequence_directory: str
    sequence_base_name: str
    sequence_extension: str
    sequence_padding: int
    layer_in: int
    layer_out: int
    offset: int
    visible: bool
    # Channel state — stored flat so JSON readers don't have to
    # mirror our ChannelSelection class shape. ``channel_tile_*``,
    # ``channel_layout_mode`` and ``channel_labels_visible`` were
    # written by older sessions (contact-sheet feature, retired in
    # v1.2). Kept readable on load (silently dropped) but no longer
    # written.
    channel_active_label: str = ""
    channel_active_channels: list[str] = field(default_factory=list)
    source_colorspace: str | None = None
    exposure: float = 0.0
    gamma: float = 1.0
    # Per-layer transparency model (v1.0): does this layer compose
    # with alpha over what's beneath, and is the source's alpha
    # encoded straight or premultiplied? Both default ``False`` so
    # legacy sessions (no key) round-trip into the safe "topmost-
    # wins, premult" combination.
    alpha_composite: bool = False
    alpha_is_straight: bool = False
    # Still-image flag + hold duration + concrete file path. Stills
    # may carry filenames without a numeric pattern (``slate.png``,
    # ``ref.exr``) which the standard sequence-rebuild path can't
    # reconstruct from base_name + padding alone — store the explicit
    # filename so the loader can find it back. ``False`` / ``1`` /
    # ``""`` for legacy sequence layers — the loader treats absence
    # of these keys as "this is a normal sequence layer" (v2
    # backward-compat).
    is_still: bool = False
    still_hold_frames: int = 1
    still_filename: str = ""


# ----------------------------------------------------------------- Color state


@dataclass
class ColorState:
    """Snapshot of the global Color panel (OCIO triple + viewing tweaks).

    Stored at the session level rather than per-layer because these
    are *viewing* parameters: they describe how the user wants the
    composited image to look on their monitor, independent of which
    layer is focused. Saving them in the session lets a re-open
    restore the look the user shipped with — without it the player
    keeps whatever display/view the last sequence used, which is
    often wrong (e.g. user reviewed a Rec709 deliverable, then
    opens an ACEScg WIP session and sees it through Rec709).

    All fields default to ``None`` / ``0`` so the loader can always
    instantiate a "do nothing" ColorState when the JSON predates
    this feature (session v1).
    """

    source_colorspace: str | None = None
    display: str | None = None
    view: str | None = None
    exposure: float = 0.0
    gamma: float = 1.0


# ----------------------------------------------------------------- Save


def save_session(
    stack: LayerStack, path: Path, *,
    color_state: ColorState | None = None,
    compare_state: dict[str, Any] | None = None,
) -> None:
    """Serialise ``stack`` to ``path`` as JSON.

    Overwrites any existing file at the path. The ``.session``
    extension is added if missing.

    ``color_state`` captures the global Color panel triple
    (source / display / view + exposure + gamma) so opening the
    session restores the same viewing look. Pass ``None`` to omit
    the block — the loader treats a missing block as "leave the
    panel as is" (v1 backward-compat).
    """
    if path.suffix != SESSION_EXTENSION:
        path = path.with_suffix(SESSION_EXTENSION)
    layers_payload: list[dict[str, Any]] = []
    for layer in stack.layers():
        sl = _SessionLayer(
            id=layer.id,
            name=layer.name,
            sequence_directory=str(layer.sequence.directory),
            sequence_base_name=layer.sequence.base_name,
            sequence_extension=layer.sequence.extension,
            sequence_padding=layer.sequence.padding,
            layer_in=layer.layer_in,
            layer_out=layer.layer_out,
            offset=layer.offset,
            visible=layer.visible,
            source_colorspace=layer.source_colorspace,
            exposure=layer.exposure,
            gamma=layer.gamma,
            alpha_composite=layer.alpha_composite,
            alpha_is_straight=layer.alpha_is_straight,
            is_still=layer.is_still,
            still_hold_frames=layer.still_hold_frames,
            still_filename=(
                layer.sequence.frames[0].path.name if layer.is_still else ""
            ),
        )
        sel = layer.channel_selection
        if sel is not None:
            sl.channel_active_label = sel.active.label
            sl.channel_active_channels = list(sel.active.channels)
        layers_payload.append(asdict(sl))
    payload: dict[str, Any] = {
        "version": SESSION_VERSION,
        "focused_id": stack.focused_id,
        "layers": layers_payload,
    }
    if color_state is not None:
        payload["color_state"] = asdict(color_state)
    if compare_state is not None:
        # Compare state is already a JSON-friendly dict (built by
        # the caller via ``CompareState.to_dict``), so we just embed
        # it. Older session readers ignore unknown top-level keys.
        payload["compare_state"] = compare_state
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("[session] saved %d layers to %s", len(layers_payload), path)


# ----------------------------------------------------------------- Load


@dataclass
class LoadResult:
    """Tally of what happened during :func:`load_session`."""

    loaded: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    # Color panel snapshot the session shipped with (v2+). ``None``
    # for v1 sessions / sessions saved without a color_state — the
    # caller leaves the panel state untouched in that case.
    color_state: ColorState | None = None
    # Compare-mode dict (v1.2+). ``None`` when the session predates
    # the feature or compare-mode was off at save time. The caller
    # passes this through ``CompareState.from_dict`` to rehydrate.
    compare_state: dict[str, object] | None = None


def load_session(stack: LayerStack, path: Path) -> LoadResult:
    """Replace ``stack`` with the layers described in ``path``.

    Each layer's sequence is rescanned from disk via the existing
    :func:`scan` helper. Sequences whose file doesn't exist
    anymore or whose pattern can't be matched are silently skipped
    (an entry is added to :class:`LoadResult.errors` so the caller
    can surface it). All other layers load with their saved trim
    / offset / channel state.
    """
    result = LoadResult()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload_version = payload.get("version")
    if payload_version not in _READABLE_VERSIONS:
        result.errors.append(
            f"Unknown session version {payload_version!r}; "
            f"expected one of {sorted(_READABLE_VERSIONS)}"
        )
        return result
    # Color state — only present from v2 onwards. v1 sessions get a
    # ``None`` color_state and the caller leaves the panel alone.
    color_payload = payload.get("color_state")
    if color_payload is not None:
        result.color_state = ColorState(
            source_colorspace=color_payload.get("source_colorspace"),
            display=color_payload.get("display"),
            view=color_payload.get("view"),
            exposure=float(color_payload.get("exposure", 0.0)),
            gamma=float(color_payload.get("gamma", 1.0)),
        )
    # Compare-mode state (v1.2+). Stays as a plain dict here; the
    # caller dispatches it through ``CompareState.from_dict`` so
    # this module doesn't need to import the compare package.
    compare_payload = payload.get("compare_state")
    if isinstance(compare_payload, dict):
        result.compare_state = compare_payload
    layer_entries = payload.get("layers", [])
    rebuilt_focused: str | None = None
    saved_focused = payload.get("focused_id", "")
    # Wrap the entire load (clear-existing + add-N) into a single
    # undo step so Ctrl+Z after "Open session" reverts to whatever
    # the user had before, not just the last layer that was added.
    with stack.batch():
        # Empty the live stack first. We do it before scanning so the
        # cache invalidates only once and the panel rebuilds cleanly.
        for existing in stack.layers():
            stack.remove(existing.id)

        for entry in layer_entries:
            try:
                layer = _rebuild_layer(entry)
            except Exception as err:
                result.skipped += 1
                result.errors.append(str(err))
                log.warning("[session] skipped layer entry: %s", err)
                continue
            # Add at bottom (position = len) so the list order matches
            # the saved order (top-most first in JSON → top of stack).
            stack.add(layer, position=len(stack))
            result.loaded += 1
            if layer.id == saved_focused:
                rebuilt_focused = layer.id
        if rebuilt_focused is not None:
            stack.set_focus(rebuilt_focused)
    return result


def _rebuild_layer(entry: dict[str, Any]) -> Layer:
    """Reconstruct a :class:`Layer` from one JSON entry.

    Re-scans the sequence directory so the SequenceInfo carries
    fresh mtimes + the current frame range (sequences may have
    grown since the session was saved). Raises if the path can't
    be resolved.

    Still layers (``is_still=True``) take a separate path: the
    explicit ``still_filename`` is rescanned as a single file so
    layers whose filename has no numeric pattern (slates, lookdev
    refs) round-trip without trying to glob a sequence pattern that
    doesn't exist.
    """
    directory = Path(entry["sequence_directory"])
    if not directory.exists():
        raise FileNotFoundError(
            f"Sequence directory missing: {directory}"
        )
    # Still path — short-circuit before the directory-scan logic
    # because ``scan(directory)`` returns the largest sequence in
    # the dir, which would silently switch a still entry to a
    # full-sequence layer if the still happens to live next to a
    # sequence (common in delivery folders).
    if bool(entry.get("is_still", False)):
        filename = str(entry.get("still_filename", "")).strip()
        if not filename:
            raise ValueError(
                "Still session entry missing ``still_filename``"
            )
        still_path = directory / filename
        if not still_path.exists():
            raise FileNotFoundError(
                f"Still file missing: {still_path}"
            )
        # ``scan(file)`` resolves the single image into a 1-frame
        # SequenceInfo (handles non-pattern names since the
        # scanner gained a still-fallback in v1.2).
        still_seq = scan(still_path, probe=False)
        if still_seq.frames and (not still_seq.width or not still_seq.height):
            still_seq = _enrich_with_header(still_seq)
        hold = max(1, int(entry.get("still_hold_frames", 1)))
        layer = Layer.from_still(
            still_seq,
            hold_frames=hold,
            offset=int(entry.get("offset", 0)),
            name=entry.get("name") or still_path.name,
        )
        layer.id = entry.get("id", layer.id)
        layer.visible = bool(entry.get("visible", True))
        layer.source_colorspace = entry.get("source_colorspace")
        layer.exposure = float(entry.get("exposure", 0.0))
        layer.gamma = float(entry.get("gamma", 1.0))
        layer.alpha_composite = bool(entry.get("alpha_composite", False))
        if "alpha_is_straight" in entry:
            layer.alpha_is_straight = bool(entry["alpha_is_straight"])
        # Stills don't carry channel selections in the same way
        # multi-AOV EXR sequences do, but the schema accepts them
        # uniformly — restore on a best-effort basis so a still
        # with a saved per-channel preview round-trips.
        active_label = entry.get("channel_active_label", "")
        active_channels = entry.get("channel_active_channels", [])
        if active_label and active_channels:
            active = ChannelGroup(
                label=active_label, channels=tuple(active_channels),
            )
            layer.channel_selection = ChannelSelection(active=active)
        return layer
    seq = scan(directory, probe=False)
    # ``scan(probe=False)`` skips the per-file header read (kept fast
    # for slow filesystems like Drive Stream), so ``seq.width`` /
    # ``seq.height`` come back as None. The rest of the app reads
    # those for things like the missing-frame placeholder size — a
    # 512×512 fallback was kicking in for session-loaded layers,
    # mismatching the real image dimensions. Probe the first frame's
    # header here so dimensions ride along with the rebuilt layer.
    if seq.frames and (not seq.width or not seq.height):
        seq = _enrich_with_header(seq)
    layer = Layer.from_sequence(
        seq,
        offset=int(entry.get("offset", seq.first_frame)),
        name=entry.get("name") or seq.display_pattern(),
    )
    # Override defaults with the saved trim / visibility / state.
    layer.id = entry.get("id", layer.id)
    layer.layer_in = int(entry.get("layer_in", seq.first_frame))
    layer.layer_out = int(entry.get("layer_out", seq.last_frame))
    layer.visible = bool(entry.get("visible", True))
    layer.source_colorspace = entry.get("source_colorspace")
    layer.exposure = float(entry.get("exposure", 0.0))
    layer.alpha_composite = bool(entry.get("alpha_composite", False))
    if "alpha_is_straight" in entry:
        layer.alpha_is_straight = bool(entry["alpha_is_straight"])
    # else: keep whatever ``Layer.from_sequence`` auto-detected from
    # the file extension (legacy sessions predate this field).
    layer.gamma = float(entry.get("gamma", 1.0))
    # Channel selection — only rebuild if both active label + at
    # least one channel survived. Older sessions also stored
    # ``channel_tile_*`` (contact-sheet tiles, retired in v1.2);
    # those keys are silently ignored.
    active_label = entry.get("channel_active_label", "")
    active_channels = entry.get("channel_active_channels", [])
    if active_label and active_channels:
        active = ChannelGroup(
            label=active_label, channels=tuple(active_channels),
        )
        layer.channel_selection = ChannelSelection(active=active)
    return layer


