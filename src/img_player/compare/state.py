"""Compare-mode state (which two layers + which mode + seam value).

Held on the app singleton (``app._compare_state``) and edited from
two places: the :class:`~img_player.ui.compare_band.CompareBand`
widget (UI) and a handful of keyboard shortcuts wired in
:class:`~img_player.ui.main_window.MainWindow`. The app's
``_on_frame_changed`` reads the current state to decide whether to
hijack the GL upload with a custom A/B composite.

Pure data — no Qt, no numpy. Lives outside the UI tree so the
compose helper (which runs on the GL upload path) can be unit-tested
with plain dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Mode tokens — kept as plain strings so QSettings round-trip is
# trivial and the values can compare cheaply in signal slots.
#
# Three blend modes: vertical wipe (left A / right B), horizontal
# wipe (top A / bottom B), and linear opacity blend. The "swap"
# behaviour (= show full B) lives outside the modes as a separate
# always-visible toggle (``swap_showing_b``) so the user can A/B
# preview at any time without leaving their chosen blend.
MODE_VERTICAL = "vertical"
MODE_HORIZONTAL = "horizontal"
MODE_OPACITY = "opacity"

COMPARE_MODES: tuple[str, ...] = (
    MODE_VERTICAL, MODE_HORIZONTAL, MODE_OPACITY,
)

# Default mode at compare-mode entry. Vertical split — the most
# common A/B review pattern (left vs right at a draggable seam).
# The user can switch to horizontal / opacity from the band.
DEFAULT_MODE = MODE_VERTICAL

# Default seam / opacity = 50% — the most common starting position
# for a wipe ("show me half-and-half"). Slider lives in [0, 1].
DEFAULT_SEAM = 0.5


@dataclass
class CompareState:
    """Snapshot of the two-layer compare overlay.

    ``enabled`` is the master switch — every other field is only
    meaningful when ``True``. The viewer's frame refresh hook checks
    ``enabled and layer_a_id and layer_b_id`` before doing any
    extra decode work; partially-configured states (one dropdown
    picked, the other still empty) fall through to the normal
    composite path.

    ``mode`` is one of :data:`COMPARE_MODES`. ``seam`` is in [0, 1]:

    * Vertical wipe: split position from left to right.
    * Horizontal wipe: split position from top to bottom.
    * Opacity: 0 = pure A, 1 = pure B. Linear ramp.

    ``swap_showing_b`` is a separate always-visible toggle that
    overrides the blend entirely when ``True`` and shows full B —
    the "preview B in isolation" gesture used to A/B-compare two
    nearly-identical plates.
    """

    enabled: bool = False
    layer_a_id: str | None = None
    layer_b_id: str | None = None
    mode: str = DEFAULT_MODE
    seam: float = DEFAULT_SEAM
    swap_showing_b: bool = False
    # Extra fields can land here without breaking session compat:
    # session save/load round-trips this dataclass via asdict / dict
    # construction, missing keys fall back to the dataclass defaults.
    extra: dict[str, object] = field(default_factory=dict)

    def is_active(self) -> bool:
        """True when the overlay should hijack the GL upload.

        ``enabled`` alone isn't enough — we also need both layers
        picked, otherwise there's nothing to compose against.
        """
        return (
            self.enabled
            and self.layer_a_id is not None
            and self.layer_b_id is not None
        )

    def with_seam(self, value: float) -> "CompareState":
        """Return a new state with ``seam`` clamped to [0, 1]."""
        clamped = max(0.0, min(1.0, float(value)))
        return CompareState(
            enabled=self.enabled,
            layer_a_id=self.layer_a_id,
            layer_b_id=self.layer_b_id,
            mode=self.mode,
            seam=clamped,
            swap_showing_b=self.swap_showing_b,
            extra=dict(self.extra),
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly dump for session persistence."""
        return {
            "enabled": self.enabled,
            "layer_a_id": self.layer_a_id,
            "layer_b_id": self.layer_b_id,
            "mode": self.mode,
            "seam": self.seam,
            "swap_showing_b": self.swap_showing_b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CompareState":
        """Reverse of :meth:`to_dict`. Unknown / malformed values fall
        back to defaults — a stale session never crashes the load.

        Older sessions may have ``mode = "swap"`` (retired in v1.2.1
        when swap moved to a standalone toggle). Treat as the new
        default mode + ``swap_showing_b=True`` so the visual result
        matches what the user saved.
        """
        raw_mode = str(data.get("mode", DEFAULT_MODE))
        legacy_swap = raw_mode == "swap"
        mode = DEFAULT_MODE if legacy_swap else raw_mode
        if mode not in COMPARE_MODES:
            mode = DEFAULT_MODE
        try:
            seam = float(data.get("seam", DEFAULT_SEAM))
        except (TypeError, ValueError):
            seam = DEFAULT_SEAM
        seam = max(0.0, min(1.0, seam))
        return cls(
            enabled=bool(data.get("enabled", False)),
            layer_a_id=(
                str(data["layer_a_id"]) if data.get("layer_a_id") else None
            ),
            layer_b_id=(
                str(data["layer_b_id"]) if data.get("layer_b_id") else None
            ),
            mode=mode,
            seam=seam,
            swap_showing_b=bool(data.get("swap_showing_b", legacy_swap)),
        )
