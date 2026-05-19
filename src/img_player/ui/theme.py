"""Design system for img_player — "Studio Dark" charter v2 (2026-Q2).

Single source of truth for all colors, typography, spacing, and geometry.
Never hard-code hex values elsewhere in the UI — always import from here.

This module went through a major refresh in Q2 2026 that introduced new
semantic tokens (BG_STRIP / BG_ROW / BG_SUNKEN / BG_TRACK, slider teal,
accent tints at multiple alpha levels, separated rgba border tokens) and
updated the button QSS to match the new charter. The OLD token names
(BG_RAISED, BORDER_DEFAULT, ACCENT, ACCENT_DIM, etc.) are kept as
backward-compat aliases that point at the new values — every legacy
widget that imports them keeps working, just with subtly updated
colours.

Quick reference::

    from img_player.ui.theme import C, F, S, G, build_stylesheet

    app.setStyleSheet(build_stylesheet())   # apply at startup
    painter.fillRect(rect, C.BG_BASE)       # QColor directly
    label.setFont(F.ui())                   # QFont helper
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont

# ---------------------------------------------------------------------------
# Colors — QColor instances
# ---------------------------------------------------------------------------
#
# Naming convention (new):
#   BG_*       — surfaces, from deepest to most raised
#   BORDER_*   — hairlines, semantic levels (subtle / default / strong)
#   T_PRI/SEC/DIM/TAG — text colours by emphasis
#   ACC*       — orange accent + its tints / glows / borders
#   SLIDER_*   — teal slider colours (annotation palette only)
#   PILL_*     — per-pill layer toggle colours
#   ...semantic (DANGER, RUNNING) for state-driven colours
#
# Legacy names (BG_RAISED, BORDER_SUBTLE, TEXT_PRIMARY, ACCENT, …) are
# preserved as aliases at the bottom of each section so the existing
# widget code keeps importing them without a search-and-replace pass.


class C:
    # ---- Surfaces ---------------------------------------------------------
    BG_DEEP    = QColor("#111114")  # app background, status bar
    BG_BASE    = QColor("#1A1A1D")  # default panel
    BG_ROW     = QColor("#202024")  # focused layer row
    BG_STRIP   = QColor("#242428")  # toolbar background (transport, top-right, …)
    BG_SURFACE = QColor("#2A2A2F")  # button face at rest
    BG_HOVER   = QColor("#34343A")  # button face on hover
    BG_SUNKEN  = QColor("#0B0B0E")  # wells (segmented track, slider track)
    BG_TRACK   = QColor("#0F1013")  # timeline + layer-bar track

    # Legacy aliases — same QColor instance referenced under both names
    # so QPainter widgets that pass C.BG_RAISED keep working unchanged.
    BG_RAISED  = BG_STRIP            # was #242428, now points at BG_STRIP
    BG_SELECT  = BG_ROW              # was #3A3A40, now points at BG_ROW

    # ---- Borders (semi-transparent white hairlines) -----------------------
    # Stored as RGBA QColors so QPainter strokes reveal the underlying
    # surface through the hairline (the brief's intent: borders are
    # "tints" not solid lines, which keeps the chrome quiet).
    BORDER_SUB = QColor(255, 255, 255, int(0.05 * 255))   # 13/255
    BORDER_DEF = QColor(255, 255, 255, int(0.10 * 255))   # 26/255
    BORDER_STR = QColor(255, 255, 255, int(0.18 * 255))   # 46/255
    BORDER_ACC_DEEP = QColor("#7A4210")                    # header info strip

    # Legacy aliases
    BORDER_SUBTLE  = BORDER_SUB
    BORDER_DEFAULT = BORDER_DEF
    BORDER_STRONG  = BORDER_STR

    # ---- Text -------------------------------------------------------------
    T_PRI = QColor("#E4E4E6")  # body + active icons
    T_SEC = QColor("#8C8C92")  # secondary labels, mono captions
    T_DIM = QColor("#525258")  # disabled
    T_TAG = QColor("#6A6A70")  # pill letter at rest

    # Legacy aliases
    TEXT_PRIMARY   = T_PRI
    TEXT_SECONDARY = T_SEC
    TEXT_DISABLED  = T_DIM

    # ---- Accent (ACES orange) --------------------------------------------
    ACC        = QColor("#E8901C")  # canonical accent
    ACC_BRIGHT = QColor("#F5A830")  # hover / lit state
    ACC_DEEP   = QColor("#7A4210")  # darker shade for borders on tints

    # Tinted variants — pre-baked alpha levels so QPainter / QSS don't
    # have to recompute. The number is the alpha percentage (×100).
    ACC_FILL      = QColor(0xE8, 0x90, 0x1C, int(0.78 * 255))   # layer bar fill
    ACC_GLOW      = QColor(0xE8, 0x90, 0x1C, int(0.20 * 255))   # soft outer glow
    ACC_TINT_10   = QColor(0xE8, 0x90, 0x1C, int(0.10 * 255))   # active button bg
    ACC_TINT_14   = QColor(0xE8, 0x90, 0x1C, int(0.14 * 255))   # active toggle bg
    ACC_TINT_30   = QColor(0xE8, 0x90, 0x1C, int(0.30 * 255))   # seam fill
    ACC_BORDER_ON = QColor(0xE8, 0x90, 0x1C, int(0.45 * 255))   # A/B tag border
    ACC_RING      = QColor(0xE8, 0x90, 0x1C, int(0.20 * 255))   # inset glow

    # Legacy aliases
    ACCENT        = ACC
    ACCENT_BRIGHT = ACC_BRIGHT
    ACCENT_DIM    = ACC_DEEP

    # ---- Slider (teal — annotation palette only) -------------------------
    # Distinct from the orange playback accent so a stray slider can't be
    # confused with a "playing" state somewhere.
    SLIDER_THUMB = QColor("#4DB6CC")
    SLIDER_GLOW  = QColor(0x4D, 0xB6, 0xCC, int(0.45 * 255))
    SLIDER_RING  = QColor(0x4D, 0xB6, 0xCC, int(0.55 * 255))

    # ---- Layer pills -----------------------------------------------------
    # Each pill's "on" colour is distinctive so the user can read the
    # layer state at a glance without parsing the letter.
    PILL_T  = QColor("#5DC9D2")  # T   alpha-over (cyan)
    PILL_AS = QColor("#B783D9")  # αS  straight-alpha (violet)
    PILL_M  = QColor("#E64A4A")  # M   mute (red)
    PILL_S  = QColor("#F5A830")  # S   solo (orange)

    # ---- Semantic --------------------------------------------------------
    DANGER  = QColor("#E64A4A")  # Clear button, M-pill, missing frames
    RUNNING = QColor("#5DC46C")  # prefetch in progress, session-OK dot

    # Legacy / specialised
    COLOR_ERROR      = DANGER
    PLAYHEAD         = QColor("#F5AA28")
    PLAYHEAD_OUTLINE = BG_DEEP
    MARKER_IO        = QColor("#DC3C3C")
    RANGE_BAR        = ACC
    CACHE_BAR_BG     = BG_DEEP
    CACHE_BAR        = QColor(0xE8, 0x90, 0x1C, 128)   # ACCENT @ 50%
    CACHE_BAR_BORDER = ACC

    # Timeline tick colours (kept close to the previous values but
    # nudged toward T_SEC / T_DIM so the timeline reads as a quiet
    # ruler rather than a noisy stripe pattern).
    TICK_MINOR = QColor("#3C3C40")
    TICK_MAJOR = QColor("#6A6A70")
    TICK_LABEL = T_SEC


# ---------------------------------------------------------------------------
# Hex / rgba strings — for embedding in QSS
# ---------------------------------------------------------------------------
#
# QSS doesn't understand QColor objects directly — it wants either a
# hex string ("#1A1A1D") or a CSS-style colour function
# ("rgba(255, 255, 255, 0.05)"). This class mirrors :class:`C` but
# returns the string form, kept in sync with the QColor values above.

class H:
    # ---- Surfaces ---------------------------------------------------------
    BG_DEEP    = "#111114"
    BG_BASE    = "#1A1A1D"
    BG_ROW     = "#202024"
    BG_STRIP   = "#242428"
    BG_SURFACE = "#2A2A2F"
    BG_HOVER   = "#34343A"
    BG_SUNKEN  = "#0B0B0E"
    BG_TRACK   = "#0F1013"

    BG_RAISED  = BG_STRIP            # legacy
    BG_SELECT  = BG_ROW              # legacy

    # ---- Borders ----------------------------------------------------------
    BORDER_SUB = "rgba(255, 255, 255, 0.05)"
    BORDER_DEF = "rgba(255, 255, 255, 0.10)"
    BORDER_STR = "rgba(255, 255, 255, 0.18)"
    BORDER_ACC_DEEP = "#7A4210"

    BORDER_SUBTLE  = BORDER_SUB
    BORDER_DEFAULT = BORDER_DEF
    BORDER_STRONG  = BORDER_STR

    # ---- Text -------------------------------------------------------------
    T_PRI = "#E4E4E6"
    T_SEC = "#8C8C92"
    T_DIM = "#525258"
    T_TAG = "#6A6A70"

    TEXT_PRIMARY   = T_PRI
    TEXT_SECONDARY = T_SEC
    TEXT_DISABLED  = T_DIM

    # ---- Accent -----------------------------------------------------------
    ACC        = "#E8901C"
    ACC_BRIGHT = "#F5A830"
    ACC_DEEP   = "#7A4210"

    ACC_FILL      = "rgba(232, 144, 28, 0.78)"
    ACC_GLOW      = "rgba(232, 144, 28, 0.20)"
    ACC_TINT_10   = "rgba(232, 144, 28, 0.10)"
    ACC_TINT_14   = "rgba(232, 144, 28, 0.14)"
    ACC_TINT_30   = "rgba(232, 144, 28, 0.30)"
    ACC_BORDER_ON = "rgba(232, 144, 28, 0.45)"
    ACC_RING      = "rgba(232, 144, 28, 0.20)"

    ACCENT        = ACC
    ACCENT_BRIGHT = ACC_BRIGHT
    ACCENT_DIM    = ACC_DEEP

    # ---- Slider -----------------------------------------------------------
    SLIDER_THUMB = "#4DB6CC"
    SLIDER_GLOW  = "rgba(77, 182, 204, 0.45)"
    SLIDER_RING  = "rgba(77, 182, 204, 0.55)"

    # ---- Layer pills -----------------------------------------------------
    PILL_T  = "#5DC9D2"
    PILL_AS = "#B783D9"
    PILL_M  = "#E64A4A"
    PILL_S  = "#F5A830"

    # ---- Semantic --------------------------------------------------------
    DANGER  = "#E64A4A"
    RUNNING = "#5DC46C"

    # ---- Media-specific --------------------------------------------------
    PLAYHEAD  = "#F5AA28"
    MARKER_IO = "#DC3C3C"
    RANGE_BAR = ACC
    CACHE_BAR = "#38B464"


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
#
# Brief calls for Inter (body) + JetBrains Mono (code / labels). We
# don't bundle the font files — the QSS lists Inter first, falls back
# to Segoe UI then sans-serif. If Inter is installed on the user's
# machine the rendering is pixel-faithful; otherwise the Windows
# default takes over and the layout still works (Segoe UI's metrics
# are close enough that no widget breaks).

class F:
    # Family chains — used by every QFont we build below AND injected
    # into the global QSS as the default font-family for QWidget.
    FAMILY_UI   = "Inter, 'Segoe UI', sans-serif"
    FAMILY_MONO = "'JetBrains Mono', 'Consolas', monospace"
    # Bare family names (without the fallback chain) for QFont
    # constructors that don't accept the comma syntax.
    UI_PRIMARY   = "Inter"
    UI_FALLBACK  = "Segoe UI"
    MONO_PRIMARY = "JetBrains Mono"

    # Size tiers (px). The brief calls out specific px values per role;
    # these constants map onto those roles.
    SIZE_BODY_TITLE = 15   # title
    SIZE_BODY       = 12   # body regular
    SIZE_BODY_SMALL = 11   # body small
    SIZE_MONO_CODE  = 11   # mono code (filenames, frame numbers)
    SIZE_MONO_LABEL = 10   # mono label (LABELS UPPERCASE 0.06em)
    SIZE_MONO_CAPS  = 9    # mono caps (tiny captions 0.18em)
    SIZE_PILL       = 10   # pill letter (T / αS / M / S)
    SIZE_CHANNEL    = 11   # channel pill (R / G / B / A)

    # Legacy aliases for the existing widgets that import SIZE_XS/SM/MD.
    SIZE_XS = SIZE_MONO_CAPS   # 9
    SIZE_SM = SIZE_MONO_LABEL  # 10
    SIZE_MD = SIZE_BODY_SMALL  # 11

    # ---- QFont helpers ---------------------------------------------------

    @staticmethod
    def ui(size: int = SIZE_BODY, bold: bool = False) -> QFont:
        """Body / UI font — Inter preferred, Segoe UI fallback."""
        f = QFont(F.UI_PRIMARY)
        # Build the substitute chain explicitly so Qt picks the
        # fallback if Inter isn't installed.
        f.setFamilies([F.UI_PRIMARY, F.UI_FALLBACK])
        f.setPixelSize(size)
        if bold:
            f.setWeight(QFont.Weight.DemiBold)
        else:
            f.setWeight(QFont.Weight.Medium)
        return f

    @staticmethod
    def title(size: int = SIZE_BODY_TITLE) -> QFont:
        """Title font — Inter SemiBold."""
        f = F.ui(size=size, bold=True)
        f.setWeight(QFont.Weight.DemiBold)
        return f

    @staticmethod
    def mono(size: int = SIZE_MONO_CODE, bold: bool = False) -> QFont:
        """Monospace font — JetBrains Mono preferred."""
        f = QFont(F.MONO_PRIMARY)
        f.setFamilies([F.MONO_PRIMARY, "Consolas", "Menlo"])
        f.setPixelSize(size)
        if bold:
            f.setWeight(QFont.Weight.Bold)
        else:
            f.setWeight(QFont.Weight.Medium)
        return f


# ---------------------------------------------------------------------------
# Spacing — 2 / 4 / 8 / 12 / 16 / 24 grid
# ---------------------------------------------------------------------------
#
# Brief expresses spacings as s.1 .. s.10. We expose them under both
# the numeric brief names (S_2, S_4, …) AND the legacy semantic names
# (XS, SM, MD, LG, XL, XXL) so old code keeps importing what it knows.

class S:
    # New scale (brief §0.2)
    S_2  = 2    # ticks, fine internal borders
    S_4  = 4    # gaps between grouped buttons
    S_6  = 6    # icon ↔ label inside a button
    S_8  = 8    # icon button padding-x, row gap
    S_10 = 10   # toolbar padding-x
    S_12 = 12   # group separator
    S_14 = 14   # padding-x button with label
    S_16 = 16   # section gap, panel padding
    S_20 = 20
    S_24 = 24

    # Legacy aliases (existing widgets) — values nudged 1px in places
    # to match the new scale.
    XS  = S_2
    SM  = S_4
    MD  = S_8
    LG  = S_12
    XL  = S_16
    XXL = S_24


# ---------------------------------------------------------------------------
# Component geometry
# ---------------------------------------------------------------------------
#
# Standardised control sizes per brief §0.5. Old constants kept as
# aliases so existing widgets keep building.

class G:
    # ---- Standard control sizes (brief §0.5) -----------------------------
    CTRL_INPUT_H   = 24    # compact selects (Cols, Rows, Output, Zoom)
    CTRL_BUTTON_H  = 28    # default toolbar button height
    CTRL_ICON_W    = 30    # icon-only button width (28h × 30w roughly square)
    CTRL_PRIMARY_W = 38    # play key (wider primary action)
    CTRL_SEP_W     = 1     # vertical separator width
    CTRL_SEP_H     = 22    # vertical separator height

    # Pill heights (T / αS / M / S) — variant per density.
    PILL_H_LOW    = 14
    PILL_H_HIGH   = 22

    # ---- Radii (brief §0.3) ----------------------------------------------
    RADIUS_SM = 2    # wells (seam track, swatches)
    RADIUS_MD = 3    # default — buttons, inputs, pills, selects
    RADIUS_LG = 6    # panels (annotation floating)
    RADIUS_XL = 8    # premium floating panels

    # ---- Legacy aliases --------------------------------------------------
    BTN_TRANSPORT_W = CTRL_ICON_W      # 30
    BTN_TRANSPORT_H = CTRL_BUTTON_H    # 28
    BTN_TEXT_W      = 32
    INPUT_H         = CTRL_INPUT_H     # 24
    ICON_SIZE       = 18
    TRANSPORT_H     = 38
    TIMELINE_H      = 52


# ---------------------------------------------------------------------------
# Global QSS — generated from the tokens above
# ---------------------------------------------------------------------------
#
# Two layers of selectors:
# 1. **Bare-class rules** — QPushButton, QComboBox, QLineEdit, … get
#    the default styling for the whole app.
# 2. **Object-name overrides** — widgets that want a different variant
#    (primary / danger / toggle / icon-only) set their ``objectName``
#    to one of the agreed names (``btnPrimary``, ``btnDanger``,
#    ``btnToggle``, ``btnIcon``) and pick up the matching ``#name``
#    rule below.
#
# Anything not covered by these defaults is overridden ad-hoc by the
# widget's own ``setStyleSheet`` call (e.g. the seam bar's custom
# track / fill). Keeps the QSS surface small and predictable.


def build_stylesheet() -> str:
    """Return the complete application-wide QSS.

    Apply once at startup::

        app.setStyleSheet(build_stylesheet())

    Subsequent ad-hoc widget styling cascades on top of this baseline
    via ``setStyleSheet`` on individual widgets.
    """
    h = H
    s = S
    g = G
    f = F

    return f"""
/* ── Base ──────────────────────────────────────────────────────────────── */

QWidget {{
    background-color: {h.BG_BASE};
    color: {h.T_PRI};
    font-family: {f.FAMILY_UI};
    font-size: {f.SIZE_BODY}px;
    border: none;
    outline: none;
}}

QMainWindow {{
    background-color: {h.BG_BASE};
}}

/* ── Menu bar ───────────────────────────────────────────────────────────── */

QMenuBar {{
    background-color: {h.BG_STRIP};
    color: {h.T_PRI};
    padding: 2px 0 1px 0;
    min-height: 30px;
    border-bottom: 1px solid {h.BORDER_SUB};
    spacing: 2px;
}}

QMenuBar::item {{
    padding: 3px 10px;
    border-radius: {g.RADIUS_SM}px;
    background: transparent;
}}

QMenuBar::item:selected {{
    background-color: {h.BG_HOVER};
}}

QMenu {{
    background-color: {h.BG_STRIP};
    border: 1px solid {h.BORDER_DEF};
    padding: {s.S_2}px;
    border-radius: {g.RADIUS_MD}px;
}}

QMenu::item {{
    padding: 4px 24px 4px 12px;
    border-radius: {g.RADIUS_SM}px;
    background: transparent;
}}

QMenu::item:selected {{
    background-color: {h.BG_HOVER};
}}

QMenu::item:disabled {{
    color: {h.T_DIM};
}}

QMenu::separator {{
    height: 1px;
    background: {h.BORDER_SUB};
    margin: {s.S_2}px {s.S_8}px;
}}

/* ── Status bar ─────────────────────────────────────────────────────────── */

QStatusBar {{
    background-color: {h.BG_DEEP};
    color: {h.T_SEC};
    border-top: 1px solid {h.BORDER_SUB};
    font-size: {f.SIZE_BODY_SMALL}px;
    padding: 0 {s.S_14}px;
}}

QStatusBar::item {{
    border: none;
}}

/* ── Dock ───────────────────────────────────────────────────────────────── */

QDockWidget {{
    background-color: {h.BG_STRIP};
    color: {h.T_PRI};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {h.BG_STRIP};
    padding: {s.S_4}px {s.S_8}px;
    border-bottom: 1px solid {h.BORDER_SUB};
    font-weight: 600;
    color: {h.T_DIM};
    font-size: {f.SIZE_MONO_CAPS}px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ── Tab widget ─────────────────────────────────────────────────────────── */

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {h.BORDER_SUB};
    background-color: {h.BG_STRIP};
}}

QTabBar {{
    background-color: {h.BG_BASE};
}}

QTabBar::tab {{
    background-color: transparent;
    color: {h.T_SEC};
    padding: 5px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: {f.SIZE_MONO_LABEL}px;
}}

QTabBar::tab:selected {{
    color: {h.T_PRI};
    border-bottom: 2px solid {h.ACC};
    background-color: {h.BG_STRIP};
}}

QTabBar::tab:hover:!selected {{
    background-color: {h.BG_HOVER};
    color: {h.T_PRI};
}}

/* ── Checkboxes + radio buttons ─────────────────────────────────────────── */

QCheckBox::indicator,
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    background-color: {h.BG_SURFACE};
    border: 1px solid {h.BORDER_DEF};
}}

QCheckBox::indicator {{
    border-radius: 2px;
}}

QRadioButton::indicator {{
    border-radius: 7px;
}}

QCheckBox::indicator:hover,
QRadioButton::indicator:hover {{
    border-color: {h.BORDER_STR};
}}

QCheckBox::indicator:checked,
QRadioButton::indicator:checked {{
    background-color: {h.ACC};
    border-color: {h.ACC};
}}

QCheckBox::indicator:disabled,
QRadioButton::indicator:disabled {{
    background-color: {h.BG_STRIP};
    border-color: {h.BORDER_SUB};
}}

/* ── Buttons — default (matches brief §1) ───────────────────────────────── */

QPushButton {{
    background-color: {h.BG_SURFACE};
    color: {h.T_PRI};
    border: 1px solid {h.BORDER_DEF};
    border-radius: {g.RADIUS_MD}px;
    padding: 0 {s.S_12}px;
    min-height: {g.CTRL_BUTTON_H}px;
    max-height: {g.CTRL_BUTTON_H}px;
    font-family: {f.FAMILY_UI};
    font-size: {f.SIZE_BODY}px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {h.BG_HOVER};
    border-color: {h.BORDER_STR};
}}

QPushButton:pressed {{
    background-color: #1F1F22;
    border-color: {h.BORDER_DEF};
}}

QPushButton:disabled {{
    color: {h.T_DIM};
    border-color: {h.BORDER_SUB};
    background-color: {h.BG_STRIP};
}}

QPushButton:focus {{
    border: 2px solid {h.ACC_BRIGHT};
}}

QPushButton:checked {{
    background-color: {h.ACC_TINT_10};
    border: 1px solid {h.ACC};
    color: {h.ACC_BRIGHT};
}}

/* Icon-only variant: no padding, fixed 30px width.
   Set objectName("btnIcon") to opt in. */
QPushButton#btnIcon {{
    padding: 0;
    min-width: {g.CTRL_ICON_W}px;
    max-width: {g.CTRL_ICON_W}px;
}}

/* Toggle variant: same as default but with a stronger active state.
   Used for the Auto smart / Show labels / Compare buttons.
   Set objectName("btnToggle") to opt in. */
QPushButton#btnToggle:checked {{
    background-color: {h.ACC_TINT_14};
    border: 1px solid {h.ACC};
    color: {h.ACC_BRIGHT};
}}

/* Large icon button — the top toolbar (compare / contact-sheet /
   reload / export). A notch bigger than #btnIcon (34×32 vs 30×28)
   so the bar's primary controls are easier to hit. Covers plain
   and checkable buttons; the :checked rule only fires on toggles. */
QPushButton#btnTopBar {{
    padding: 0;
    min-width: 34px;
    max-width: 34px;
    min-height: 32px;
    max-height: 32px;
}}

QPushButton#btnTopBar:checked {{
    background-color: {h.ACC_TINT_14};
    border: 1px solid {h.ACC};
    color: {h.ACC_BRIGHT};
}}

/* Primary variant — used by the play key. Wider footprint and a
   subtle orange gradient. Set objectName("btnPrimary") to opt in. */
QPushButton#btnPrimary {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {h.ACC_BRIGHT}, stop:1 {h.ACC}
    );
    color: #1A1206;
    border: 1px solid rgba(0, 0, 0, 0.5);
    min-width: {g.CTRL_PRIMARY_W}px;
    max-width: {g.CTRL_PRIMARY_W}px;
    font-weight: 600;
}}

QPushButton#btnPrimary:hover {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFB845, stop:1 {h.ACC_BRIGHT}
    );
}}

QPushButton#btnPrimary:pressed {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #D88828, stop:1 #B86F1A
    );
}}

QPushButton#btnPrimary:checked {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {h.ACC_BRIGHT}, stop:1 {h.ACC}
    );
    color: #1A1206;
    border: 1px solid {h.ACC_BRIGHT};
}}

/* Danger variant — used by the Clear button. Set objectName("btnDanger"). */
QPushButton#btnDanger {{
    background-color: rgba(230, 74, 74, 0.10);
    border: 1px solid rgba(230, 74, 74, 0.55);
    color: #FF8A8A;
}}

QPushButton#btnDanger:hover {{
    background-color: rgba(230, 74, 74, 0.18);
    border-color: rgba(230, 74, 74, 0.65);
}}

QPushButton#btnDanger:pressed {{
    background-color: rgba(230, 74, 74, 0.25);
    color: #FFFFFF;
}}

/* ── Inputs ─────────────────────────────────────────────────────────────── */

QComboBox,
QDoubleSpinBox,
QSpinBox,
QLineEdit {{
    background-color: {h.BG_SURFACE};
    color: {h.T_PRI};
    border: 1px solid {h.BORDER_DEF};
    border-radius: {g.RADIUS_MD}px;
    padding: 0 {s.S_8}px;
    min-height: {g.CTRL_INPUT_H}px;
    max-height: {g.CTRL_INPUT_H}px;
    selection-background-color: {h.ACC_DEEP};
    selection-color: {h.T_PRI};
    font-family: {f.FAMILY_MONO};
    font-size: {f.SIZE_MONO_CODE}px;
}}

QComboBox:hover,
QDoubleSpinBox:hover,
QSpinBox:hover,
QLineEdit:hover {{
    border-color: {h.BORDER_STR};
    background-color: {h.BG_HOVER};
}}

QComboBox:focus,
QDoubleSpinBox:focus,
QSpinBox:focus,
QLineEdit:focus {{
    border-color: {h.ACC};
    color: {h.ACC_BRIGHT};
}}

QComboBox::drop-down {{
    border: none;
    width: 18px;
    subcontrol-origin: padding;
    subcontrol-position: center right;
}}

QComboBox::down-arrow {{
    width: 8px;
    height: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {h.BG_STRIP};
    border: 1px solid {h.BORDER_DEF};
    selection-background-color: {h.BG_HOVER};
    selection-color: {h.T_PRI};
    outline: none;
    padding: {s.S_2}px;
    font-family: {f.FAMILY_MONO};
    font-size: {f.SIZE_MONO_CODE}px;
}}

QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button,
QSpinBox::up-button,
QSpinBox::down-button {{
    background-color: transparent;
    border: none;
    width: 14px;
}}

/* ── Group box ──────────────────────────────────────────────────────────── */

QGroupBox {{
    border: 1px solid {h.BORDER_DEF};
    border-radius: {g.RADIUS_MD + 1}px;
    margin-top: 1.2em;
    padding-top: {s.S_4}px;
    color: {h.T_DIM};
    font-size: {f.SIZE_MONO_CAPS}px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {s.S_8}px;
    padding: 0 {s.S_4}px;
    background-color: {h.BG_STRIP};
}}

/* ── Scroll bars ────────────────────────────────────────────────────────── */

QScrollBar:vertical {{
    background: {h.BG_STRIP};
    width: 6px;
    margin: 0;
    border-radius: 3px;
}}

QScrollBar::handle:vertical {{
    background: {h.BORDER_STR};
    border-radius: 3px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {h.T_DIM};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
    height: 0;
}}

QScrollBar:horizontal {{
    background: {h.BG_STRIP};
    height: 6px;
    margin: 0;
    border-radius: 3px;
}}

QScrollBar::handle:horizontal {{
    background: {h.BORDER_STR};
    border-radius: 3px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {h.T_DIM};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
    width: 0;
}}

/* ── Tooltip ────────────────────────────────────────────────────────────── */

QToolTip {{
    background-color: {h.BG_STRIP};
    color: {h.T_PRI};
    border: 1px solid {h.BORDER_DEF};
    padding: {s.S_2}px {s.S_4}px;
    border-radius: {g.RADIUS_SM}px;
    font-size: {f.SIZE_BODY_SMALL}px;
}}

/* ── Label ──────────────────────────────────────────────────────────────── */

QLabel {{
    background: transparent;
    color: {h.T_SEC};
    border: none;
}}

/* ── Message box / Dialog ──────────────────────────────────────────────── */

QMessageBox {{
    background-color: {h.BG_BASE};
}}

QMessageBox QLabel {{
    color: {h.T_PRI};
}}

QDialog {{
    background-color: {h.BG_BASE};
}}

/* ── Table / tree / list ────────────────────────────────────────────────── */

QTableWidget,
QTreeWidget,
QListWidget {{
    background-color: {h.BG_SURFACE};
    border: 1px solid {h.BORDER_DEF};
    border-radius: {g.RADIUS_SM}px;
    gridline-color: {h.BORDER_SUB};
    selection-background-color: {h.BG_ROW};
    selection-color: {h.T_PRI};
    outline: none;
}}

QHeaderView::section {{
    background-color: {h.BG_STRIP};
    color: {h.T_SEC};
    border: none;
    border-bottom: 1px solid {h.BORDER_DEF};
    padding: {s.S_2}px {s.S_4}px;
    font-size: {f.SIZE_MONO_CAPS}px;
    font-weight: 600;
}}
"""
