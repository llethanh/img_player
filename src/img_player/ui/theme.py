"""Design system for img_player — "Studio Dark" charter.

Single source of truth for all colors, typography, spacing, and geometry.
Never hard-code hex values elsewhere in the UI — always import from here.

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

class C:
    # Backgrounds (4 depth levels)
    BG_DEEP    = QColor("#141416")
    BG_BASE    = QColor("#1C1C1F")
    BG_RAISED  = QColor("#242428")
    BG_SURFACE = QColor("#2C2C30")
    BG_HOVER   = QColor("#35353A")
    BG_SELECT  = QColor("#3A3A40")

    # Borders
    BORDER_SUBTLE  = QColor("#28282C")
    BORDER_DEFAULT = QColor("#38383C")
    BORDER_STRONG  = QColor("#56565C")

    # Text
    TEXT_PRIMARY   = QColor("#E4E4E6")
    TEXT_SECONDARY = QColor("#8A8A8E")
    TEXT_DISABLED  = QColor("#52525A")

    # Accent — warm amber
    ACCENT        = QColor("#E8901C")
    ACCENT_BRIGHT = QColor("#F5A830")
    ACCENT_DIM    = QColor("#7A4210")

    # Media-specific
    PLAYHEAD         = QColor("#F5AA28")
    PLAYHEAD_OUTLINE = QColor("#141416")
    MARKER_IO        = QColor("#DC3C3C")
    # Range bar: the IN/OUT span. Was green (#56B46A) — switched to a
    # warm orange that picks up the rest of the accent palette so the
    # whole timeline reads as one family with the playhead.
    RANGE_BAR        = QColor("#E8901C")  # = ACCENT
    # Cache bar — second iteration based on user feedback:
    #   * Slot background = pure deep black so empty / not-yet-cached
    #     frames *clearly* read as missing.
    #   * Each cached run = translucent orange fill + opaque orange
    #     border. The semi-transparency makes the cached runs feel
    #     like windows over the black slot rather than solid blocks.
    CACHE_BAR_BG     = QColor("#141416")              # = BG_DEEP, opaque
    CACHE_BAR        = QColor(0xE8, 0x90, 0x1C, 128)  # ACCENT @ 50% alpha
    CACHE_BAR_BORDER = QColor("#E8901C")              # ACCENT, opaque

    # Timeline ticks
    TICK_MINOR = QColor("#3C3C40")
    TICK_MAJOR = QColor("#6A6A70")
    TICK_LABEL = QColor("#8A8A8E")

    # Semantic
    COLOR_ERROR = QColor("#C04040")


# ---------------------------------------------------------------------------
# Hex strings — for embedding in QSS
# ---------------------------------------------------------------------------

class H:
    BG_DEEP    = "#141416"
    BG_BASE    = "#1C1C1F"
    BG_RAISED  = "#242428"
    BG_SURFACE = "#2C2C30"
    BG_HOVER   = "#35353A"
    BG_SELECT  = "#3A3A40"

    BORDER_SUBTLE  = "#28282C"
    BORDER_DEFAULT = "#38383C"
    BORDER_STRONG  = "#56565C"

    TEXT_PRIMARY   = "#E4E4E6"
    TEXT_SECONDARY = "#8A8A8E"
    TEXT_DISABLED  = "#52525A"

    ACCENT        = "#E8901C"
    ACCENT_BRIGHT = "#F5A830"
    ACCENT_DIM    = "#7A4210"

    # Media-specific — needed when we embed colours in QSS / rich text
    # (status bar dots, etc.) instead of using QPainter directly.
    PLAYHEAD  = "#F5AA28"
    MARKER_IO = "#DC3C3C"
    # Note: status bar uses these for the live indicators. Cache fill
    # dot stays "good"-coded (green-ish) when the cache is healthy, so
    # we keep CACHE_BAR distinct from the timeline accent here.
    RANGE_BAR = "#E8901C"   # warm accent — was green (#56B46A)
    CACHE_BAR = "#38B464"   # cache OK indicator (status bar dot)


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

class F:
    FAMILY_UI   = "Segoe UI"
    FAMILY_MONO = "JetBrains Mono"

    SIZE_XS = 9
    SIZE_SM = 10
    SIZE_MD = 11

    @staticmethod
    def ui(size: int = 10, bold: bool = False) -> QFont:
        f = QFont("Segoe UI")
        f.setPixelSize(size)
        if bold:
            f.setWeight(QFont.Weight.DemiBold)
        return f

    @staticmethod
    def mono(size: int = 9) -> QFont:
        f = QFont("JetBrains Mono")
        f.setPixelSize(size)
        return f


# ---------------------------------------------------------------------------
# Spacing — 4 px grid
# ---------------------------------------------------------------------------

class S:
    XS  = 2
    SM  = 4
    MD  = 8
    LG  = 12
    XL  = 16
    XXL = 24


# ---------------------------------------------------------------------------
# Component geometry
# ---------------------------------------------------------------------------

class G:
    BTN_TRANSPORT_W = 30
    BTN_TRANSPORT_H = 28
    BTN_TEXT_W      = 32
    INPUT_H         = 24
    ICON_SIZE       = 18

    TRANSPORT_H = 38
    TIMELINE_H  = 52

    RADIUS_SM = 2
    RADIUS_MD = 3
    RADIUS_LG = 5


# ---------------------------------------------------------------------------
# Global QSS
# ---------------------------------------------------------------------------

def build_stylesheet() -> str:
    """Return the complete QSS for the application.

    Call once at startup::

        app.setStyleSheet(build_stylesheet())
    """
    h = H
    s = S
    g = G
    f = F

    return f"""
/* ── Base ──────────────────────────────────────────────────────────────── */

QWidget {{
    background-color: {h.BG_BASE};
    color: {h.TEXT_PRIMARY};
    font-family: "{f.FAMILY_UI}", "Helvetica Neue", sans-serif;
    font-size: {f.SIZE_SM}px;
    border: none;
    outline: none;
}}

QMainWindow {{
    background-color: {h.BG_BASE};
}}

/* ── Menu bar ───────────────────────────────────────────────────────────── */

QMenuBar {{
    background-color: {h.BG_RAISED};
    color: {h.TEXT_PRIMARY};
    padding: 1px 0;
    border-bottom: 1px solid {h.BORDER_SUBTLE};
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
    background-color: {h.BG_RAISED};
    border: 1px solid {h.BORDER_DEFAULT};
    padding: {s.XS}px;
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
    color: {h.TEXT_DISABLED};
}}

QMenu::separator {{
    height: 1px;
    background: {h.BORDER_SUBTLE};
    margin: {s.XS}px {s.MD}px;
}}

/* ── Status bar ─────────────────────────────────────────────────────────── */

QStatusBar {{
    background-color: {h.BG_RAISED};
    color: {h.TEXT_SECONDARY};
    border-top: 1px solid {h.BORDER_SUBTLE};
    font-size: {f.SIZE_XS}px;
    padding: 0 {s.SM}px;
}}

QStatusBar::item {{
    border: none;
}}

/* ── Dock ───────────────────────────────────────────────────────────────── */

QDockWidget {{
    background-color: {h.BG_RAISED};
    color: {h.TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {h.BG_RAISED};
    padding: {s.SM}px {s.MD}px;
    border-bottom: 1px solid {h.BORDER_SUBTLE};
    font-weight: 600;
    color: {h.TEXT_DISABLED};
    font-size: {f.SIZE_XS}px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ── Tab widget ─────────────────────────────────────────────────────────── */

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {h.BORDER_SUBTLE};
    background-color: {h.BG_RAISED};
}}

QTabBar {{
    background-color: {h.BG_BASE};
}}

QTabBar::tab {{
    background-color: transparent;
    color: {h.TEXT_SECONDARY};
    padding: 5px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: {f.SIZE_SM}px;
}}

QTabBar::tab:selected {{
    color: {h.TEXT_PRIMARY};
    border-bottom: 2px solid {h.ACCENT};
    background-color: {h.BG_RAISED};
}}

QTabBar::tab:hover:!selected {{
    background-color: {h.BG_HOVER};
    color: {h.TEXT_PRIMARY};
}}

/* ── Buttons ────────────────────────────────────────────────────────────── */

QPushButton {{
    background-color: {h.BG_SURFACE};
    color: {h.TEXT_PRIMARY};
    border: 1px solid {h.BORDER_DEFAULT};
    border-radius: {g.RADIUS_MD}px;
    padding: 2px {s.MD}px;
    min-height: {g.BTN_TRANSPORT_H}px;
}}

QPushButton:hover {{
    background-color: {h.BG_HOVER};
    border-color: {h.BORDER_STRONG};
}}

QPushButton:pressed {{
    background-color: {h.BG_SELECT};
    border-color: {h.ACCENT};
}}

QPushButton:disabled {{
    color: {h.TEXT_DISABLED};
    border-color: {h.BORDER_SUBTLE};
    background-color: {h.BG_RAISED};
}}

/* ── Inputs ─────────────────────────────────────────────────────────────── */

QComboBox,
QDoubleSpinBox,
QSpinBox,
QLineEdit {{
    background-color: {h.BG_SURFACE};
    color: {h.TEXT_PRIMARY};
    border: 1px solid {h.BORDER_DEFAULT};
    border-radius: {g.RADIUS_SM}px;
    padding: 0 {s.SM}px;
    min-height: {g.INPUT_H}px;
    max-height: {g.INPUT_H}px;
    selection-background-color: {h.ACCENT_DIM};
    selection-color: {h.TEXT_PRIMARY};
}}

QComboBox:hover,
QDoubleSpinBox:hover,
QSpinBox:hover,
QLineEdit:hover {{
    border-color: {h.BORDER_STRONG};
}}

QComboBox:focus,
QDoubleSpinBox:focus,
QSpinBox:focus,
QLineEdit:focus {{
    border-color: {h.ACCENT};
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
    background-color: {h.BG_RAISED};
    border: 1px solid {h.BORDER_DEFAULT};
    selection-background-color: {h.BG_HOVER};
    selection-color: {h.TEXT_PRIMARY};
    outline: none;
    padding: {s.XS}px;
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
    border: 1px solid {h.BORDER_DEFAULT};
    border-radius: {g.RADIUS_MD + 1}px;
    margin-top: 1.2em;
    padding-top: {s.SM}px;
    color: {h.TEXT_DISABLED};
    font-size: {f.SIZE_XS}px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {s.MD}px;
    padding: 0 {s.SM}px;
    background-color: {h.BG_RAISED};
}}

/* ── Scroll bars ────────────────────────────────────────────────────────── */

QScrollBar:vertical {{
    background: {h.BG_RAISED};
    width: 6px;
    margin: 0;
    border-radius: 3px;
}}

QScrollBar::handle:vertical {{
    background: {h.BORDER_STRONG};
    border-radius: 3px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {h.TEXT_DISABLED};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
    height: 0;
}}

QScrollBar:horizontal {{
    background: {h.BG_RAISED};
    height: 6px;
    margin: 0;
    border-radius: 3px;
}}

QScrollBar::handle:horizontal {{
    background: {h.BORDER_STRONG};
    border-radius: 3px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {h.TEXT_DISABLED};
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
    background-color: {h.BG_RAISED};
    color: {h.TEXT_PRIMARY};
    border: 1px solid {h.BORDER_DEFAULT};
    padding: {s.XS}px {s.SM}px;
    border-radius: {g.RADIUS_SM}px;
    font-size: {f.SIZE_XS}px;
}}

/* ── Label ──────────────────────────────────────────────────────────────── */

QLabel {{
    background: transparent;
    color: {h.TEXT_SECONDARY};
    border: none;
}}

/* ── Message box ────────────────────────────────────────────────────────── */

QMessageBox {{
    background-color: {h.BG_BASE};
}}

QMessageBox QLabel {{
    color: {h.TEXT_PRIMARY};
}}

/* ── Dialog ─────────────────────────────────────────────────────────────── */

QDialog {{
    background-color: {h.BG_BASE};
}}

/* ── Table / tree ───────────────────────────────────────────────────────── */

QTableWidget,
QTreeWidget,
QListWidget {{
    background-color: {h.BG_SURFACE};
    border: 1px solid {h.BORDER_DEFAULT};
    border-radius: {g.RADIUS_SM}px;
    gridline-color: {h.BORDER_SUBTLE};
    selection-background-color: {h.BG_SELECT};
    selection-color: {h.TEXT_PRIMARY};
    outline: none;
}}

QHeaderView::section {{
    background-color: {h.BG_RAISED};
    color: {h.TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {h.BORDER_DEFAULT};
    padding: {s.XS}px {s.SM}px;
    font-size: {f.SIZE_XS}px;
    font-weight: 600;
}}
"""
