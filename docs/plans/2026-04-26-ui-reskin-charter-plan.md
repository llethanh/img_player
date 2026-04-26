# UI re-skin — Plan d'implémentation

**Date :** 2026-04-26
**Spec de référence :** [`2026-04-26-ui-reskin-charter-design.md`](../specs/2026-04-26-ui-reskin-charter-design.md)
**Statut :** actif
**Estimation totale :** ~3 jours

---

## Principes de travail

1. **Un slice = un commit = un état runnable visiblement amélioré.** Aucun slice ne casse l'app — on peut s'arrêter après n'importe quel slice et avoir une amélioration nette.
2. **Tous les hex / px sortent de `theme.py`.** Aucune valeur magique hard-codée hors de ce module.
3. **Chaque slice livre un test pytest-qt** qui construit le widget touché — minimum syndical anti-régression.
4. **Pas de PR-spam** : on travaille sur une seule branche `feat/ui-reskin-charter` pour les 4 slices, puis une PR consolidée à la fin. Chaque slice = un commit dans cette branche.
5. **Vérification visuelle après chaque slice** : on lance `python -m img_player` sur la séquence locale et on vérifie l'aspect.

Convention de branche : `feat/ui-reskin-charter`.

---

## Slice 1 — Effective FPS metric + status bar refonte

**Objectif :** afficher en live le FPS effectif pendant la lecture, dans une status bar refondue à deux blocs (message contextuel à gauche, perf indicators à droite avec dots colorés conditionnels).

### Tâches

#### 1.1 — Mesure FPS effective dans `PlayerController`
- [ ] Ajouter `import time` et `from collections import deque` dans `controller.py`
- [ ] Ajouter constante module `_TICK_WINDOW = 24`
- [ ] Initialiser `self._tick_timestamps: deque[float] = deque(maxlen=_TICK_WINDOW)` dans `__init__`
- [ ] Append `time.monotonic()` au début de `_tick()`
- [ ] Implémenter property `effective_fps() -> float | None` (returns None si `not is_playing` ou < 2 samples)
- [ ] Clear le deque dans `play()`, `pause()`, `seek()` (pas dans `step()`)

**Acceptance :** test unitaire qui force 24 ticks à 41.67 ms simulés (`monkeypatch time.monotonic`) et asserte `effective_fps() == 24 ± 0.5`.

#### 1.2 — Helpers de formatage (nouveau `src/img_player/ui/status_format.py`)
- [ ] Module pur, **aucune** dépendance Qt
- [ ] Constantes `_FPS_OK = 0.95`, `_FPS_WARN = 0.80`, `_CACHE_FULL = 0.80`
- [ ] `fps_dot_color(effective: float | None, target: float) -> str | None`
  - `None` quand `effective is None`
  - `H.CACHE_BAR` si `effective / target >= 0.95`
  - `H.ACCENT` si `>= 0.80`
  - `H.MARKER_IO` sinon
- [ ] `cache_dot_color(ratio: float) -> str | None`
  - `H.CACHE_BAR` si `ratio >= 0.80`
  - `None` sinon
- [ ] `format_perf_html(*, cache_n, cache_total, cache_ratio, fps_effective, fps_target, ram_gb) -> str`
  - Retourne du rich text HTML utilisable par `QLabel.setText()` avec `setTextFormat(RichText)`
  - Format exact : `<span style='color:HEX'>●</span> cache N/total &nbsp;&nbsp; <span ...>●</span> NN.N fps &nbsp;&nbsp; RAM X.X GB`
  - Si dot is None : pas de span ●, juste le texte (espacement préservé)
  - Si fps_effective is None : afficher `— fps`

**Acceptance :** tests unitaires sur les 3 fonctions :
- `test_fps_dot_color_paused_returns_none`
- `test_fps_dot_color_thresholds_green_amber_red`
- `test_cache_dot_color_thresholds`
- `test_format_perf_html_includes_dots_and_metrics`
- `test_format_perf_html_paused_shows_em_dash`

#### 1.3 — Refonte status bar dans `MainWindow`
- [ ] Imports : `from PySide6.QtWidgets import QLabel`, `from img_player.ui.theme import F, H, S`, `from PySide6.QtCore import Qt`
- [ ] Dans `__init__`, après création du `QStatusBar`, ajouter deux QLabel :
  ```python
  self.status_left = QLabel()
  self.status_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
  self.status_left.setStyleSheet(f"color: {H.TEXT_SECONDARY}; font-size: {F.SIZE_XS}px;")

  self.status_right = QLabel()
  self.status_right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
  self.status_right.setTextFormat(Qt.TextFormat.RichText)
  self.status_right.setFont(F.mono(F.SIZE_XS))

  self.statusBar().addWidget(self.status_left, 1)
  self.statusBar().addPermanentWidget(self.status_right)
  ```
- [ ] `set_status(msg: str)` (méthode existante) re-route vers `self.status_left.setText(msg)` (au lieu de `statusBar().showMessage()`)

**Acceptance :** smoke test pytest-qt qui crée la `MainWindow` et asserte que `status_left` et `status_right` existent et acceptent `setText`.

#### 1.4 — Logique dans `app._refresh_status`
- [ ] Imports : `from img_player.ui.status_format import format_perf_html`
- [ ] Refondre `_refresh_status()` :
  - Sortie tôt si `controller.sequence is None`
  - Calculer `cache_n`, `cache_total`, `cache_ratio`, `eff`, `ram_gb`
  - Construire un message contextuel : `"Loaded {pattern} ({n} frames)"` quand on a une séquence chargée et qu'on lit ; ou laisser le dernier message set par les autres handlers (open, mark in/out, etc.) — décider selon ce qui rend mieux à l'écran
  - `self._window.status_right.setText(format_perf_html(...))`
- [ ] Ne pas toucher à la logique des autres callers de `set_status` (mark_in, mark_out, open, etc.) — ils continuent à écrire dans la zone gauche tel quel.

**Acceptance :** smoke manuel — lancer `python -m img_player <seq>`, voir la status bar split, lancer la lecture, voir le compteur FPS bouger et le dot changer de couleur si on stress (genre `--cache-gb 1` pour forcer du miss).

### Tests

- [ ] `tests/unit/test_controller_fps.py` (1.1)
- [ ] `tests/unit/test_status_format.py` (1.2 — pure functions)
- [ ] `tests/unit/test_main_window.py::test_status_bar_widgets_present` (1.3 — pytest-qt)

### Critère "done"

- [ ] Tests verts (`pytest tests/unit/`)
- [ ] Lancement de l'app sur séquence locale → status bar refondue, compteur FPS live, dots conditionnels (vérifié à l'œil avec `--cache-gb 1` pour stress test)
- [ ] Aucune régression sur `python -m img_player --benchmark` (sortie console identique)
- [ ] Commit message : `feat(ui): live effective FPS metric + split status bar`

---

## Slice 2 — Custom SVG icons + transport

**Objectif :** remplacer les icons Qt natifs (`SP_MediaPlay`, etc.) par les icons stylisées du mockup, paramétrables en couleur via `theme.py`.

### Tâches

#### 2.1 — Module `src/img_player/ui/icons.py`
- [ ] Imports : `functools.lru_cache`, `PySide6.QtCore.QByteArray, Qt`, `PySide6.QtGui.QIcon, QPixmap, QPainter`, `PySide6.QtSvg.QSvgRenderer`, `from img_player.ui.theme import H`
- [ ] Dictionnaire `_TEMPLATES` avec les 7 entries (`play`, `pause`, `stop`, `prev`, `next`, `first`, `last`) — SVG XML inline avec placeholder `{color}` (cf. spec section "Slice 2 / Wiring")
- [ ] Function `make_icon(name: str, color: str = H.TEXT_PRIMARY, size: int = 18) -> QIcon`
  - Lookup `_TEMPLATES[name]`, format avec color
  - `QSvgRenderer(QByteArray(xml.encode("utf-8")))`
  - `QPixmap(size, size)` rempli de `Qt.transparent`
  - Render via `QPainter` puis return `QIcon(pixmap)`
- [ ] Hi-DPI : si `QApplication.primaryScreen().devicePixelRatio() > 1`, render à `size * dpr` puis `pixmap.setDevicePixelRatio(dpr)`
- [ ] Décoré par `@lru_cache(maxsize=64)` sur `(name, color, size)`

**Acceptance :** `make_icon("play")` retourne un `QIcon` non-null pour les 7 noms.

#### 2.2 — Wiring dans `transport.py`
- [ ] Imports : `from img_player.ui.icons import make_icon`, `from img_player.ui.theme import H`
- [ ] Remplacer chaque `style.standardIcon(QStyle.StandardPixmap.SP_MediaXxx)` par l'appel à `make_icon` correspondant :
  - First : `make_icon("first")`
  - Prev : `make_icon("prev")`
  - Play : `make_icon("play", color=H.ACCENT)` (orange)
  - Stop : `make_icon("stop")`
  - Next : `make_icon("next")`
  - Last : `make_icon("last")`
- [ ] Toggle play/pause dans `update_from_state` : swap entre `make_icon("play", color=H.ACCENT)` et `make_icon("pause")` (sans accent)
- [ ] Supprimer `style = self.style()` qui n'est plus utilisé

**Acceptance :** smoke test pytest-qt qui construit `TransportBar` et asserte que le bouton play a un icon non-null.

### Tests

- [ ] `tests/unit/test_icons.py::test_make_icon_returns_non_null` — pour chaque nom dans `_TEMPLATES`
- [ ] `tests/unit/test_icons.py::test_lru_cache_returns_same_instance` — `make_icon("play") is make_icon("play")`
- [ ] `tests/unit/test_transport.py::test_transport_bar_constructs_with_custom_icons` — pytest-qt smoke

### Critère "done"

- [ ] Tests verts
- [ ] Lancement app : transport bar avec icons stylisées, play en orange
- [ ] Toggle play/pause swap correctement les icons
- [ ] Commit message : `feat(ui): custom SVG icons for transport bar`

---

## Slice 3 — Brackets viewer

**Objectif :** ajouter 4 brackets décoratifs aux coins du viewer (subtils, blanc semi-transparent), implémentés en `QWidget` overlay transparent au-dessus du `GLViewport`.

### Tâches

#### 3.1 — Module `src/img_player/ui/brackets_overlay.py`
- [ ] Imports : `PySide6.QtCore.Qt, QRect`, `PySide6.QtGui.QPainter, QPen, QColor`, `PySide6.QtWidgets.QWidget`
- [ ] Classe `BracketsOverlay(QWidget)` :
  - Constantes module `BRACKET_SIZE = 20`, `BRACKET_INSET = 20`, `BRACKET_COLOR = QColor(255, 255, 255, 30)` (alpha ~0.12), `BRACKET_WIDTH = 1`
  - `__init__` : `setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)` pour pas bloquer drag&drop
  - `paintEvent(event)` : dessine 4 brackets en L aux coins (TL, TR, BL, BR) — chaque bracket = 1 ligne verticale + 1 ligne horizontale partant du coin

**Acceptance :** widget construit sans crash, `paintEvent` ne plante pas.

#### 3.2 — Wiring dans `viewer_widget.py`
- [ ] Lire le fichier actuel pour comprendre la structure
- [ ] Si actuellement `QVBoxLayout` avec `GLViewport` direct dedans :
  - Remplacer par `QStackedLayout` en `StackingMode.StackAll`
  - Ajouter le `GLViewport` comme premier widget (couche du dessous)
  - Ajouter le `BracketsOverlay` comme second widget (couche du dessus)
- [ ] Important : `BracketsOverlay` doit avoir la même taille que le `GLViewport` automatiquement (le `QStackedLayout` s'en charge)

**Acceptance :** smoke test pytest-qt qui construit `ViewerWidget` et vérifie que le drag&drop fonctionne toujours (event posé sur l'overlay propage au GLViewport).

### Tests

- [ ] `tests/unit/test_brackets_overlay.py::test_overlay_constructs`
- [ ] `tests/unit/test_brackets_overlay.py::test_overlay_is_transparent_to_mouse` — post un `QMouseEvent`, asserte qu'il n'est pas consommé
- [ ] `tests/unit/test_viewer_widget.py::test_drag_drop_still_works_with_overlay`

### Critère "done"

- [ ] Tests verts
- [ ] Lancement app : 4 brackets visibles aux coins du viewer (très discrets — c'est voulu)
- [ ] Drag & drop d'une séquence sur le viewer fonctionne toujours
- [ ] Commit message : `feat(ui): decorative brackets overlay on viewer`

---

## Slice 4 — Panel migration polish

**Objectif :** auditer les panels qui héritent de la stylesheet globale mais peuvent diverger de la charte sur des détails (paddings, hard-coded colours, group-box titles), et corriger ces divergences en utilisant les tokens de `theme.py`.

### Tâches

#### 4.1 — Audit visuel
- [ ] Lancer `python -m img_player <seq>` après les 3 slices précédents
- [ ] Faire un screenshot de chaque écran/panel : MainWindow, ColorPanel, ChannelPanel, ShortcutsDialog (F1)
- [ ] Comparer chaque screenshot à la section correspondante dans `ui_mockup.html`
- [ ] Lister les divergences dans une note au début du commit ou dans le PR description

#### 4.2 — Corrections probables (à confirmer après audit)
- [ ] **`color_panel.py`** :
  - Vérifier que les group-box titles utilisent le style "section label" (uppercase, letter-spacing, TEXT_DISABLED) — devrait être hérité du QSS `QGroupBox::title`
  - Remplacer toute valeur hex hard-codée par `H.*`
  - Vérifier les `QLabel` "Source colorspace", "Display", "View" — couleur TEXT_SECONDARY
- [ ] **`channel_panel.py`** :
  - Les rows de channels doivent avoir un dot coloré devant (R rouge, G vert, B bleu, A neutre, AOVs greyed) selon le mockup
  - Si actuellement implémenté en `QListWidget` sans icons : ajouter via `QListWidgetItem.setIcon()` avec un petit `QPixmap` rempli de couleur
  - Mono font pour les noms de channels
- [ ] **`main_window.py`** :
  - Vérifier que le menu padding respecte la charte
  - Le titre app de la menubar (`img_player — seq_beauty.####.exr`) à droite — peut nécessiter un widget custom
- [ ] **`shortcuts_dialog.py`** :
  - Les "key labels" (background sombre, padding) sont stylés inline aujourd'hui — déplacer le style dans le QSS global ou le centraliser via une classe widget
  - Couleurs depuis `H.*`

#### 4.3 — Aucun nouveau test
- [ ] Aucune nouvelle logique métier — c'est du polish visuel
- [ ] Les smoke tests existants suffisent (les widgets continuent à se construire)
- [ ] Validation manuelle uniquement (screenshots avant/après)

### Critère "done"

- [ ] Tous les screenshots du slice 4.1 matchent visuellement le mockup (modulo HUD overlay reporté)
- [ ] Aucun hex / valeur magique hors de `theme.py` (audit `grep -E '#[0-9a-fA-F]{6}' src/img_player/ui/*.py` ne ramène rien hors de `theme.py`)
- [ ] Tous les tests passent encore
- [ ] Commit message : `polish(ui): align all panels with Studio Dark charter`

---

## Après les 4 slices

### PR consolidée

- [ ] `git push origin feat/ui-reskin-charter`
- [ ] `gh pr create --title "feat(ui): re-skin to Studio Dark charter (4 slices)"` avec body qui résume :
  - Lien vers spec et plan
  - Récap des 4 slices avec un screenshot avant/après par slice (capture de chaque commit)
  - Note sur ce qui est reporté (HUD overlay, etc.)
  - Test plan : smoke tests verts, validation visuelle vs mockup

### Re-bench post-merge

- [ ] Lancer `python -m img_player --benchmark --passes 3 PATH` après merge sur main
- [ ] Vérifier que les chiffres restent stables (aucun slice ne touche le hot path playback)

### Re-build du bundle Windows

- [ ] `build_exe.bat` (sur clone hors-Drive)
- [ ] Re-zip + upload comme nouvelle release `v0.2.0` sur GitHub
- [ ] Notify l'user pour qu'il re-download et profite du re-skin sur son poste pro

---

## Risques opérationnels

| Risque | Mitigation |
|---|---|
| Slice 1 modifie `controller.py` qui est dans le hot path → risque de régression de perf | On clear le deque sur play/pause/seek, deque maxlen=24 = négligeable. Re-bench après slice 1 pour confirmer. |
| Slice 3 `QStackedLayout(StackAll)` peut intercepter le drag&drop | Test explicite dans 3.2. Si problème, fallback `eventFilter` manuel. |
| Slice 4 audit déborde sur 2-3 jours | Time-box à 1 jour, tout le reste va dans un follow-up "polish 2". |
| Hi-DPI : icons SVG flous | Render à 2× quand `devicePixelRatio > 1` (tâche 2.1). |
| Rich text dans QStatusBar mal supporté sur certaines versions Qt | Fallback : plain text avec couleurs via QSS class si on détecte un problème en dev. |

---

## Definition of done globale (du chantier complet)

- [ ] 4 slices commités sur `feat/ui-reskin-charter`, PR mergée sur main
- [ ] Lancement `python -m img_player <seq>` matche visuellement `ui_mockup.html` (modulo HUD)
- [ ] Tous les tests passent (`pytest tests/unit/`)
- [ ] `python -m img_player --benchmark` produit les mêmes timings qu'avant le chantier
- [ ] `grep -E '#[0-9a-fA-F]{6}' src/img_player/ui/*.py | grep -v theme.py` ne ramène rien
- [ ] Bundle Windows v0.2.0 publié sur GitHub Releases
