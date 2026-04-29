# v1.0 — Multi-Sequence Layer Stack

**Status** : Phase 1 + 2a + 2b + 2c + 3 livrés sur la branche
`feature/v1.0-multi-layer`. Phases 4 / 5 / 6 restent à faire — voir
section "Next session" en bas.

## Décisions de design (toutes verrouillées avec le user)

| Axe | Choix |
|---|---|
| Compositing | **Topmost visible wins** (binaire, pas d'opacité, pas de blend modes) |
| Master range | Union des `[offset, offset + length - 1]` de tous les layers |
| Trous (master frames non couverts) | Écran noir |
| Annotations + comments | **Per-layer**, sur frame source (sidecar à côté de la séquence) |
| Channel selection | Per-layer |
| Contact-sheet config (tiles + layout + labels) | Per-layer |
| OCIO source colorspace | Per-layer |
| Exposure / gamma | Per-layer |
| OCIO display + view | Global |
| RGBA mute | Global |
| FPS | Master (global) |
| Master in/out | Global (boucle de lecture) |
| Trim per-layer (layer in / layer out) | Per-layer (en plus du master in/out) |
| Layer focused ≠ Layer affiché | Oui — concepts distincts |
| Numérotation | Layer 1 = top de stack = priorité max |
| Cache | Master-frame keyed, invalide sur changements de stack |
| Add layer (drop) | Modal "Add / Replace / Cancel" à chaque drop |
| Layer panel | Toujours visible, collapsable |
| Persistance | Fichier `.session` JSON |

## Architecture livrée

```
src/img_player/
├── layers/                 (NEW — Phase 1)
│   ├── __init__.py         re-exports Layer, LayerStack
│   ├── models.py           Layer dataclass (frame arithmetic + per-layer state)
│   └── stack.py            LayerStack QObject (signals + topmost_visible_at)
├── cache/
│   └── master_frame_cache.py  (NEW — Phase 2a)
│                           MasterFrameCache, master-frame keyed,
│                           auto-invalidating on LayerStack signals
├── ui/
│   └── layer_panel.py      (NEW — Phase 3)
│                           LayerPanel + LayerRow widgets
├── app.py                  (UPDATED — Phase 2b)
│                           - _build_models creates self._layer_stack
│                           - MainWindow built with layer_stack=…
│                           - _on_new_sequence clears the stack
├── scan_handler.py         (UPDATED — Phase 2b)
│                           apply_scan_result mirrors loaded sequence
│                           into the stack as a single Layer at
│                           offset = sequence.first_frame
├── ui/main_window.py       (UPDATED — Phase 3)
│                           accepts layer_stack ctor arg, hosts
│                           LayerPanel between timeline and transport
└── preferences.py          (UPDATED — Phase 3)
                            layer_panel_collapsed property

tests/unit/
├── test_layer_basics.py    14/14 ✅ pure-data (no Qt)
├── test_layer_stack.py     qtbot, signals + topmost-visible logic
├── test_master_frame_cache.py qtbot, decode mocked, invalidation paths
└── test_layer_panel.py     qtbot, panel + row interactions
```

## État actuel — comportement utilisateur

**Single-layer** : intact, identique au pré-v1.0. La cache est maintenant
`MasterFrameCache` mais avec offset = first_frame, master_frame == source_frame
donc tout marche pareil numériquement.

**Multi-layer fonctionnel basique** :
- File → Add layer… (Ctrl+Shift+O) ajoute une 2e séquence par-dessus la première
- LayerPanel affiche les rows (numéro / œil / nom / ↑↓)
- Toggle œil sur la layer du top → révèle la layer du dessous (topmost-visible-wins)
- Reorder via ↑↓ change la priorité de stack
- Click sur la row focus la layer (highlight visuel)
- Cache résout proprement le topmost-visible à chaque frame

**Limitations connues (= phases 4-6)** :
- Master timeline range encore bound au controller._sequence (= 1ère seq chargée).
  Les layers en dehors de cette range ne sont pas navigables.
- Pas de drag offset / trim — les nouvelles layers vont par défaut à
  offset = seq.first_frame (= alignées avec la 1ère).
- Per-layer state (channel / exposure / colorspace) pas encore connecté à l'UI.
- Le drag&drop replace encore (pas de modal Add/Replace/Cancel).
- Pas de persistance .session.

## Phase 2c — Cache + controller integration ✅

**Livré** dans les commits `983e7e9` (cache as primary) et `f6b8183` (reload fix).

### Ce qui a été fait

1. **MasterFrameCache** enrichi pour drop-in compat :
   - `attach(seq)`, `detach()`, `set_channels()`, `clear_pending()`, `reload(seq)`
   - Path + mtime indexes per layer (O(1) lookup)
   - `_pre_mark_missing` : eager flag des trous quand layers_changed
   - `reload` mtime-aware avec preservation des frames non modifiées

2. **PlayerController** : type `cache: CacheLike = FrameCache | MasterFrameCache`
   — accepte les deux. Aucun autre changement (l'API surface est identique).

3. **app.py** : construit `MasterFrameCache(self._layer_stack)` au lieu de
   `FrameCache`. Order: stack → cache → controller.

4. **scan_handler** : ne mute plus le stack (cache.attach le fait via
   controller.load_sequence).

5. **RuntimeMonitor** : type `cache: FrameCache | MasterFrameCache`.

### Tests post-2c

- `test_layer_basics.py` 14/14 ✅
- `test_channel_selection.py` 16/16 ✅
- Compile-check OK sur tout le src/.
- Tests qtbot (controller, cache) : à exécuter localement avec un Qt env propre.

## Phase 4 — Drag offsets / trim / snap ✅ (commit `0ff0c5e` + `6eaeb8e`)

LayerBar custom-painted widget per row :
- Drag corps → ``layer.offset``
- Drag poignée gauche (IN) → ``layer.layer_in`` + ``layer.offset`` (édition NLE-style : bord gauche bouge, droit reste)
- Drag poignée droite (OUT) → ``layer.layer_out``
- Snap à 6 px contre playhead, master in/out, et bords des autres layers
- Atomic stack.update au release (= 1 seul ``layer_modified`` signal, 1 invalidation cache)

## Phase 5a — Channel selection per-layer ✅ (commit `c89cb46`)

- Chaque layer porte sa ``channel_selection`` / ``layout_mode`` / ``labels_visible``
- ``set_channel_selection`` mute le layer focusé via ``stack.update``
- ``_display_array`` lit le state de la layer **affichée** (= topmost-visible)
- ``_on_layer_focus_changed`` repopule le menu CH depuis la layer focusée (signaux bloqués pour pas overwrite)

**Differé en 5b/5c** :
- Annotations + comments per-layer (chaque layer son sidecar) — demande refactoring AnnotationStore / CommentStore
- Exposure / gamma / source_colorspace per-layer dans le ColorPanel — UI tracking du focused

## Phase 6 a+b — Drop modal + .session ✅ (commit en cours)

- Modal "Add as layer / Replace / Cancel" au drop quand la stack n'est pas vide. Checkbox "remember for this session" lock le choix sur l'app.
- ``layers/session.py`` : save / load JSON ``.session`` avec re-scan disque (path/dir/base/ext/padding round-trip).
- File → Open Session… / Save Session…

## Phase 6c — Export multi-layer composite (DEFERRED v1.1)

L'export actuel sort la séquence de la layer **focusée** (= ``controller.sequence``).
Pour l'instant le user focus la layer qu'il veut exporter, configure ses channels, et lance Export.

L'export multi-layer composite (= walker le master timeline, résoudre topmost-visible
à chaque master frame, décoder + composer + écrire) demande un refacto significatif :

- ``RenderContext`` doit prendre un ``LayerStack`` (pas un ``SequenceInfo``)
- ``FrameRenderer.render(master_frame, …)`` résout topmost-visible elle-même
- Annotation bake reste skipped en multi-layer (cohérent avec live)

**Différé en v1.1** parce que :
- L'export single-layer répond déjà au cas le plus courant
- Le refacto touche le code testé du writer + du progress
- Pas de blocker pour la v1.0 ship

## Bilan v1.0

```
Phases livrées :
  1   Modèle (Layer + LayerStack)        ✅
  2a  MasterFrameCache                   ✅
  2b  Shadow integration                 ✅
  2c  Cache+controller intégrés          ✅
  3   LayerPanel UI                      ✅
  4   Drag offset / trim / snap          ✅
  5a  Channel selection per-layer        ✅
  6a  Drop modal Add/Replace/Cancel      ✅
  6b  .session save/load                 ✅

Defer v1.1 :
  5b  Annotations / comments per-layer
  5c  Exposure / gamma / colorspace per-layer
  6c  Export multi-layer composite
```

La v1.0 est shippable en l'état pour les workflows :
- Single-sequence (= identique au pré-v1.0)
- Multi-layer comparison (toggle œil entre 2-N layers à la même position)
- Multi-layer end-to-end (offsets différents, "playlist de shots")
- Trim avant lecture (head + tail)
- Save / load des configurations multi-layer

**C'est LE gros morceau qui reste avant que le multi-layer fonctionne pour
de vrai.** À faire en présence du user pour test interactif à chaque étape.

### Migration plan

1. **`PlayerController`** :
   - Constructor : prendre `MasterFrameCache` + `LayerStack` au lieu de `FrameCache`.
   - `load_sequence(seq)` → wrap dans un Layer + add to stack (déjà fait en partie via scan_handler).
   - `_sequence` → remplacer par référence à `_stack`. `.sequence` property retourne le focused layer's sequence pour backward-compat.
   - `_clamp_to_sequence` / `_effective_in/out_frame` → utiliser `stack.master_range()` au lieu de `seq.first/last_frame`.
   - `_prefetch_full_sequence` → itérer `stack.master_range()`.
   - `cache.attach()` → supprimer (MasterFrameCache se synchronise via stack signals).

2. **`app.py`** :
   - `_build_models` : remplacer `FrameCache` par `MasterFrameCache(self._layer_stack)`.
   - `_shutdown` : `self._cache.shutdown()` (API miroir).
   - `_on_reload_sequence` : la cache reload est différente — refaire avec mtime tracking sur les Layers (pas encore implémenté dans MasterFrameCache).
   - `_on_new_sequence` : `cache.clear()` au lieu de `cache.detach()`.

3. **`MasterFrameCache` enrichissements** (à ajouter avant la bascule) :
   - **Path index par layer** (O(1) lookup au lieu de O(n) actuel).
   - **mtime tracking par Layer** pour le reload.
   - **Pre-mark missing frames** au layer.add (= ce que faisait `FrameCache.attach`).

4. **Tests à valider après migration** :
   - Tous les tests `test_controller_*.py` doivent passer.
   - Lecture single-layer fluide (24 fps sur 4K).
   - Cache fill bar du timeline visible.
   - Reload (Ctrl+R) fonctionne.
   - New (Ctrl+N) fonctionne.
   - Export fonctionne.

5. **Risques** :
   - Le cache.attach() faisait du pre-marking missing — sans ça, le
     timeline cache bar ne montre les trous qu'après une tentative de
     play. Acceptable transitoirement.
   - L'epoch logic est différente : MasterFrameCache bump l'epoch sur
     chaque signal du stack (visibility, reorder, modify). Le single-
     layer flow ne touche pas ces signaux, donc même comportement.

## Phase 4 — Drag offset + trim handles + snap

Ajout de l'interaction sur la barre de chaque LayerRow :
- Bar visuelle qui matérialise `[layer.master_start, layer.master_end]` sur le master timeline.
- Drag horizontal de la barre → change `offset`.
- 2 poignées (in / out) sur la barre → change `layer_in` / `layer_out`.
- Snap à `master_in_point`, `master_out_point`, `playhead`, et aux extrémités d'autres layers.

À faire après Phase 2c parce qu'il faut que le master timeline range soit fonctionnel.

## Phase 5 — Per-layer state migration

Les per-layer fields (channel selection, exposure, gamma, source colorspace, contact-sheet
config) existent dans le modèle Layer mais ne sont pas encore connectés à l'UI.
Cette phase rebranche le **layer focused** sur :
- Channel menu (transport bar) — lit/écrit `focused.channel_selection` etc.
- Color panel — lit/écrit `focused.exposure / gamma / source_colorspace`.
- Annotation overlay — lit `focused.annotations_path` pour le sidecar (au lieu d'un store global).
- Comment panel — idem.

**Très risqué** parce que ça touche tous les handlers existants. À faire après
2c et 4 pour pouvoir tester chaque sous-feature isolément.

## Phase 6 — Polish

- Dialog "Add / Replace / Cancel" au drop d'un dossier (avec checkbox "remember
  for this session").
- File menu : "Open as layer…" / "Open session…" / "Save session…".
- Bouton "+" dans le header du panel.
- Sauvegarde / chargement de fichiers `.session` (JSON avec offsets, trims,
  visibility, names, focused_id).
- Export en mode multi-layer : itère le master timeline, résout topmost-visible
  par frame, décode + composite.
- Shortcuts clavier (move layer up / down, delete, …).

## Effort restant estimé

| Phase | Sessions estimées |
|---|---|
| 2c — Cache + controller integration | 1 grosse session |
| 4 — Drag offsets + trim + snap | 1 grosse session |
| 5 — Per-layer state migration | 1-2 sessions (très impactant) |
| 6 — Polish + persistance + export | 1 session |
| **Total** | **4-5 sessions** avant la v1.0 stable |
