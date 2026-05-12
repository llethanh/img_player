# Disk cache — état + roadmap

Référence pour reprendre le dev sur une autre machine. La feature est
livrée et fonctionnelle ; ce fichier liste ce qui reste à faire pour
la polir.

## État au 2026-05-12 (v1.5.4)

### Livré

- **v1.5.0** — Première implémentation du tier disque (lz4 + half-float
  + SQLite). Caching au niveau composite.
- **v1.5.1** — Fix d'un bug critique : la clé source était calculée en
  live-state au lieu du submit-state → mélangeait les channels entre
  cache slots. Bumped key format v1 → v2.
- **v1.5.2** *(A — perf)* — Skip de l'astype float16→float32 dans
  `_deserialize` (le GL viewport accepte `GL_HALF_FLOAT` natif).
  Batch des UPDATE `last_access` toutes les 2 s au lieu d'un par read.
  Gain attendu : 22 ms → ~14-16 ms par disk hit à HD ; ~30 ms gagnés
  par frame 4K (l'astype était le plus gros poste).
- **v1.5.3** *(B + D)* — Pre-paint timeline : `contains_keys` bulk
  query SQLite + `disk_available_master_frames` qui AND par
  contributor ; nouvelle passe rendu "dim-orange" derrière le bright
  orange dans la cache bar. Stats live dans Preferences > Disk cache :
  `DiskCacheStats` dataclass + `QTimer` 1.5 s pour refresh hits /
  misses / writes / read MB / written MB.
- **v1.5.4** *(C — per-layer caching)* — Migration du composite-level
  cache vers du per-contributor caching. Helper
  `_read_contributor_cached(entry)` wrap le cycle `read_frame +
  ensure_rgba + premult-if-straight` avec lookup disque per-layer.
  Le composite est rebuild à chaque fois (~5-10 ms d'over-blend
  marginal). Bénéfice : reorder / hide / add unrelated layer → les
  autres contributors restent hot sur disque.

### Pas encore livré

#### E — Robustesse (priorité haute)

- ~~**Timeout shutdown plus généreux**~~ **(livré, E1)** :
  `DiskCache.shutdown` passé à 10 s + callback de progression. App
  affiche une bulle "Flushing disk cache… (N pending)" centrée sur
  la main window quand la queue > 5 entries au close. Préférences
  passées en non-modal au passage. Commit `c263d53`.
- ~~**Nettoyage des blobs v1 orphelins**~~ **(livré, E2)** :
  `_sweep_orphans()` appelé à l'init, après la migration v4. Scan
  via `os.walk` (rapide, <1s sur 2k entries SSD), diff contre les
  `entries.blob_path` SQLite, unlink des `.bin` non référencés.
  Log INFO "swept N orphan blob(s) (X MB) out of Y scanned in Z ms"
  uniquement si quelque chose a été nettoyé.
- ~~**File watcher pour invalidation auto**~~ **(livré, E3)** :
  `SourceWatcher` (PySide6 `QFileSystemWatcher`) sur le dossier
  parent de chaque layer chargé. Debounce 200 ms pour coalescer un
  re-render burst. Au signal `sources_changed` → appel automatique
  de `_on_reload_sequence` (même chemin que Ctrl+R). Les entries
  disque pour l'ancien mtime restent en cache jusqu'à éviction LRU
  mais ne peuvent pas servir des pixels stale (le mtime fait partie
  de la clé).
- ~~**Migration v1.5.0 → v1.5.4 propre**~~ **(livré, E4)** : PRAGMA
  `user_version` lu au boot via `_migrate_if_needed`. Si version
  on-disk < `_CACHE_FORMAT_VERSION` (= 2 aujourd'hui) ET qu'il y a
  des entries → wipe auto des blobs + DELETE des rows + bump.
  Fresh DB stampée silencieusement. Plus de "Clear cache now" manuel
  après upgrade de format.

#### F — Multi-process safety (priorité moyenne)

- ~~**Lock file**~~ **(livré, F)** : `<cache_dir>/.lock` acquis au
  boot via `msvcrt.locking` (Windows) / `fcntl.flock` (POSIX) en
  mode `LK_NBLCK` / `LOCK_NB`. Si déjà locké → `_read_only = True`,
  writer thread non démarré, `put` / `remove` / `clear` /
  `set_budget` no-op silencieux (ou warning explicit pour `clear`).
  Migration + sweep aussi skipés. Statut exposé via
  `is_read_only()` + `DiskCacheStats.read_only` ; bannière "⚠
  Read-only — another Flick instance owns this cache" dans
  Preferences > Disk cache.

#### Perf supplémentaires (priorité basse)

- **Skip `np.save` / `np.load`** : header overhead (~50-100 bytes +
  ~1-2 ms de parsing). Remplacer par raw bytes + petit header
  custom `(shape_h, shape_w, shape_c, dtype_code)` packé via
  `struct`. Gain estimé : ~1-2 ms par read.
- **Option "no compression"** : pour les utilisateurs avec un NVMe
  rapide + beaucoup d'espace disque, skip lz4 et stocker raw
  float16. Gain : ~5-10 ms par read, coût : 2× plus d'espace
  disque. Switch dans Preferences > Disk cache > "Storage" :
  *Compressed (lz4)* / *Raw (uncompressed)*.
- **Pack multi-frames par blob** : actuellement 1 fichier par frame
  = beaucoup de syscalls (open + read + close). Packer N frames
  contiguës d'une même séquence dans un même fichier blob, indexé
  par offset. Plus complexe, gain attendu marginal sauf sur HDD.

## Conventions / notes utiles

- **Clé source** : SHA-1 de `(canonical_path, mtime_ms, size, sorted_channels, alpha_flags)`
  préfixé par `v2|`. Composite key = SHA-1 de `composite-v2|` + concat
  des per-layer keys.
- **Blob magic** : `b"FCD1"` au début de chaque blob (avant le payload
  lz4). Permet de détecter les blobs corrompus / format pré-1.5.0.
- **Layout** : `cache_dir/ab/cd/<hash>.bin` (sharding 2-niveaux sur
  les 4 premiers chars du hash) — évite 100k fichiers dans un même
  dossier.
- **Counters runtime** (`DiskCacheStats`) : reset à chaque process
  start. NON persistés.
- **Per-layer caching** : cache POST-premult, PRE-`_force_alpha_one`.
  Le `is_opaque_floor` flag varie selon la composition du stack
  → ne doit PAS être dans la clé.

## Workflow pour reprendre

Sur l'autre machine :

```bash
cd "G:\Mon Drive\_PERSO\IA\VibeCod\img_player\img_player_V001"
git pull
# (re)lis ce fichier, choisis E / F / perf
```

Test à faire AVANT de coder la suite : valider C (per-layer caching)
fonctionne correctement sur multi-layer. Si bug → corriger avant E/F.
Voir le résumé de tests dans le dernier message Claude de la session.
