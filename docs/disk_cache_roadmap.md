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

- **Timeout shutdown plus généreux** : actuellement `DiskCache.shutdown`
  flush avec un timeout de 2 s. Si le writer thread a 100 frames en
  queue à 50 ms l'unité = 5 s nécessaires. Frames décodées juste avant
  la fermeture sont perdues. Passer le timeout à 10 s + offrir un
  retour visuel ("flushing disk cache…") si > 1 s. Code à modifier :
  `cache/disk_cache.py::shutdown()`.
- **Nettoyage des blobs v1 orphelins** : la v1.5.0 a écrit des blobs
  sous un schéma de clé bogué. Le bump v1 → v2 les a invalidés mais
  ils restent sur disque jusqu'à éviction LRU. Au boot, scan le
  cache_dir pour les fichiers `.bin` orphelins (non référencés par
  l'index SQLite) et les supprimer. Action : nouvelle méthode
  `DiskCache._sweep_orphans()` appelée à l'init, log "swept N
  orphaned blobs (X MB)" en INFO.
- **File watcher pour invalidation auto** : si une source EXR est
  re-rendue mid-session, le `mtime` change. Aujourd'hui le cache sert
  des pixels stale jusqu'à un redémarrage. Implémenter via
  `QFileSystemWatcher` sur le dossier source des layers chargés ;
  quand un fichier change, calculer la clé (nouvelle vs ancienne) et
  invalider l'entrée. Attention : un re-render touche typiquement
  tous les fichiers — débouncer le watcher (~200 ms) pour éviter une
  cascade d'invalidations.
- **Migration v1.5.0 → v1.5.4 propre** : au boot, si la version
  enregistrée (à ajouter dans `index.sqlite` comme PRAGMA
  `user_version`) est < 2, faire un clear automatique avec message
  utilisateur. Évite que l'utilisateur ait à "Clear cache now" à la
  main après mise à jour.

#### F — Multi-process safety (priorité moyenne)

- **Lock file** : si l'utilisateur lance 2 instances de Flick en
  parallèle, elles peuvent écrire concurrent dans `index.sqlite`.
  SQLite WAL gère mais les blobs side-channel pas vraiment.
  Implémentation : créer `<cache_dir>/.lock` au boot via `fcntl` /
  `msvcrt.locking`. Si déjà locké → fallback à un mode "read-only"
  (queries OK, pas de write) avec un warning log. Code à ajouter :
  `cache/disk_cache.py::__init__()`.

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
