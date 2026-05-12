# Flick Player — notes Claude Code

Lu automatiquement à chaque session par Claude Code, ce fichier voyage
avec le repo (sync via Drive + git). Mets-le à jour en commitant
quand le workflow change.

## Au démarrage de chaque session

**Toujours commencer par** :
```bash
git pull
```

Si je code depuis une nouvelle machine, voir la section "Setup machine
neuve" plus bas.

## Workflow scénario 3 (édition Drive + build local)

Deux emplacements possibles du repo, **selon ce que tu fais** :

| Emplacement | Quand l'utiliser |
|---|---|
| `G:\Mon Drive\_PERSO\IA\VibeCod\img_player\img_player_V001\` | Édition de code, sessions Claude Code, tests pytest |
| `C:\Users\<user>\dev\img_player\` | Build PyInstaller (`build_exe.bat`) — Drive sync casse le bundle |

GitHub (`https://github.com/llethanh/img_player.git`) est la source de
vérité. Drive et le clone local sont deux miroirs synchronisés via
`git pull` / `git push`.

**Règle d'or** : ne jamais éditer simultanément depuis 2 machines.
Drive ne merge pas ; il crée `file (1).py` et casse git. Toujours
`git pull` avant de coder, `git push` avant de quitter une machine.

## Setup machine neuve

Une seule fois par machine :

```bash
# Conda env (Miniforge installé au préalable)
cd "G:\Mon Drive\_PERSO\IA\VibeCod\img_player\img_player_V001"
conda env create -f environment.yml
conda activate img_player

# Si la machine doit aussi builder (= produire un .exe)
git clone https://github.com/llethanh/img_player.git C:\Users\<user>\dev\img_player
```

Ensuite, sessions normales = juste `git pull` puis on code.

## Lancer les tests

```bash
conda activate img_player
pytest tests/
```

**Tests deselectés (pré-existants, pas régressions)** — ignorer si
ils tombent :
- `tests/integration/test_controller.py::test_set_in_out_clamps_current`
- `tests/unit/test_layer_panel.py::TestReorderButtons` (le module entier)
- `tests/unit/test_worker_pool.py::test_clear_drops_pending_tasks`

## Builder un bundle

⚠️ Uniquement depuis le clone local (`C:\Users\<user>\dev\img_player`).
Le `.bat` détecte les chemins Drive / OneDrive / Dropbox et refuse de
tourner.

```cmd
cd C:\Users\<user>\dev\img_player
git pull
build_exe.bat
```

Output : `dist\img_player\` (~700 MB).

Pour wrap en installer Inno Setup voir `installer/README.md`.

## État courant (mai 2026)

- **v1.5.5** sur main — release "Disk cache, hardened"
- Disk cache 3-tiers (RAM → disque lz4+half-float → source decode)
  livré. Survit close/reopen. Pre-paint timeline en orange clair
  pour les frames disponibles disque. Stats live dans
  Preferences > Disk cache.
- **Roadmap entière E + F livrée** (voir
  [`docs/disk_cache_roadmap.md`](docs/disk_cache_roadmap.md)) :
  E1 shutdown drain 10s + FlushIndicator, E2 sweep blobs orphelins,
  E3 auto-reload via QFileSystemWatcher, E4 PRAGMA user_version
  migration, F lock cross-process + read-only fallback.
- **Perf disk-cache** : format v2 struct-header (1.5× faster) +
  v3 no-compression option pour NVMe rapides (5.3× faster, toggle
  dans Preferences > Disk cache > Storage).
- 59 tests unit + integration pour la feature disk-cache (passent
  en ~7s).
- Lecture vidéo (mp4/mov/mkv/m4v/avi) + audio sounddevice opérationnels
- Toggles M/S par layer pour mute/solo audio
- PlayerController en mode wall-clock (anti-drift A/V)
- PyAV + FFmpeg DLLs bundlés via `img_player.spec`
- Inno Setup template prêt dans `installer/`
- Site `docs/website/index.html` à jour avec hero / features / changelog

## Mémoire transverse (~/.claude/MEMORY.md)

Si tu es sur la machine principale (lam), il existe une mémoire user
locale plus large dans `C:\Users\lam\.claude\projects\...\memory\`
qui couvre : profil user, charte design, feature log, comparaisons
avec OpenRV, etc. Cette mémoire est locale à la machine et ne
voyage pas avec git — elle est manuellement maintenue.
