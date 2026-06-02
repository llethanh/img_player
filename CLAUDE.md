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
| `C:\dev\FlickPlayer\` | Build PyInstaller (`build_exe.bat`) — Drive sync casse le bundle |

GitHub (`https://github.com/llethanh/FlickPlayer.git`) est la source de
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
git clone https://github.com/llethanh/FlickPlayer.git C:\dev\FlickPlayer
```

Ensuite, sessions normales = juste `git pull` puis on code.

## Lancer les tests

```bash
conda activate img_player
pytest tests/
```

**Toute la suite doit passer au vert** (971 tests, ~20 s en local).
Pas de deselects pré-existants : la dette d'anciens tests obsolètes
qui traînait jusqu'à mai 2026 a été nettoyée dans le pass
"Mock-fragility + behavior-drift" — voir le commit `tests: revive 36
broken tests` pour le détail (FrameCache duck-typing + master-frame
vs source-frame confusion + obsolete UI feature removed).

## Builder un bundle

⚠️ Uniquement depuis le clone local (`C:\dev\FlickPlayer`).
Le `.bat` détecte les chemins Drive / OneDrive / Dropbox et refuse de
tourner.

```cmd
cd C:\dev\FlickPlayer
git pull
build_exe.bat
```

Output : `dist\FlickPlayer_v<X.Y.Z>\` (~380 MB depuis v1.8.0,
PyInstaller 6.20 dédup plus agressif que les anciennes versions).

Pour wrap en installer Inno Setup voir `installer/README.md`.

## État courant (mai 2026)

- **v1.5.13** sur main — release "Smoother playback + code health"
- **Perf hot path (Tier 1)** : OpenGL uniform-location caching +
  LUT bind-once → paint mean −25 % (6 802 → 5 089 µs), paint max
  −92 % (593 → 47 ms, plus de spikes visibles). OCIO shader bundle
  LRU. Composite math `np.multiply(out=tmp)` scratch buffer.
  Scanner `os.scandir`. Lazy imports hot-path hoistés.
- **Refacto structurel (Tier 2)** : 180-LOC `_evict_if_over_budget`
  split en 25 + 5 helpers ; `_on_frame_changed` + `_refresh_after_stack_change`
  splittés en helpers nommés ; `cache/_common.py` extrait ; canonical
  `enrich_with_header` ; `_signature_token` helper.
- **Dette de tests purgée** : suite 971/0/0 (avant : 20 failed,
  16 errors, 3 deselected). Causes racines : Mock `spec=FrameCache`
  rejetait nouvelles méthodes ; master-frame vs source-frame
  confusion ; comportement changé non répercuté ; features UI
  obsolètes.
- Disk cache 3-tiers (RAM → disque lz4+half-float → source decode)
  livré v1.5.5. Survit close/reopen. Pre-paint timeline en orange
  clair pour les frames disponibles disque.
- **Disk cache roadmap E + F livrée** :
  E1 shutdown drain 10s + FlushIndicator, E2 sweep blobs orphelins,
  E3 auto-reload via QFileSystemWatcher, E4 PRAGMA user_version
  migration, F lock cross-process + read-only fallback.
- **Perf disk-cache** : format v2 struct-header (1.5× faster) +
  v3 no-compression option pour NVMe rapides (5.3× faster, toggle
  dans Preferences > Disk cache > Storage).
- **3-tier prefs system** (v1.5.8+) : user TOML > site TOML >
  hardcoded. `flick.toml` à côté de `FlickPlayer.exe` ou dans
  `%APPDATA%\FlickPlayer\flick.toml`.
- **8 builtin OCIO configs** (v1.5.10) : ACES 1.3 / 2.0, CG / Studio,
  default ACES 1.3 matchant Nuke / Maya / OpenRV.
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
