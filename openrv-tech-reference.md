# Référence Technique — Lecteur de Séquences d'Images
## Basé sur l'analyse d'OpenRV (Academy Software Foundation)

> Document de contexte technique à soumettre au projet de lecteur de séquences.  
> Objectif : mettre à jour l'architecture et les choix technologiques du projet en cours.

---

## 1. Qu'est-ce qu'OpenRV

OpenRV est la version open source de RV, un logiciel de review et playback média récompensé par un Sci-Tech Award. C'est le viewer de référence industrie pour les artistes VFX et animation.

- **Mainteneur** : Academy Software Foundation (ASWF)
- **Licence** : Apache 2.0 (usage commercial autorisé, fork autorisé)
- **Repo** : https://github.com/AcademySoftwareFoundation/OpenRV
- **Plateformes** : Linux, macOS, Windows
- **Langage principal** : C++

---

## 2. Architecture générale

```
┌─────────────────────────────────────┐
│         UI Layer (Qt / Python)      │
│   Toolbar · Preferences · Annotate  │
├─────────────────────────────────────┤
│       Application Core (C++)        │
│     RvApplication · RvDocument      │
├─────────────────────────────────────┤
│     Media Processing Pipeline       │
│        IPGraph · IPNode             │
├─────────────────────────────────────┤
│         Rendering System            │
│    ImageRenderer · OpenGL/GLSL      │
├─────────────────────────────────────┤
│         External Libraries          │
│  OpenEXR · FFmpeg · OCIO · OTIO     │
└─────────────────────────────────────┘
```

### Composants clés

| Composant | Rôle |
|---|---|
| `RvApplication` / `RvDocument` | Gestion UI, session, multi-document |
| `IPGraph` | Graphe de traitement image (non-destructif, orienté nœuds) |
| `IPNode` | Nœud de traitement unitaire (source, LUT, CDL, composite...) |
| `ImageRenderer` | Rendu final OpenGL à l'écran |
| Plugin / RVPKG | Système de packages pour extensions |

---

## 3. Technologies d'optimisation du playback temps réel

C'est le cœur du sujet. La performance temps réel repose sur **trois piliers** :

### 3.1 Pipeline nœuds découplé (IPGraph)

Le traitement image est modélisé comme un graphe de nœuds. Cela permet :
- Le **préchargement anticipé** des frames suivantes (prefetch)
- Le **cache mémoire** des frames déjà décodées
- L'exécution **parallèle** de plusieurs nœuds (multi-threading)

### 3.2 Upload GPU asynchrone via Pixel Buffer Objects (PBO)

C'est la technique clé pour le streaming de séquences haute résolution.

**Principe :**
- Les frames sont décodées en RAM (CPU)
- Le transfert RAM → VRAM se fait via **DMA** (Direct Memory Access) sans bloquer le CPU
- OpenGL planifie le transfert en tâche de fond
- Pendant ce temps, le CPU peut décoder la frame suivante

**Résultat :** pipeline CPU/GPU totalement découplé, pas de stall, playback fluide.

```
Frame N-1        Frame N          Frame N+1
[CPU decode] → [PBO DMA upload] → [GPU render]
     ↑                                  ↓
  [prefetch]                      [affichage écran]
```

### 3.3 Traitement couleur sur GPU (OCIO + GLSL shaders)

Toutes les opérations couleur (exposition, LUT, CDL, tone mapping) sont compilées en **fragment shaders GLSL** et exécutées entièrement sur GPU. Zéro overhead CPU pour le color management.

---

## 4. Stack technique recommandée pour le projet

### Choix de langage
- **Core moteur : C++** → performance maximale pour decode, cache, rendu
- **UI + outils : Python** → rapidité de développement, scripting, extensions

C'est exactement l'architecture d'OpenRV.

### Formats à supporter

| Format | Lib recommandée | Notes |
|---|---|---|
| EXR (multi-couches) | **OpenEXR 3.x** | Standard VFX, support HDR natif |
| PNG / TIFF / JPG | **stb_image** ou **OpenImageIO** | Léger et rapide |
| mp4 / mov | **FFmpeg 6.x** | Décodage hardware possible |
| DPX | **OpenImageIO** | Format film |

### Dépendances core recommandées

```
OpenEXR >= 3.2     → séquences EXR multi-couches
FFmpeg >= 6.1      → vidéo mp4/mov, codec AV1
OpenColorIO >= 2.3 → gestion couleur GPU
OpenImageIO        → formats PNG/TIFF/JPG/DPX
Qt >= 6.5          → UI cross-platform
OpenGL + GLSL      → rendu et color processing GPU
Boost >= 1.82      → utilitaires C++
```

---

## 5. Ce qu'il faut extraire d'OpenRV (stratégie de fork partiel)

L'approche recommandée n'est **pas** de forker tout OpenRV, mais d'extraire uniquement le **moteur de playback** :

### À garder / extraire
- `IPGraph` / `IPNode` → pipeline de traitement image
- `ImageRenderer` → rendu OpenGL
- Système de cache et prefetch de frames
- Les I/O : OpenEXR, FFmpeg, stb_image

### À remplacer
- **UI Qt d'OpenRV** → remplacée par l'UI custom du projet
- **Système de packages RVPKG** → remplacé par le système d'extension propre

### Architecture cible du projet

```
┌──────────────────────────────────────┐
│         UI Custom (branding)         │
│     Layout · Contrôles · Timeline    │
├──────────────────────────────────────┤
│      Outils Review / Annotation      │
│   Comparaison A/B · Notes · Markers  │
├──────────────────────────────────────┤
│     Moteur OpenRV (extrait/forké)    │
│       IPGraph · Cache · Prefetch     │
├──────────────────────────────────────┤
│            Rendu GPU                 │
│     OpenGL · GLSL · PBO · OCIO       │
├──────────────────────────────────────┤
│           Décodeurs                  │
│   OpenEXR · FFmpeg · OpenImageIO     │
└──────────────────────────────────────┘
```

---

## 6. Fonctionnalités clés d'OpenRV à s'inspirer

| Fonctionnalité | Description |
|---|---|
| Playback temps réel | Séquences EXR 4K+ à framerate cible |
| Color management | OCIO intégré, LUT custom, CDL |
| Comparaison A/B | Wipe, side-by-side, différence |
| Annotation | Dessin frame, markers timeline |
| Remote sync | Review synchronisé multi-utilisateurs |
| OpenTimelineIO | Import/export timelines éditoriales |
| SDI output | Via Blackmagic DeckLink SDK (optionnel) |
| Scripting | Python API pour automatisation pipeline |

---

## 7. Références

- **Repo OpenRV** : https://github.com/AcademySoftwareFoundation/OpenRV
- **Documentation** : https://aswf-openrv.readthedocs.io
- **ASWF** : https://www.aswf.io
- **OpenColorIO** : https://opencolorio.org
- **OpenEXR** : https://openexr.com
- **OpenTimelineIO** : https://opentimelineio.readthedocs.io
- **FFmpeg** : https://ffmpeg.org

---

*Document généré à partir d'une analyse technique d'OpenRV — avril 2026*
