# Baseline — img_player avant optimisations Phase 1

*Date : 2026-04-26 — branche `perf-baseline-bench`*

## Configuration de la mesure

| Élément              | Valeur                                                |
|----------------------|-------------------------------------------------------|
| Séquence             | `SH0010_Rendered_RGB.####.exr` (90 frames)            |
| Résolution           | **3840 × 2160 × 4** (4K UHD, RGBA)                    |
| Format               | EXR multichannel (lecture R/G/B/A only via OIIO)      |
| Taille moyenne/frame | ~63 MiB décodés (float16)                             |
| Stockage             | SSD interne (`C:\Users\lam\PERSO\images\…`)           |
| GPU                  | AMD Radeon 780M (iGPU, APU mobile)                    |
| OS / Python          | Windows / 3.11                                        |
| Cache RAM            | 8 GiB                                                 |
| Workers de decode    | 6                                                     |
| Target playback      | 24 fps · 3 passes complètes · warmup 30 frames        |

Commande exécutée :

```bash
python -m img_player --benchmark --passes 3 --warmup-frames 30 \
  --bench-output perf/baseline.json \
  "C:\Users\lam\PERSO\images\SH0010_Rendered_RGB"
```

## Résultats

```
Tick (controller QTimer):
  effective fps  :  12.359  (target 24.000)
  cache hit rate :  71.48 %
  inter-tick gap   n=269  mean= 80.91  p50= 78.00  p95=187.60  p99=286.80  max=375.00 ms

Paint (paintGL body):
  effective fps  :   9.235
  upload           n=201  mean=50368  p50=47000  p95=109000  p99=219000  max=235000 µs
  paint total      n=201  mean=53174  p50=47000  p95=110000  p99=219000  max=235000 µs
  inter-paint gap  n=200  mean=108.29 p50= 79.00 p95=281.80  p99=610.93  max=1218.00 ms

Decode (worker pool):
  decode time      n=54   mean=1866  p50=1859   p95=1984    p99=2185    max=2375    ms
  total decoded   : 3417 MiB (54 frames)
```

## Analyse

### 1. On rate la moitié du framerate cible

**12.4 fps réel** vs 24 fps demandés. Sur 24 ticks par seconde voulus, on
en délivre la moitié. Le `cache hit rate` à **71%** signifie que pour 28%
des ticks la frame demandée n'est pas encore décodée — elle compte comme
drop dans le compteur du player.

### 2. Le `paintGL` est le goulot principal du main thread

* `paint_us mean = 53 ms` → **chaque paint dépasse le budget de 41 ms** d'un
  frame à 24 fps. Le main thread Qt n'a aucune chance de soutenir 24 fps
  tant qu'un seul `paintGL` peut prendre 235 ms (`max`).
* L'upload représente **95% du temps de paint** (50 ms sur 53 ms).
  Autrement dit : si l'upload était gratuit, le paint serait à **3 ms**.

`glTexSubImage2D` synchrone bloque le main thread sur 32 MiB de pixels
float16 vers VRAM. Sur GPU intégré 780M le chemin DMA est partagé avec la
RAM système → le coût est élevé.

➡ **Candidat #1 d'optim : PBO async upload.** Délègue le transfert au
driver en mode asynchrone, libère le main thread, pipeline avec la frame
suivante. PyOpenGL le supporte sans dépendance native supplémentaire.

### 3. Le decode est lent aussi

* `decode mean = 1866 ms / frame`. À 6 workers en parallèle = **3.2
  frames/seconde** soutenu. Pour 24 fps il faudrait 7-8× ce throughput,
  soit en cache permanent, soit en accélérant le decode.
* Notre cache de 8 GiB tient ~130 frames de 63 MiB → la séquence entière
  (90 frames) tient en cache. **Une fois l'état stationnaire atteint**, le
  decode n'est plus sur le chemin critique. Le warmup actuel met **8.9 s**
  pour caser 30 frames.

➡ **Candidat #2 : OIIO threading config** (`oiio.attribute("threads",
N)`). Permet à OIIO de paralléliser intra-frame en plus du parallélisme
inter-frame de notre worker pool. Gain attendu 2-4× sur EXR multichannel.

### 4. Inter-tick gap dispersé

* `inter-tick mean = 81 ms` → le QTimer arme à 41 ms mais les ticks sont
  retardés par les paints synchrones qui reviennent dans le même thread.
* `p99 = 287 ms` → sept frames de retard ponctuel.

C'est une conséquence directe du #2 : si le paint passe à <10 ms, le tick
reste à 41 ms.

### 5. Allocation numpy → GC pauses ?

Pas instrumenté directement, mais : 63 MiB par frame alloués + libérés
fréquemment = pression sur l'allocateur Python + GC. Visible dans le
`paint_us p99 = 219 ms` qui est ~4× le mean — typique d'une pause GC.

➡ **Candidat #3 : buffer pool numpy + `gc.freeze()/disable()` durant
playback.**

## Plan d'optimisation Phase 1

Ordre par gain attendu sur cette baseline :

| #  | Optim                                | Cible (où ça mord)              | Gain estimé    | Difficulté |
|----|--------------------------------------|---------------------------------|----------------|------------|
| 1  | **PBO async upload**                 | `paint_us`, `inter-paint gap`   | -40 ms / frame | Moyenne    |
| 2  | **OIIO `attribute("threads", N)`**   | `decode_ms`                     | 2-4× decode    | Triviale   |
| 3  | **`gc.freeze() + gc.disable()`**     | `paint_us p99`                  | Élimine spikes | Triviale   |
| 4  | **Asymmetric eviction**              | `cache_hit_rate`                | +10-15 pts     | Facile     |
| 5  | **Buffer pool numpy**                | Pression GC, `paint_us p99`     | -2-3 ms        | Moyenne    |

Critère de validation : on relance `python -m img_player --benchmark
--bench-output perf/optim_<n>.json …` après chaque optim et on compare la
ligne `Tick / effective fps` vs cette baseline.

**Cible Phase 1** : sur cette même séquence et matos, atteindre **24 fps
effectifs avec ≥ 95% cache hit rate** une fois le cache chaud.

## Notes méthodologiques

* Le bench utilise `LoopMode.LOOP` et compte une "passe" comme un wrap
  `last_frame → first_frame`.
* La phase de warmup (avant `recorder.enable()`) est exclue des stats.
* Les samples sont stockés en mémoire dans des `deque` thread-safe puis
  agrégés à la fin — overhead négligeable (~50 ns par hook quand actif,
  une seule branche `if not _ENABLED` quand inactif).
* L'upload est mesuré en wall-clock autour de `glTexSubImage2D` — il
  capture exactement ce qu'on cherche : le temps pendant lequel le main
  thread est bloqué. Pas de `glFinish` injecté (qui défoncerait la perf).
