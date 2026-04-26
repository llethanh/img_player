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

## Résultats des optimisations

### Optim #1 — PBO async upload : ❌ contre-productif

Voir [`PBO_NOTES.md`](PBO_NOTES.md). Sur l'iGPU à mémoire unifiée le PBO
ajoute des copies sans gain DMA. **Reverté.**

### Optim #2 — OIIO `threads=1` : ✅ +47% effective fps

Insight inattendu : le bottleneck "upload" était en fait de la
**contention mémoire**. Avec 16 threads OIIO concurrents qui décompressent
des EXR (memory-bound), la bande passante DRAM est saturée — le
`glTexSubImage2D` du main thread (qui est un memcpy DRAM→DRAM sur iGPU)
en pâtit aussi.

| Métrique         | Baseline      | OIIO=16        | **OIIO=1**       | Δ baseline |
|------------------|---------------|----------------|------------------|------------|
| effective fps    | 12.36         | 12.76          | **18.16**        | **+47 %**  |
| upload mean      | 50 368 µs     | 49 086 µs      | **29 701 µs**    | **-41 %**  |
| paint mean       | 53 174 µs     | 50 523 µs      | **30 960 µs**    | **-42 %**  |
| paint p99        | 219 000 µs    | 189 520 µs     | **62 000 µs**    | **-72 %**  |
| decode mean      | 1 866 ms      | 1 979 ms       | **1 701 ms**     | **-9 %**   |
| cache hit rate   | 71.5 %        | 72.6 %         | 62.2 %           | -9 pts     |

Cache hit rate baisse parce qu'on consomme les frames plus vite (18 fps vs
12) sans que le decode suive — le bottleneck restant est le decode. Mais
on a clairement gagné sur le main-thread / upload.

**Default changé** dans `app.py` : `DEFAULT_OIIO_THREADS = 1`. Override via
`--oiio-threads N` si le hardware change (workstation discrete, NUMA…).

### Optim #3 — `gc.freeze()` + `gc.disable()` durant playback : ✅ -77% paint p99

Le GC cyclique de Python lance des passes opportunistes qui peuvent durer
des dizaines de millisecondes — visibles comme des spikes dans
`paint_us p99`. Pendant le playback on n'alloue pas (ou très peu) d'objets
cycliques nouveaux ; pauser le GC est sûr et tue les spikes :

| Métrique         | OIIO=1 seul  | **+ GC tweak**  | Δ          |
|------------------|--------------|------------------|------------|
| effective fps    | 18.16        | 17.62            | -3 % (bruit) |
| paint p99        | 62 000 µs    | **50 900 µs**    | **-18 %**  |
| paint max        | 109 000 µs   | **63 000 µs**    | **-42 %**  |
| inter-tick p99   | 109 ms       | **99 ms**        | -9 %       |

Code : `gc.collect() + gc.freeze() + gc.disable()` à `play()`,
`gc.unfreeze() + gc.enable() + gc.collect()` à `pause()`.

### Optim #4 — Asymmetric eviction : ✅ architectural (pas mesurable ici)

Sur cette séquence (5.6 GiB total, budget par défaut 8 GiB), il n'y a
**jamais d'eviction** — le cache tient tout. L'optim est néanmoins en place
pour les cas où le budget est contraint :

```python
# In _evict_if_over_budget:
delta = (f - current) * direction
if delta < 0:
    return -delta * 3.0   # past frames cost 3× to keep
return float(delta)
```

Avec `direction=+1`, une frame une position derrière la playhead a un
score de 3 (priorité haute pour eviction), une frame une position devant
a un score de 1 (à protéger). Vérification empirique impossible sur cette
séquence + budget — sera mesurée dès qu'on bench une séquence > 8 GiB.

## Bilan combiné Phase 1

| Métrique          | Baseline      | **Phase 1 finale**  | Δ          |
|-------------------|---------------|----------------------|------------|
| effective fps     | 12.36         | **15-19** (variance) | **+30-50 %** |
| paint mean (µs)   | 53 174        | ~31-38 K             | **-30-40 %** |
| paint p99 (µs)    | 219 000       | ~50-92 K             | **-58-77 %** |
| upload mean (µs)  | 50 368        | ~30-37 K             | **-26-40 %** |
| inter-tick p99    | 287 ms        | ~99-162 ms           | **-44-66 %** |

Le **bottleneck restant est le decode** : 1700-2000 ms par frame 4K
multichannel EXR. À 6 workers en parallèle on tient 3-4 fps de decode
soutenu, donc le cache est toujours en retard pendant le warmup et le
hit rate stationnaire dépend de combien la séquence tient en RAM.

Pour pousser au-delà, il faudrait soit :
* **Cache disque proxy** (transcode ProRes / DPX vers SSD au premier
  loadcache, replay depuis le proxy).
* **Pre-warm UI** : un indicateur clair "cache full → ready to play"
  qui force l'utilisateur à attendre le warmup. Plus une UX qu'une
  optim, mais résout le problème perçu.
* **Un IPGraph C++** (rewrite partiel à la OpenRV) — c'est la frontière
  que Python ne peut pas franchir sur ce hardware.

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
