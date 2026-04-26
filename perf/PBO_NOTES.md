# Note d'expérimentation — PBO async upload

*Date : 2026-04-26 — branche `perf-baseline-bench`, commit `6c0fc81` puis revert*

## Hypothèse de départ

`glTexSubImage2D(..., pixels_ptr)` mesuré à **50 ms** pour un upload 4K
float16 (32 MiB) sur AMD Radeon 780M. Le main thread Qt est bloqué
pendant ce temps. Théorie OpenRV : passer par un Pixel Buffer Object
(`GL_PIXEL_UNPACK_BUFFER`) avec ping-pong de 2 PBOs permet au driver de
faire un DMA asynchrone, libérant le CPU.

Pattern implémenté :

```
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_back)
glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, NULL, GL_STREAM_DRAW)  # orphan
ptr = glMapBufferRange(.., GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT)
ctypes.memmove(ptr, pixels.ctypes.data, nbytes)
glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)
glBindTexture(GL_TEXTURE_2D, image_tex)
glTexSubImage2D(.., NULL)   # reads from bound PBO, can DMA async
```

## Résultat mesuré

Sur la même séquence (4K UHD multichannel EXR, 90 frames, 3 passes,
target 24 fps, 6 workers, 8 GiB cache), AMD Radeon 780M :

| Métrique         | Baseline (sync) | PBO ping-pong | Delta  |
|------------------|-----------------|---------------|--------|
| effective fps    | 12.36           | **12.00**     | -3 %   |
| cache hit rate   | 71.5 %          | 74.8 %        | +3 pts |
| upload mean (µs) | 50 368          | **57 225**    | +14 %  |
| upload p99 (µs)  | 219 000         | 250 000       | +14 %  |
| paint mean (µs)  | 53 174          | 58 294        | +10 %  |

**Le PBO est plus lent.** ~7 ms de plus en moyenne, +30 ms au p99.

## Pourquoi ça ne marche pas sur cet hardware

L'AMD Radeon 780M est un **iGPU à mémoire unifiée** : le GPU et le CPU
partagent la même DRAM physique. Conséquences :

1. **Pas de DMA cross-bus** — il n'y a pas de PCIe à traverser, donc
   rien à pipeliner. Le « upload » est un memcpy DRAM → DRAM, qui est
   limité par la bande passante mémoire (~50-80 GB/s) et qui se fait
   sur le CPU côté driver.
2. **Le PBO ajoute une copie** — `memmove(pixels → mapped PBO)` puis
   `driver_copy(PBO → texture)`. Là où le path direct ne fait qu'**une**
   copie (`pixels → texture`), le PBO en fait deux.
3. **`glTexSubImage2D(NULL)` reste bloquant** — sur ce driver, la
   transformation de format pixel (`HALF_FLOAT` source → `RGBA16F`
   storage tilé) est faite synchronément côté CPU avant le DMA. Le
   pipelining en aval n'aide donc pas.

## Quand le PBO aiderait

* **GPU dédié + PCIe** (NVIDIA / AMD desktop discrete). Le DMA PCIe est
  lent (~16 GB/s en x16 PCIe 4.0) et asynchrone. Le pipelining cache
  cette latence.
* **Workload qui fait du travail CPU utile entre l'upload et le draw**
  (par exemple décoder / pré-traiter la frame N+1 pendant que la frame
  N upload). Notre `paintGL` actuel n'a quasiment rien à faire entre
  l'upload et le draw — le bénéfice est nul même sur GPU dédié dans
  cette archi.

## Décision

Code reverté au path direct `glTexSubImage2D` synchrone. À retester
**uniquement** quand on aura :

1. Un GPU dédié sous la main, OU
2. Un re-design de la boucle render qui produit du travail CPU
   parallèle à l'upload (par exemple un IPGraph à la OpenRV qui prépare
   la frame N+1).

Le scaffolding du recorder reste en place (`bench/`) — c'est lui qui
nous a permis de chiffrer cette régression en 5 minutes au lieu de
deviner à l'œil.

## Recommandation pour les optims suivantes

Vu que **l'upload domine** mais que **le PBO ne peut pas l'accélérer
sur cet hardware**, les vraies leviers restent :

1. **OIIO threading** (`oiio.attribute("threads", N)`) — accélère le
   decode, ce qui réduit la pression sur le cache et permet de garder
   plus de frames en RAM. Indirectement, plus de cache hits → moins de
   pénalité quand on a un upload coûteux.
2. **`gc.freeze()` + `gc.disable()` durant playback** — supprime les
   spikes p99 dus aux pauses GC, qui sont visibles dans la baseline
   (`paint p99` = 4× le mean).
3. **Asymmetric eviction** — garder en priorité les frames devant la
   playhead pour éviter le re-decode.

Le upload restera plafonné à ~50 ms tant qu'on tourne sur cet iGPU.
**Cible Phase 1 ajustée** : tenir 24 fps **une fois le cache chaud**,
en acceptant que le warmup soit lent (decode-bound). Sur GPU dédié, on
visera 24 fps full hot + cold.
