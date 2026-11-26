# Parameter and behaviour changelog

Every change to pipeline parameters or behaviour, with reason and measured effect. Changes after the freeze (28 Nov 2026 18:00) must also say why they were unavoidable.

| Date | Change | Files | Reason | Measured effect |
|---|---|---|---|---|
| 2026-11-26 | Fixed RNG per video: `rng(0,'twister')` at the start of each video, previous state restored on exit; cache keys gain `rngSeed` so unseeded caches are re-extracted | `src/face/extractRGBTraces.m`, `src/face/process_video.m`, `src/pipeline/faceTraces.m` | `estgeotform2d` (MSAC) samples points at random and nothing seeded the RNG, so face boxes differed run to run (subject 27: 107 of 1431 frames, up to 4 px, between two identical R2024b runs). UBFC2 POS/combined pooled MAE: original 2.165 (22 Nov) vs unseeded R2026b re-run 2.187. | see Phase 0 reproduction log (to be filled from `run_ubfc_all` seeded runs) |
