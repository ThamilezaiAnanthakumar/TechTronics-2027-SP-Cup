# AiPEX iteration log (README Section 8.5)

Protocol: AiPEX-dev only (62 videos, `shared/splits/aipex_dev.txt`), 5-fold CV stratified by country; AiPEX-test (26) untouched until after the freeze. Keep rule: CV MAE improves, paired bootstrap 95 % CI of the gain excludes 0 (5,000 resamples), UBFC dev pooled MAE not worse by > 0.1 bpm, India–Sierra Leone gap not wider by > 0.5 bpm. One hypothesis per experiment. Machine-readable rows: `log.csv`.

Reference points (all 88 videos, pre-split, from docs/REPRODUCTION.md): Muftee POS/right cheek 18.06; Thamil POS 18.61 -> + Kalman 13.05; published BKF 5.8.

## E001 — timing (diagnosis) — rejected
Hypothesis: a wrong frame-rate assumption scales every HR (expect est/ref clustered at ~1.2).
Finding: header and timestamps are exactly 15 fps in all 62 dev videos; the files contain no other timing information. The est/ref ratio is broad (Thamil POS 1.13–1.38 p10–p90; Muftee averaged-spectrum POS centred at 1.05 with tails 0.83–1.65), so there is no single scale factor. The upward bias of the window-mean estimator is better explained by noise-dominated window peaks averaging towards mid-band (42–180 bpm). Not pursued further: estimating a "true fps" by fitting dev would be tuning without independent evidence.
Side facts: crops are 200×200 px (dataset README says 60×30 / 25×25 — upscaled), the three crops have identical frame counts, recordings last 27–700 s at 15 fps.
Next: E000 baseline on dev CV with the harness, then ladder items 2–4 (pre-cropped input / skin mask, patch fusion, video-level estimator).
