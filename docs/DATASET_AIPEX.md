# AiPEX rppg_biases dataset (Dasari et al., npj Digit Med 2021)

- Location: `D:\SPCup Data\dataset\AiPEX rppg_biases\rppg_biases-main` (`cfg.aipexRoot`).
- 44 subjects from India and 44 from Sierra Leone. Per subject, three pre-cropped ROI videos: `videoN_forehead.avi` (60x30 px), `videoN_leftcheek.avi`, `videoN_rightcheek.avi` (25x25 px). 132 files per country.
- Reference: one mean HR per video (Masimo, FDA-approved) in `Ground_Truth/<Country>_GT.csv`.
- **One video id = one subject.** The dataset README states that the three videos are "generated from the facial video of each subject", so video ids are distinct subjects; `make_aipex_split` also asserts the ids are unique per country.

## Frozen split (README Section 8.5)
- `tools/make_aipex_split.m`, `rng(2027,'twister')`, 30 % per country held out: **13 India + 13 Sierra Leone = 26 test**, 62 dev.
- `shared/splits/aipex_test.txt`, `shared/splits/aipex_dev.txt`. The script refuses to overwrite an existing split.
- Nobody looks at per-video results of AiPEX-test until the final run after the freeze.
