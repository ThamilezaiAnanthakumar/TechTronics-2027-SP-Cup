# Source of `contrib/snr_fusion/`

Imported **unmodified** (byte-identical; checked with `git hash-object` against the commit tree)
from the teammate's branch. Nothing under `config/` or `src/` in this folder has been edited.

| | |
|---|---|
| Author | Mohamed Irshad Nadha |
| Branch | `origin/c2-skin-tone` |
| Commit 1 | `3ff7a1529090c6e3e4d463472f4600b37f28bef2` — 2026-11-27 10:58 +0530 — "C2: Python skin-tone evaluation pipeline with ITA, background reference and SNR fusion" |
| Commit 2 | `425fdc12082024659cc811cbae38e78e7ccf54a7` — 2026-11-27 12:01 +0530 — "added kalman filter + SNR" (copied state) |
| Files | `src/python/{README.md, run_c2.py, ita_survey.py}`, `src/python/facevitals/{__init__,datasets,face,rppg,tracker,retinaface}.py`, `config/c2_config.json` |

The SNR-fusion code (`rppg.region_snr`, `rppg.sanity_hr_snr`, `SNR_*` constants), the AiPEX
loader and the AiPEX results table in `src/python/README.md` are **identical in both commits**;
commit 2 added the RetinaFace/KLT tracker port (not used on AiPEX: crops skip face detection),
`kalman_denoise` and the `pos_kf*` variants, and two keys in `c2_config.json`
(`face_module`, `tracker_max_width`).

## Command that produces the table (from `src/python/README.md`, "Run")

```bash
# 5 independent processes, then merge + statistics
python src/python/run_c2.py --shard 1/5      # ... up to --shard 5/5
python src/python/run_c2.py --analyze-only --method compare   # pos / pos_eq / pos_snr per dataset & subject
python src/python/run_c2.py --analyze-only --method pos_snr   # ITA-group table with SNR fusion
python src/python/run_c2.py --analyze-only --method pos       # ITA-group table, baseline
```

Default `--datasets` is UBFC MPU CLBP AiPEX; AiPEX rows do not depend on the other datasets, so the
reproduction used `--datasets AiPEX`. `REPO` is the folder two levels above `src/python`
(`face.py`: `Path(__file__).parents[3]`); it must contain `config/c2_config.json` and
`config/face_config.json` (the latter from the teammate's branch, unchanged), and the
outputs go to `REPO/outputs/tables/` and `REPO/outputs/traces_py/`. Data root = folder that
contains `rppg_biases-main/` (`--data-root` or `FACEVITALS_DATA`).

The overall AiPEX numbers in their README (MAE, median, within 5 bpm, bias, 72/10, Wilcoxon) are
**not printed by any function in `run_c2.py`**: `compare()` prints only the mean MAE and SUC@5 per
method plus the per-subject table (`outputs/tables/c2_method_compare.csv`); median, bias and the
better/worse counts were derived from `c2_subjects.csv` outside the committed code.
Our recomputation of those columns is `tools/snr_fusion_analysis.py`.

Reproduction run and results: `docs/SNR_FUSION.md`.
