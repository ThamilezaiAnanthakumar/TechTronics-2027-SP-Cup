# UBFC2 failure analysis (README Section 2.2)

Source: `tools/diagnose_ubfc_outliers.m` on the seeded `run_ubfc_all` outputs (commit 318426d, Viola-Jones + KLT front end, POS/combined, 10 s windows). Figures `outputs/diagnose_ubfc/<subject>.png` (spectrogram with PPG reference, per-ROI SNR, face-box motion; no face images). Analysis only — subjects 27, 41 and 49 are in the UBFC test split and are not used for tuning.

| Subject | Split | MAE | median AE | windows >5 bpm | SNR combined (dB) | box motion 95th pct (px/frame) | ref HR median / range | Failure type |
|---|---|---|---|---|---|---|---|---|
| 15 | dev | 24.8 | 26.3 | 76 % | −4.4 | 0.22 (spikes ~1/s) | 114 / 15 | **Low SNR + front-end artefact**: weak pulse in all ROIs (−2.6 to −3.9 dB); the estimate locks to a competing 75–80 bpm band while the reference is 110–120. Box jumps every ~1 s (Viola-Jones re-detection every 30 frames) inject step artefacts. |
| 49 | test | 5.5 | 1.5 | 17 % | 0.8 | 0.87 | 86 / 11 | **Motion episodes**: accurate most of the time, errors in bursts that coincide with motion (95th pct 2–3x the controls; scale change 0.57 %). |
| 27 | test | 5.3 | 0.7 | 13 % | −0.7 | 1.38 | 88 / 28 | **Motion episodes + fast HR change** (reference spans 28 bpm); 6 % harmonic windows. |
| 41 | test | 5.1 | 1.0 | 14 % | 0.1 | 2.12 | 101 / 14 | **Motion episodes** (largest motion of the set); 5 % harmonic windows. |
| 1 (control) | dev | 1.6 | 1.5 | 0 % | 3.0 | 0.25 | 107 / 13 | — |
| 12 (control) | dev | 1.2 | 0.9 | 2 % | 6.0 | 0.44 | 66 / 6 | — |

Implications (design, not tuning):
- Subject 15: the RetinaFace front end with 1-Euro ROI smoothing removes the 1 Hz box steps; SNR-weighted top-k fusion and physiology-constrained tracking (README 4.6–4.8) should keep the track near the true rate once acquired. Re-check on dev with the integrated pipeline.
- 49 / 27 / 41: errors are episodic, not systematic — the case for motion suppression and Viterbi tracking (README 4.8), and for reporting confidence/status instead of a wrong number.
- The 30-frame re-detection step artefact is a property of the Viola-Jones front end in every subject; it is harmless at high SNR (controls) but dominant at low SNR.
