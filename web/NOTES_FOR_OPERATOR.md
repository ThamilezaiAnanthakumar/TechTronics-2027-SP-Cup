# Notes for the operator (web/)

Decisions that need your approval. All numbers below come from **synthetic**
tests (Vitest, `web/tests/`); none has been checked on real video.

## 1. Web-local tunables — pending move to `shared/config.default.json`

They live in `web/src/localConfig.ts` (marked "pending move to shared/").
Nothing under `shared/` was changed.

| key | value | what it is |
|---|---|---|
| `motion_suppression` | true | README 4.8.2 switch (the README calls it optional / ablated) |
| `motion_beta` | 0.2 | β: attenuate only where normalised motion power M(f) > β (README gives no value) |
| `motion_gamma` | 0.7 | γ, README initial value |
| `motion_min_rms` | 0.002 | gate: band-passed head motion RMS (IOD / rad) below which M(f) is ignored |
| `motion_min_snr_db` | 0 | gate: motion must be periodic (spectral SNR of the motion ≥ 0 dB); landmark jitter (≈ −10 dB) never triggers suppression |
| `motion_harmonic` | true | also suppress 2·f_motion (sway below the band leaks into the band at its 2nd harmonic) |
| `motion_before_harmonic` | true | attenuate P_fused **before** harmonic summation (see 2a) |
| `motion_dilate_bpm` | 3 | running-max dilation of M(f) (see 2b) |
| `motion_patch_level` | true | apply the attenuation to each patch spectrum before SNR / peak (see 2c) |
| `motion_reject_step` | 0 (off) | frame rejection on large inter-frame motion; no measurable effect on synthetic jerks, kept off |
| `patch_erode_px` | 0 | front end: erode patch polygons (see section 3) |
| `subpatches` | true | front end emits 2×2 sub-patches in `TraceFrame.subpatches` (see section 5) |
| `subpatch_fusion` | true | core uses sub-patches as fusion units, with adaptive pooling |
| `pool_snr_db` | −3 | pool a family when a sub-patch SNR stays below this … |
| `pool_hold_s` | 5 | … for this long (README 4.4: 5 s); also the time before splitting again |
| `appearance_max_bhattacharyya` | 0.5 | README 4.1 appearance gate threshold (see section 4) |
| `appearance_ema` | 0.05 | appearance model update per second while the target is tracked |

## 2. Motion suppression: deviations from the literal README 4.8.2 text

The README says "attenuate H by (1 − γ·M(f))". Synthetic results (40 s, 30 fps,
face swaying between two differently coloured lights; 30 lagged 1 Hz outputs
after warm-up; count = outputs within 5 bpm of the true 72 bpm):

| case | off | literal (on H, after harmonic sum) | fused P before harmonic sum | + patch level (default) |
|---|---|---|---|---|
| 1.6 Hz sway, 2 % modulation | 2/30 (locks 96) | 18/30 (measured with ±6 bpm dilation; locks 48 when it fails) | 24/30 | **30/30** |
| 0.5 Hz sway, 8 % modulation, 2nd harmonic | 0/30 (locks 60) | – | 0/30 | **30/30** |
| 1.6 Hz sway, 4 % modulation, 50 % harmonic | 0/30 | – | 19/30 | **30/30** |
| 0.9 Hz / 1.6 Hz sway, 4 % pure sinusoid | 0/30 | – | 0/30 | 6/30, 1/30 |

a. **Before the harmonic sum.** Attenuating H after `H = P + α·P(2f)` leaves
   `α·P(f_m)` at `f_m/2`, so the tracker moves to the sub-harmonic (48 bpm for a
   96 bpm sway). Attenuating P_fused first fixes this; for the patch-level
   variant it is implied.

b. **Dilation of M(f) by ±3 bpm.** M(f) has the Hann main-lobe shape, so
   `(1 − γM)` leaves the lobe shoulders almost intact and the motion peak just
   shifts by ~3 bpm (96 → 93). ±6 bpm biased HR by +0.7 bpm when the motion
   harmonic lay 12 bpm from the pulse; ±3 bpm gave < 0.1 bpm bias.

c. **Patch level.** Applying the same attenuation to each patch spectrum before
   its SNR / peak / coherence are measured means the README 4.6 weights (formula
   unchanged) favour the patches whose spectrum is pulse-dominated after
   suppression. The fused spectrum then carries exactly the same attenuation as
   in the literal version (Σ wᵢ·A·Pᵢ = A·Σ wᵢ·Pᵢ); only the weights differ.

d. **Limit of γ = 0.7.** Because H is never attenuated below 0.3, a motion
   peak more than ≈ 3× the pulse peak in every patch still wins (last row of the
   table). I did **not** change γ. If the ablation on real data shows sway
   capture, the evidence would be for γ → 0.85–0.9 or for a motion-regression
   step (regress the integrated pose out of the RGB traces) — your call.

## 3. Stability under small motion (synthetic face through the FrontEnd)

Harness: `tests/faceSynth.ts` renders a synthetic face (soft-edged brows,
eyes, nostrils, lips touching the patch borders, warm side light, 1 px skin
texture, sensor noise SD 3, pulse 0.15 % on G) at IOD 70 px in a 640×480 frame,
15 fps, 40 s; landmarks go through the same 1-Euro filter as `pipeline.ts`.
Metric: outputs after warm-up with status OK and |HR − 72| < 5, summed over 3
seeds (90 outputs per row).

| condition | patch erosion 0 px | 1 px | 2 px |
|---|---|---|---|
| still | 85/90 | 81/90 | 0/90 |
| ±2 px white jitter + 8 px / 0.05 Hz drift + roll | 87/90 | 78/90 | 0/90 |
| ±2 px correlated jitter (AR 0.8) + drift | 86/90 | 61/90 | 1/90 |
| ±3 px correlated jitter (AR 0.9) + drift | 84/90 | 56/90 | 5/90 |

Findings:
- Landmark jitter of ±1–3 px (white or temporally correlated) plus slow drift
  does **not** reduce the OK fraction on this synthetic face. The 1-Euro filter
  plus skin/texture selection and the trimmed mean already absorb it. I could
  not reproduce the "HR disappears with slight head movement" symptom with
  jitter alone; periodic motion that modulates the light (task 1) and small
  faces (below) reproduce it.
- Patch erosion hurts (fewer pixels; at 2 px patches fall below `npix_min`), so
  `patch_erode_px` stays 0. Trimmed-mean fraction and occlusion EMA are shared
  constants (`trimmed_mean_frac`, `occlusion_ema_s`); I did not change them and
  found no evidence that they should change.
- **Small faces.** At IOD 40 px (face ≈ 1 m from a 640×480 webcam) the same
  harness gave 0/90: half the patches are below `npix_min` (80 px at VGA) and
  the rest are LOW_SNR. Before the fix in this commit series those
  small-but-unoccluded lower-face patches also switched the status to MASKED,
  which is wrong (README 4.3.5 says *occluded*). The core now counts a patch
  as occluded only when it is invalid **and** its skin ratio is below
  `occlusion_off`. Whether `npix_min_qvga` = 20 is too strict for distant faces
  is your call (shared constant; evidence: synthetic only).

## 4. Appearance gate (README 4.1 / 11.2.4)

`src/front/appearance.ts`: 256-bin histogram (8×8 normalised-chromaticity
r, g bins × 4 soft brightness bins) of the central half of the face box, taken
from a 160×120 thumbnail on the main thread; Bhattacharyya distance (OpenCV
definition). The model is learned during the first 2 s (or at a click) and
updated once per second while tracked (EMA 0.05). It is checked on
re-acquisition and whenever the nearest face jumps by > 0.3 face sizes; a
rejected candidate is not adopted (a click still overrides).
Synthetic distances: same face 0.04, same face with −15 % exposure 0.27, three
other skin tones 0.86–0.91 → threshold 0.5. **Untested on real faces**; two
people with similar skin tone under the same light will likely pass the gate,
as with any colour-only gate. The README gives no threshold; please confirm.

## 5. Sub-patches (README 4.2) and adaptive pooling (README 4.4)

- **Contract extension (needs approval).** Sub-patches travel in a new optional
  field `TraceFrame.subpatches` (ids `patch_id*4 + row*2 + col`, as in
  `shared/patches.json` `subpatch_split`); `patches` still carries the 10
  parents, so other readers are unaffected. Please ratify or change this in
  `shared/traceframe.schema.json` (which does not exist yet).
- Split: each patch's pixels are divided at their centroid along the eye-line
  axes (roll removed). Selection masks are the patch's; occlusion EMA /
  hysteresis and `npix_min` apply per sub-patch.
- Pooling rule as implemented: a family is merged into its parent as soon as
  one sub-patch fails coverage (below `npix_min` in too many frames — this is
  the README's "median npix below a threshold"), or after one sub-patch's SNR
  stays below `pool_snr_db` for 5 s; it splits again after all four have been
  fine for 5 s. `top_patches` in the CSV-side rows lists parent ids;
  `n_patches_used` counts fusion units.
- `pool_snr_db` = −3 rather than 0: at 0 dB, ordinary noisy sub-patches pooled
  their family and re-mixed a flickering corner into the parent (synthetic:
  7/30 vs 30/30 good outputs). An alternative you may prefer: pool only the
  weak sub-patches with each other instead of the whole family.
- Effect (synthetic): a colour flicker at 1.6 Hz in one corner of every patch
  captures patch-level fusion (0/30 outputs within 5 bpm) but not sub-patch
  fusion (30/30). On the uniform synthetic face at IOD 70 px most families are
  pooled (several sub-patches are below 80 px), so results equal patch level
  (85/90 vs 85/90 still, 85/90 vs 86/90 jittered). The 1 Hz analysis now takes
  up to ≈140 ms in the worker (50 analyses per tick) vs ≈40–80 ms before.
- Small-face numbers after the MASKED fix (same harness, IOD 40 px, 3 seeds):
  39/90 still and 30/90 with jitter (0/90 before, when 31 of the 90 outputs were
  wrongly MASKED, which also zeroes the group-C weights).
