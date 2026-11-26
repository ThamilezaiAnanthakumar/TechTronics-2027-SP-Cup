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

## 3. Front end (task 2) — see the section below, filled in as work proceeds.
