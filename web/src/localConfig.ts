// localConfig.ts — web-local tunables NOT (yet) in shared/config.default.json.
//
// pending move to shared/ (operator approval)
//
// Each value is a web-side choice made on synthetic tests only; none has been
// validated on real video. They are listed in web/NOTES_FOR_OPERATOR.md. When
// the operator approves, move them to shared/config.default.json so MATLAB and
// fvcore read the same values (README 11.1: no platform hard-codes parameters).

export interface LocalConfig {
  // ---- README 4.8.2 motion suppression (optional, ablated) ----
  /** Master switch for spectral motion suppression. */
  motion_suppression: boolean;
  /** beta: attenuate only where the normalised motion spectrum M(f) > beta. */
  motion_beta: number;
  /** gamma: H(f) *= (1 - gamma * M(f)); gamma < 1 so H is never zeroed. README initial 0.7. */
  motion_gamma: number;
  /**
   * Gate: suppression is active only when the band-passed head motion has an
   * RMS amplitude above this (sum over channels; dx, dy in IOD, dscale, roll in
   * rad). Below it the "motion spectrum" is landmark noise and must not carve
   * holes into H.
   */
  motion_min_rms: number;
  /**
   * Gate: the wide-band motion spectrum must be periodic enough: its SNR
   * (README 4.6 definition applied to motion: power within +-snr_halfwidth of
   * the motion peak and its 2nd harmonic over the rest) must reach this, in dB.
   * Landmark jitter is broadband (about -10 dB) and must not trigger suppression.
   */
  motion_min_snr_db: number;
  /**
   * Also suppress at 2x the motion frequencies (the intensity a moving face
   * reflects is not a linear function of position, so sway at f_m leaks into
   * 2 f_m too).
   */
  motion_harmonic: boolean;
  /**
   * Apply the attenuation to P_fused before harmonic summation (true) or to H
   * after it (false, literal README 4.8.2). Before is better: otherwise the
   * alpha P(2f) term re-injects the motion peak at f_m / 2.
   */
  motion_before_harmonic: boolean;
  /**
   * Dilate M(f) by a running maximum over +-this many bpm. A raw M(f) has the
   * Hann main-lobe shape, so (1 - gamma M) leaves the shoulders of a motion
   * peak almost untouched and the peak simply moves a few bpm. 0 = off.
   */
  motion_dilate_bpm: number;
  /**
   * Also attenuate each patch spectrum before its SNR / peak are measured, so
   * the README 4.6 weights (formula unchanged) see motion-suppressed spectra.
   */
  motion_patch_level: boolean;
  /**
   * Frame rejection: blank all patch samples of a frame whose inter-frame
   * translation hypot(dx, dy) (IOD) or |dscale| exceeds this. 0 disables.
   * Short gaps are then bridged by the resampler; long ones lower coverage.
   * Off by default: on synthetic head jerks (0.25 s every 0.8-1.5 s with
   * transient patch errors up to 160 units) it changed nothing measurable
   * (HR MAE 2.38 vs 2.42 bpm), because the Kalman + POS chain already
   * rejects the spikes and the interpolated gaps cost pulse cycles.
   */
  motion_reject_step: number;

  // ---- front end (README 4.3, web implementation choices) ----
  /** Erode each patch polygon by this many pixels at its edges (0 = off). */
  patch_erode_px: number;

  // ---- target selection (README 4.1 / 11.2.4 appearance gate) ----
  /** Reject a re-acquisition candidate whose colour-histogram Bhattacharyya distance exceeds this. */
  appearance_max_bhattacharyya: number;
  /** EMA weight of each confirmed target observation in the appearance model (one per second). */
  appearance_ema: number;
}

export const LOCAL_CONFIG: LocalConfig = {
  motion_suppression: true,
  motion_beta: 0.2,
  motion_gamma: 0.7,
  motion_min_rms: 0.002,
  motion_min_snr_db: 0,
  motion_harmonic: true,
  motion_before_harmonic: true,
  motion_dilate_bpm: 3,
  motion_patch_level: true,
  motion_reject_step: 0,
  patch_erode_px: 0,
  appearance_max_bhattacharyya: 0.5,
  appearance_ema: 0.05,
};
