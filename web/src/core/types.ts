// TraceFrame contract (README 11.1) and core outputs.

export interface TracePatch {
  id: number;          // stable ID from shared/patches.json
  group: number;       // 0=A, 1=B, 2=C
  cluster: number;     // 0 (pigmentation split not active in the web MVP)
  rgb_mean: [number, number, number]; // trimmed mean, 0..255
  npix: number;
  valid: number;       // 1 = usable this frame
  skin_ratio: number;  // selected / total pixels in the patch
}

export interface TraceFrame {
  t_s: number;           // real capture/media timestamp, seconds, monotonic
  face_status: number;   // 0 = no face, 1 = tracked, 2 = re-acquired this frame
  track_quality: number; // 0..1
  motion: [number, number, number, number]; // dx, dy (/IOD), dscale, droll (rad)
  n_patches: number;
  patches: TracePatch[];
}

export type Status = 'OK' | 'WARMUP' | 'NO_FACE' | 'MASKED' | 'LOW_SNR' | 'LOW_COVERAGE';

/** One row of processed_output.csv (README 4.9). */
export interface HrRow {
  time_s: number;
  hr_bpm: number; // NaN unless OK / MASKED
  confidence: number;
  status: Status;
  n_patches_used: number;
  top_patches: number[];
}

export interface PatchQuality {
  id: number;
  coverage: number;
  snr_db: number;
  peak_bpm: number;
  weight: number;   // normalised fusion weight among the used top-k (0 if unused)
  used: boolean;
}

/** Everything the UI needs from one 1 Hz analysis step. */
export interface CoreTick {
  t_s: number;          // end of the analysis window (current time)
  status: Status;       // status of this (newest) window
  fs_proc: number;
  patches: PatchQuality[];
  gridBpm: Float64Array;   // HR grid
  emission: Float64Array;  // harmonic-summed fused spectrum on the grid, max-normalised
  trackBpm: number;        // current best (non-lagged) Viterbi state, bpm
  bvp: Float64Array;       // fused BVP of the newest window (z-scored)
  row?: HrRow;             // lagged fixed-lag decision, if one was made this tick
  /** README 4.8.2: band-passed head motion RMS, its dominant rate, and whether suppression acted. */
  motion?: { rms: number; peakBpm: number; active: boolean };
}

/**
 * Signal core interface. The interim TypeScript core implements it now;
 * fvcore-WASM replaces it later with a one-line change (README Section 16).
 */
export interface SignalCore {
  reset(): void;
  /** Push one TraceFrame; returns the 1 Hz ticks completed by this frame. */
  push(frame: TraceFrame): CoreTick[];
  /** Flush remaining lagged decisions at end of input. */
  finish(): HrRow[];
}
