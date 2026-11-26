// 1-Euro filter (Casiez et al., CHI 2012) applied per coordinate to the target's
// landmarks before the patch polygons are built (README Section 4.2: 1-Euro
// smoothing of patch vertices, min cutoff 1 Hz, beta 0.02 — same values as
// Sulojan's MATLAB OneEuroFilter). Coordinates are filtered in pixels so beta has
// the same meaning as in MATLAB.

function alpha(cutoffHz: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dt);
}

export class OneEuroLandmarks {
  private x: Float64Array | null = null;   // filtered value (px)
  private dx: Float64Array | null = null;  // filtered derivative (px/s)
  private tPrev = NaN;

  constructor(
    private minCutoff = 1.0,
    private beta = 0.02,
    private dCutoff = 1.0,
  ) {}

  reset() { this.x = null; this.dx = null; this.tPrev = NaN; }

  /** lm: normalised [x0, y0, x1, y1, ...]; W x H: pixel size; t: seconds. Returns normalised, smoothed. */
  filter(lm: Float32Array, W: number, H: number, t: number): Float32Array {
    const n = lm.length;
    if (!this.x || this.x.length !== n || !(t > this.tPrev)) {
      this.x = new Float64Array(n);
      for (let i = 0; i < n; i++) this.x[i] = lm[i] * (i % 2 === 0 ? W : H);
      this.dx = new Float64Array(n);
      this.tPrev = t;
      return lm;
    }
    const dt = t - this.tPrev;
    this.tPrev = t;
    const aD = alpha(this.dCutoff, dt);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const s = i % 2 === 0 ? W : H;
      const v = lm[i] * s;
      const d = (v - this.x[i]) / dt;
      this.dx![i] = this.dx![i] + aD * (d - this.dx![i]);
      const a = alpha(this.minCutoff + this.beta * Math.abs(this.dx![i]), dt);
      this.x[i] = this.x[i] + a * (v - this.x[i]);
      out[i] = this.x[i] / s;
    }
    return out;
  }
}
