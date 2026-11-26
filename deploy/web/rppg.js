// rppg.js — JavaScript port of the MATLAB rPPG signal chain (src/signal/*.m).
// Pure functions, no DOM: runs in the browser and in Node (tests/parity.mjs).
//
//   rgbToPulse        <- src/signal/rgbToPulse.m        (GREEN | CHROM | POS)
//   preprocessPulse   <- src/signal/preprocessPulse.m   (detrend, bandpass, z-norm)
//   estimateHR        <- src/signal/estimateHR.m        (sliding-window FFT peak)
//   LiveEstimator     streaming wrapper: irregular camera timestamps -> uniform
//                     grid -> same chain -> HR of the most recent window.

// Mirrors config/getConfig.m. Keep the two in sync.
export const DEFAULT_CFG = Object.freeze({
  hrBand: [0.7, 3.0],      // Hz (42-180 bpm)
  filterOrder: 4,          // Butterworth, zero-phase
  detrendLambda: 100,      // smoothness-priors detrending
  posWindowSec: 1.6,
  hrWindowSec: 10,
  hrStepSec: 1,
  nfftMin: 16384,
  methods: ['GREEN', 'CHROM', 'POS'],
  roiSet: ['forehead', 'leftCheek', 'rightCheek', 'combined'],
});

// ---------------------------------------------------------------- helpers
const mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
function std(a) { // sample std (N-1), as MATLAB std
  const m = mean(a); let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; }
  return a.length > 1 ? Math.sqrt(s / (a.length - 1)) : 0;
}

// Minimal complex arithmetic for the filter design
const C = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1]],
  mul: (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
  div: (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; },
  scale: (a, k) => [a[0] * k, a[1] * k],
  sqrt: (a) => { const r = Math.hypot(a[0], a[1]); const re = Math.sqrt((r + a[0]) / 2); const im = Math.sign(a[1] || 1) * Math.sqrt(Math.max(0, (r - a[0]) / 2)); return [re, im]; },
};

// ------------------------------------------------------- Butterworth design
// Same as MATLAB butter(order, band/(fs/2), 'bandpass') -> zp2sos:
// analog prototype, prewarp, lp2bp, bilinear. Returns biquads (b, a) + gain.
const filterCache = new Map();
export function butterBandpass(order, band, fs) {
  const key = `${order}|${band[0]}|${band[1]}|${fs}`;
  if (filterCache.has(key)) return filterCache.get(key);
  const fs2 = 4; // MATLAB designs at fs = 2 -> bilinear uses 2*fs = 4
  const u1 = fs2 * Math.tan(Math.PI * (band[0] / (fs / 2)) / 2);
  const u2 = fs2 * Math.tan(Math.PI * (band[1] / (fs / 2)) / 2);
  const Bw = u2 - u1, Wo2 = u1 * u2;

  const polesA = [];
  for (let k = 1; k <= order; k++) {
    const th = Math.PI * (2 * k + order - 1) / (2 * order);
    const pb = C.scale([Math.cos(th), Math.sin(th)], Bw / 2);
    const disc = C.sqrt(C.sub(C.mul(pb, pb), [Wo2, 0]));
    polesA.push(C.add(pb, disc), C.sub(pb, disc));
  }
  // bilinear: z = (fs2 + s) / (fs2 - s); gain = Bw^n * fs2^n / prod(fs2 - p)
  let den = [1, 0];
  const polesD = polesA.map((s) => { den = C.mul(den, C.sub([fs2, 0], s)); return C.div(C.add([fs2, 0], s), C.sub([fs2, 0], s)); });
  const gain = C.div([Math.pow(Bw * fs2, order), 0], den)[0];

  // biquads as zp2sos: take pole pairs from closest to the unit circle down,
  // give each the two remaining zeros (n at +1, n at -1) nearest to it, then
  // order sections by increasing pole radius ('up'). The cascade is the same
  // filter either way; the grouping only changes filtfilt's edge behaviour.
  const upper = polesD.filter((p) => p[1] > 1e-12).sort((p, q) => Math.hypot(...q) - Math.hypot(...p));
  if (upper.length !== order) throw new Error('butterBandpass: unexpected real poles');
  const zeros = [...Array(order).fill(1), ...Array(order).fill(-1)];
  const sos = upper.map((p) => {
    const pick = () => {
      let best = 0;
      zeros.forEach((z, i) => { if (Math.hypot(p[0] - z, p[1]) < Math.hypot(p[0] - zeros[best], p[1])) best = i; });
      return zeros.splice(best, 1)[0];
    };
    const z1 = pick(), z2 = pick();
    return { r: Math.hypot(...p), b: [1, -(z1 + z2), z1 * z2], a: [1, -2 * p[0], p[0] * p[0] + p[1] * p[1]] };
  }).sort((s, t) => s.r - t.r);
  const out = { sos, gain };
  filterCache.set(key, out);
  return out;
}

// Direct form II transposed biquad with initial state, in place on a copy
function biquad(sec, x, zi0, zi1) {
  const [b0, b1, b2] = sec.b, [, a1, a2] = sec.a;
  const y = new Float64Array(x.length);
  let z1 = zi0, z2 = zi1;
  for (let n = 0; n < x.length; n++) {
    const xn = x[n], yn = b0 * xn + z1;
    z1 = b1 * xn - a1 * yn + z2;
    z2 = b2 * xn - a2 * yn;
    y[n] = yn;
  }
  return y;
}

// Zero-phase filtering as MATLAB filtfilt(sos, g, x): each section is
// filtfilt'ed in turn with odd reflection padding and steady-state initial
// conditions scaled by the first padded sample. As in MATLAB, the padding
// length comes from the whole filter order: nfact = 3 * (2 * nSections).
// Verified against MATLAB R2024b to ~1e-14.
export function filtfilt({ sos, gain }, x) {
  const nfact = 6 * sos.length;
  let y = Float64Array.from(x);
  if (y.length <= nfact) throw new Error(`filtfilt: signal too short (${y.length})`);
  for (const sec of sos) {
    const [b0, b1, b2] = sec.b, [, a1, a2] = sec.a;
    const B1 = b1 - b0 * a1, B2 = b2 - b0 * a2;
    const zi0 = (B1 + B2) / (1 + a1 + a2), zi1 = B2 - a2 * zi0;
    const N = y.length, P = new Float64Array(N + 2 * nfact);
    for (let i = 0; i < nfact; i++) {
      P[i] = 2 * y[0] - y[nfact - i];
      P[N + nfact + i] = 2 * y[N - 1] - y[N - 2 - i];
    }
    P.set(y, nfact);
    let f = biquad(sec, P, zi0 * P[0], zi1 * P[0]).reverse();
    f = biquad(sec, f, zi0 * f[0], zi1 * f[0]).reverse();
    y = f.slice(nfact, nfact + N);
  }
  const g2 = gain * gain; // forward + backward pass
  for (let i = 0; i < y.length; i++) y[i] *= g2;
  return y;
}

export function bandpassFilter(x, fs, cfg = DEFAULT_CFG) {
  return filtfilt(butterBandpass(cfg.filterOrder, cfg.hrBand, fs), x);
}

// --------------------------------------------- smoothness-priors detrending
// y = x - (I + lambda^2 D2'D2) \ x, solved with a banded (pentadiagonal) LDL'.
export function detrendSmoothnessPriors(x, lambda) {
  const N = x.length;
  if (N < 3) { const m = mean(x); return Float64Array.from(x, (v) => v - m); }
  const L2 = lambda * lambda;
  const a0 = new Float64Array(N).fill(1), a1 = new Float64Array(N - 1), a2 = new Float64Array(N - 2);
  for (let r = 0; r < N - 2; r++) {
    a0[r] += L2; a0[r + 1] += 4 * L2; a0[r + 2] += L2;
    a1[r] += -2 * L2; a1[r + 1] += -2 * L2;
    a2[r] += L2;
  }
  const D = new Float64Array(N), l1 = new Float64Array(N), l2 = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    l2[i] = i >= 2 ? a2[i - 2] / D[i - 2] : 0;
    l1[i] = i >= 1 ? (a1[i - 1] - (i >= 2 ? l2[i] * D[i - 2] * l1[i - 1] : 0)) / D[i - 1] : 0;
    D[i] = a0[i] - (i >= 1 ? l1[i] * l1[i] * D[i - 1] : 0) - (i >= 2 ? l2[i] * l2[i] * D[i - 2] : 0);
  }
  const z = new Float64Array(N);
  for (let i = 0; i < N; i++) z[i] = x[i] - (i >= 1 ? l1[i] * z[i - 1] : 0) - (i >= 2 ? l2[i] * z[i - 2] : 0);
  for (let i = 0; i < N; i++) z[i] /= D[i];
  const trend = new Float64Array(N);
  for (let i = N - 1; i >= 0; i--) {
    trend[i] = z[i] - (i + 1 < N ? l1[i + 1] * trend[i + 1] : 0) - (i + 2 < N ? l2[i + 2] * trend[i + 2] : 0);
  }
  return Float64Array.from(x, (v, i) => v - trend[i]);
}

// ------------------------------------------------------------ rgbToPulse
// rgb: array of [R, G, B] rows (spatial means per frame).
export function rgbToPulse(rgb, fs, method, cfg = DEFAULT_CFG) {
  const N = rgb.length;
  const R = Float64Array.from(rgb, (v) => v[0]);
  const G = Float64Array.from(rgb, (v) => v[1]);
  const B = Float64Array.from(rgb, (v) => v[2]);
  switch (method.toUpperCase()) {
    case 'GREEN': {
      const mg = mean(G);
      return Float64Array.from(G, (g) => g / mg - 1);
    }
    case 'CHROM': {
      const mr = mean(R), mg = mean(G), mb = mean(B);
      const X = new Float64Array(N), Y = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        const r = R[n] / mr, g = G[n] / mg, b = B[n] / mb;
        X[n] = 3 * r - 2 * g;
        Y[n] = 1.5 * r + g - 1.5 * b;
      }
      const Xf = bandpassFilter(X, fs, cfg), Yf = bandpassFilter(Y, fs, cfg);
      const alpha = std(Xf) / std(Yf);
      return Float64Array.from(Xf, (v, i) => v - alpha * Yf[i]);
    }
    case 'POS': {
      const l = Math.round(cfg.posWindowSec * fs);
      const H = new Float64Array(N);
      const s1 = new Float64Array(l), s2 = new Float64Array(l);
      for (let n = 0; n + l <= N; n++) {
        let mr = 0, mg = 0, mb = 0;
        for (let k = n; k < n + l; k++) { mr += R[k]; mg += G[k]; mb += B[k]; }
        mr /= l; mg /= l; mb /= l;
        for (let k = 0; k < l; k++) {
          const r = R[n + k] / mr, g = G[n + k] / mg, b = B[n + k] / mb;
          s1[k] = g - b;            // [0 1 -1]
          s2[k] = -2 * r + g + b;   // [-2 1 1]
        }
        const sd2 = std(s2);
        if (sd2 === 0) continue;
        const ratio = std(s1) / sd2;
        let mh = 0;
        for (let k = 0; k < l; k++) mh += s1[k] + ratio * s2[k];
        mh /= l;
        for (let k = 0; k < l; k++) H[n + k] += s1[k] + ratio * s2[k] - mh;
      }
      return H;
    }
    default:
      throw new Error(`rgbToPulse: unknown method "${method}"`);
  }
}

export function preprocessPulse(p, fs, cfg = DEFAULT_CFG) {
  let pf = detrendSmoothnessPriors(p, cfg.detrendLambda);
  pf = bandpassFilter(pf, fs, cfg);
  const m = mean(pf), sd = std(pf);
  return Float64Array.from(pf, (v) => (sd > 0 ? (v - m) / sd : v - m));
}

// ------------------------------------------------------------------- FFT
function fftInPlace(re, im) { // iterative radix-2, length must be a power of 2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

const hannCache = new Map();
function hann(L) { // MATLAB hann(L): symmetric
  if (!hannCache.has(L)) hannCache.set(L, Float64Array.from({ length: L }, (_, n) => (L === 1 ? 1 : 0.5 * (1 - Math.cos(2 * Math.PI * n / (L - 1))))));
  return hannCache.get(L);
}

// Spectrum of one window: Hann, zero-padded FFT, band power, parabolic peak.
export function windowSpectrum(seg, fs, cfg = DEFAULT_CFG) {
  const L = seg.length;
  const nfft = Math.max(cfg.nfftMin, 2 ** Math.ceil(Math.log2(L)));
  const w = hann(L), m = mean(seg);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let i = 0; i < L; i++) re[i] = (seg[i] - m) * w[i];
  fftInPlace(re, im);
  const half = nfft / 2 + 1;
  const pw = new Float64Array(half);
  for (let k = 0; k < half; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
  const k0 = Math.ceil(cfg.hrBand[0] * nfft / fs - 1e-9), k1 = Math.floor(cfg.hrBand[1] * nfft / fs + 1e-9);
  let kmax = k0;
  for (let k = k0; k <= k1; k++) if (pw[k] > pw[kmax]) kmax = k;
  let delta = 0;
  if (kmax > 0 && kmax < half - 1) {
    const a = pw[kmax - 1], b = pw[kmax], c = pw[kmax + 1], den = a - 2 * b + c;
    if (den !== 0) delta = 0.5 * (a - c) / den;
  }
  const fPeak = (kmax + delta) * fs / nfft;
  const f = Float64Array.from({ length: k1 - k0 + 1 }, (_, i) => (k0 + i) * fs / nfft);
  const P = pw.slice(k0, k1 + 1);
  return { f, P, fPeak, hr: 60 * fPeak, snr: spectralSNR(f, P, fPeak) };
}

// Power within +-0.1 Hz of the peak (and +-0.2 Hz of its 2nd harmonic when in
// band) over the rest of the band, in dB. Quality indicator only.
export function spectralSNR(f, P, f0) {
  let sig = 0, noise = 0;
  for (let i = 0; i < f.length; i++) {
    const near = Math.abs(f[i] - f0) <= 0.1 || Math.abs(f[i] - 2 * f0) <= 0.2;
    if (near) sig += P[i]; else noise += P[i];
  }
  return noise > 0 ? 10 * Math.log10(sig / noise) : Infinity;
}

// Sliding-window HR, same windows as estimateHR.m. tHR relative to sample 1.
export function estimateHR(signal, fs, cfg = DEFAULT_CFG) {
  const N = signal.length;
  const L = Math.round(cfg.hrWindowSec * fs), S = Math.max(1, Math.round(cfg.hrStepSec * fs));
  if (N < L) throw new Error(`estimateHR: signal (${N}) shorter than one window (${L})`);
  const tHR = [], hr = [], snr = [];
  for (let s = 0; s + L <= N; s += S) {
    const r = windowSpectrum(signal.subarray ? signal.subarray(s, s + L) : signal.slice(s, s + L), fs, cfg);
    tHR.push((s + (L - 1) / 2) / fs); hr.push(r.hr); snr.push(r.snr);
  }
  return { tHR, hr, snr };
}

// Offline chain for one RGB trace: pulse -> preprocess -> windowed HR.
export function processTrace(rgb, fs, method, cfg = DEFAULT_CFG) {
  const pf = preprocessPulse(rgbToPulse(rgb, fs, method, cfg), fs, cfg);
  return { pf, ...estimateHR(pf, fs, cfg) };
}

// ------------------------------------------------------------ live stream
// Camera frames arrive at irregular times. Samples are kept with their
// timestamps, resampled onto a uniform fs grid (linear interpolation, gaps
// from lost face bridged like fillGaps in extractRGBTraces.m), then the
// offline chain runs on the last bufferSec and HR is read from the final
// hrWindowSec window.
export class LiveEstimator {
  constructor({ cfg = DEFAULT_CFG, fs = 30, bufferSec = 20, rois = ['forehead', 'leftCheek', 'rightCheek'] } = {}) {
    Object.assign(this, { cfg, fs, bufferSec, rois });
    this.reset();
  }

  reset() { this.t = []; this.samples = []; }

  // tSec: capture time (s). sums: { roi: { s: [R, G, B] sums, n: pixel count } | null }
  push(tSec, sums) {
    if (this.t.length && tSec <= this.t[this.t.length - 1]) return; // duplicate frame
    this.t.push(tSec); this.samples.push(sums);
    const keep = tSec - this.bufferSec - 2;
    let drop = 0;
    while (drop < this.t.length && this.t[drop] < keep) drop++;
    if (drop) { this.t.splice(0, drop); this.samples.splice(0, drop); }
  }

  duration() { return this.t.length > 1 ? this.t[this.t.length - 1] - this.t[0] : 0; }

  // Mean RGB per frame for one ROI ('combined' pools pixels of all base ROIs)
  roiSeries(roi) {
    const names = roi === 'combined' ? this.rois : [roi];
    const t = [], rgb = [];
    for (let i = 0; i < this.t.length; i++) {
      let r = 0, g = 0, b = 0, n = 0;
      for (const name of names) {
        const v = this.samples[i][name];
        if (v && v.n > 0) { r += v.s[0]; g += v.s[1]; b += v.s[2]; n += v.n; }
      }
      if (n > 0) { t.push(this.t[i]); rgb.push([r / n, g / n, b / n]); }
    }
    return { t, rgb };
  }

  // Uniform resampling of the last bufferSec seconds
  uniform(roi) {
    const { t, rgb } = this.roiSeries(roi);
    if (t.length < 2) return null;
    const tEnd = this.t[this.t.length - 1];
    const tStart = Math.max(this.t[0], tEnd - this.bufferSec);
    const n = Math.floor((tEnd - tStart) * this.fs) + 1;
    const out = new Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const tq = tStart + i / this.fs;
      while (j < t.length - 2 && t[j + 1] < tq) j++;
      if (tq <= t[0]) out[i] = rgb[0].slice();
      else if (tq >= t[t.length - 1]) out[i] = rgb[t.length - 1].slice();
      else {
        const w = (tq - t[j]) / (t[j + 1] - t[j]);
        out[i] = [0, 1, 2].map((c) => rgb[j][c] + w * (rgb[j + 1][c] - rgb[j][c]));
      }
    }
    // fraction of the newest window actually covered by valid face samples
    const winStart = tEnd - this.cfg.hrWindowSec;
    const nValid = t.filter((x) => x >= winStart).length;
    const nAll = this.t.filter((x) => x >= winStart).length;
    return { rgb: out, tStart, tEnd, coverage: nAll ? nValid / nAll : 0 };
  }

  // HR from the most recent window for one method x ROI, or null if not ready
  estimate(method, roi) {
    if (this.duration() < this.cfg.hrWindowSec) return null;
    const u = this.uniform(roi);
    const L = Math.round(this.cfg.hrWindowSec * this.fs);
    if (!u || u.rgb.length < L) return null;
    const pf = preprocessPulse(rgbToPulse(u.rgb, this.fs, method, this.cfg), this.fs, this.cfg);
    const seg = pf.slice(pf.length - L);
    const spec = windowSpectrum(seg, this.fs, this.cfg);
    return { method, roi, hr: spec.hr, snr: spec.snr, f: spec.f, P: spec.P, pulse: seg, coverage: u.coverage, t: u.tEnd };
  }
}

// ------------------------------------------------------------- face ROIs
// Mirrors config/getConfig.m (cfg.roi, skin mask) and src/face/roiBoxes.m,
// roiSumRGB.m. Fractions are [x1 x2 y1 y2] of the face box.
export const ROI_CFG = Object.freeze({
  roi: { forehead: [0.30, 0.70, 0.08, 0.25], leftCheek: [0.15, 0.38, 0.50, 0.72], rightCheek: [0.62, 0.85, 0.50, 0.72] },
  useSkinMask: true, skinCb: [77, 127], skinCr: [133, 173], minSkinPixels: 50,
});

// faceBox {x, y, w, h} in pixels -> { roi: {x, y, w, h} } integer rects clipped to the image
export function roiBoxes(box, width, height, roiCfg = ROI_CFG) {
  const out = {};
  for (const [name, r] of Object.entries(roiCfg.roi)) {
    const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
    const x1 = clip(Math.round(box.x + r[0] * box.w), width - 1), x2 = clip(Math.round(box.x + r[1] * box.w), width - 1);
    const y1 = clip(Math.round(box.y + r[2] * box.h), height - 1), y2 = clip(Math.round(box.y + r[3] * box.h), height - 1);
    out[name] = { x: x1, y: y1, w: x2 - x1 + 1, h: y2 - y1 + 1 };
  }
  return out;
}

// Sum of R,G,B over skin pixels (YCbCr thresholds as MATLAB rgb2ycbcr on uint8).
// data: RGBA Uint8ClampedArray (ImageData.data). Falls back to all pixels when
// fewer than minSkinPixels are skin.
export function roiSumRGB(data, roiCfg = ROI_CFG) {
  let sr = 0, sg = 0, sb = 0, n = 0, ar = 0, ag = 0, ab = 0, an = 0;
  const [cb0, cb1] = roiCfg.skinCb, [cr0, cr1] = roiCfg.skinCr;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    ar += r; ag += g; ab += b; an++;
    if (!roiCfg.useSkinMask) continue;
    const cb = Math.round(128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255);
    const cr = Math.round(128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255);
    if (cb >= cb0 && cb <= cb1 && cr >= cr0 && cr <= cr1) { sr += r; sg += g; sb += b; n++; }
  }
  if (roiCfg.useSkinMask && n >= roiCfg.minSkinPixels) return { s: [sr, sg, sb], n, skinFrac: n / Math.max(an, 1) };
  return { s: [ar, ag, ab], n: an, skinFrac: an ? n / an : 0 };
}
