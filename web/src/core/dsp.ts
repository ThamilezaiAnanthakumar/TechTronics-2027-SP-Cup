// dsp.ts — TypeScript port of deploy/web/rppg.js (itself a port of
// src/signal/*.m): Butterworth band-pass + filtfilt, smoothness-priors
// detrend, POS / CHROM, FFT. tests/seedParity.test.ts checks this file
// against rppg.js sample by sample. MATLAB golden-vector parity is pending
// (shared/golden does not exist yet).

export const mean = (a: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
};

/** Sample std (N-1), as MATLAB std. */
export function std(a: ArrayLike<number>): number {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    s += d * d;
  }
  return a.length > 1 ? Math.sqrt(s / (a.length - 1)) : 0;
}

export function median(a: ArrayLike<number>): number {
  const s = Float64Array.from(a).sort();
  const n = s.length;
  if (!n) return NaN;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

export function zscore(x: Float64Array): Float64Array {
  const m = mean(x), sd = std(x);
  return Float64Array.from(x, (v) => (sd > 0 ? (v - m) / sd : v - m));
}

// ------------------------------------------------------ complex helpers
type Cx = [number, number];
const C = {
  add: (a: Cx, b: Cx): Cx => [a[0] + b[0], a[1] + b[1]],
  sub: (a: Cx, b: Cx): Cx => [a[0] - b[0], a[1] - b[1]],
  mul: (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
  div: (a: Cx, b: Cx): Cx => {
    const d = b[0] * b[0] + b[1] * b[1];
    return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
  },
  scale: (a: Cx, k: number): Cx => [a[0] * k, a[1] * k],
  sqrt: (a: Cx): Cx => {
    const r = Math.hypot(a[0], a[1]);
    const re = Math.sqrt((r + a[0]) / 2);
    const im = Math.sign(a[1] || 1) * Math.sqrt(Math.max(0, (r - a[0]) / 2));
    return [re, im];
  },
};

// --------------------------------------------------- Butterworth design
export interface Sos {
  sos: { r: number; b: [number, number, number]; a: [number, number, number] }[];
  gain: number;
}

const filterCache = new Map<string, Sos>();

/** MATLAB butter(order, band/(fs/2), 'bandpass') -> zp2sos. */
export function butterBandpass(order: number, band: readonly [number, number], fs: number): Sos {
  const key = `${order}|${band[0]}|${band[1]}|${fs}`;
  const hit = filterCache.get(key);
  if (hit) return hit;
  const fs2 = 4;
  const u1 = fs2 * Math.tan((Math.PI * (band[0] / (fs / 2))) / 2);
  const u2 = fs2 * Math.tan((Math.PI * (band[1] / (fs / 2))) / 2);
  const Bw = u2 - u1, Wo2 = u1 * u2;

  const polesA: Cx[] = [];
  for (let k = 1; k <= order; k++) {
    const th = (Math.PI * (2 * k + order - 1)) / (2 * order);
    const pb = C.scale([Math.cos(th), Math.sin(th)], Bw / 2);
    const disc = C.sqrt(C.sub(C.mul(pb, pb), [Wo2, 0]));
    polesA.push(C.add(pb, disc), C.sub(pb, disc));
  }
  let den: Cx = [1, 0];
  const polesD = polesA.map((s) => {
    den = C.mul(den, C.sub([fs2, 0], s));
    return C.div(C.add([fs2, 0], s), C.sub([fs2, 0], s));
  });
  const gain = C.div([Math.pow(Bw * fs2, order), 0], den)[0];

  const upper = polesD.filter((p) => p[1] > 1e-12).sort((p, q) => Math.hypot(...q) - Math.hypot(...p));
  if (upper.length !== order) throw new Error('butterBandpass: unexpected real poles');
  const zeros = [...Array(order).fill(1), ...Array(order).fill(-1)] as number[];
  const sos = upper
    .map((p) => {
      const pick = () => {
        let best = 0;
        zeros.forEach((z, i) => {
          if (Math.hypot(p[0] - z, p[1]) < Math.hypot(p[0] - zeros[best], p[1])) best = i;
        });
        return zeros.splice(best, 1)[0];
      };
      const z1 = pick(), z2 = pick();
      return {
        r: Math.hypot(...p),
        b: [1, -(z1 + z2), z1 * z2] as [number, number, number],
        a: [1, -2 * p[0], p[0] * p[0] + p[1] * p[1]] as [number, number, number],
      };
    })
    .sort((s, t) => s.r - t.r);
  const out = { sos, gain };
  filterCache.set(key, out);
  return out;
}

function biquad(sec: Sos['sos'][number], x: Float64Array, zi0: number, zi1: number): Float64Array {
  const [b0, b1, b2] = sec.b, a1 = sec.a[1], a2 = sec.a[2];
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

/** Zero-phase filtering as MATLAB filtfilt(sos, g, x). */
export function filtfilt({ sos, gain }: Sos, x: ArrayLike<number>): Float64Array {
  const nfact = 6 * sos.length;
  let y = Float64Array.from(x);
  if (y.length <= nfact) throw new Error(`filtfilt: signal too short (${y.length})`);
  for (const sec of sos) {
    const [b0, b1, b2] = sec.b, a1 = sec.a[1], a2 = sec.a[2];
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
  const g2 = gain * gain;
  for (let i = 0; i < y.length; i++) y[i] *= g2;
  return y;
}

export function bandpass(x: ArrayLike<number>, fs: number, band: readonly [number, number], order: number) {
  return filtfilt(butterBandpass(order, band, fs), x);
}

// ---------------------------------------- smoothness-priors detrending
/** y = x - (I + lambda^2 D2'D2) \ x, banded LDL' (O(N)). */
export function detrendSmoothnessPriors(x: ArrayLike<number>, lambda: number): Float64Array {
  const N = x.length;
  if (N < 3) {
    const m = mean(x);
    return Float64Array.from(x, (v) => v - m);
  }
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

// ------------------------------------------------------------ POS / CHROM
export type Rgb = [number, number, number];

/** POS (Wang 2017), window l samples, overlap-add. */
export function pos(rgb: Rgb[], l: number): Float64Array {
  const N = rgb.length;
  const H = new Float64Array(N);
  const s1 = new Float64Array(l), s2 = new Float64Array(l);
  for (let n = 0; n + l <= N; n++) {
    let mr = 0, mg = 0, mb = 0;
    for (let k = n; k < n + l; k++) { mr += rgb[k][0]; mg += rgb[k][1]; mb += rgb[k][2]; }
    mr /= l; mg /= l; mb /= l;
    for (let k = 0; k < l; k++) {
      const r = rgb[n + k][0] / mr, g = rgb[n + k][1] / mg, b = rgb[n + k][2] / mb;
      s1[k] = g - b;
      s2[k] = -2 * r + g + b;
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

/** CHROM (de Haan 2013), whole-segment version as in rgbToPulse.m. */
export function chrom(rgb: Rgb[], fs: number, band: readonly [number, number], order: number): Float64Array {
  const N = rgb.length;
  const mr = mean(rgb.map((v) => v[0])), mg = mean(rgb.map((v) => v[1])), mb = mean(rgb.map((v) => v[2]));
  const X = new Float64Array(N), Y = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    const r = rgb[n][0] / mr, g = rgb[n][1] / mg, b = rgb[n][2] / mb;
    X[n] = 3 * r - 2 * g;
    Y[n] = 1.5 * r + g - 1.5 * b;
  }
  const Xf = bandpass(X, fs, band, order), Yf = bandpass(Y, fs, band, order);
  const alpha = std(Xf) / std(Yf);
  return Float64Array.from(Xf, (v, i) => v - alpha * Yf[i]);
}

// -------------------------------------------------------------------- FFT
export function fftInPlace(re: Float64Array, im: Float64Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
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

const hannCache = new Map<number, Float64Array>();
/** MATLAB hann(L), symmetric. */
export function hann(L: number): Float64Array {
  let w = hannCache.get(L);
  if (!w) {
    w = Float64Array.from({ length: L }, (_, n) => (L === 1 ? 1 : 0.5 * (1 - Math.cos((2 * Math.PI * n) / (L - 1)))));
    hannCache.set(L, w);
  }
  return w;
}

/** One-sided power spectrum of a Hann-windowed, zero-padded segment. df = fs/nfft. */
export function powerSpectrum(seg: ArrayLike<number>, nfftMin: number): { P: Float64Array; nfft: number } {
  const L = seg.length;
  const nfft = Math.max(nfftMin, 2 ** Math.ceil(Math.log2(L)));
  const w = hann(L), m = mean(seg);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let i = 0; i < L; i++) re[i] = (seg[i] - m) * w[i];
  fftInPlace(re, im);
  const half = nfft / 2 + 1;
  const P = new Float64Array(half);
  for (let k = 0; k < half; k++) P[k] = re[k] * re[k] + im[k] * im[k];
  return { P, nfft };
}

/** Parabolic interpolation of a peak at index k; returns the fractional offset. */
export function parabolicOffset(y: ArrayLike<number>, k: number): number {
  if (k <= 0 || k >= y.length - 1) return 0;
  const a = y[k - 1], b = y[k], c = y[k + 1], den = a - 2 * b + c;
  return den !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / den)) : 0;
}
