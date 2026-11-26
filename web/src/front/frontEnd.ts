// frontEnd.ts — per-frame pixel selection (README 4.3, 11.2.5) producing
// TraceFrames (README 11.1). Runs in the Web Worker on ImageData.
//
// mask = skin AND NOT specular AND NOT texture, then trimmed mean 10/10.
//  - skin: adaptive 2-D Gaussian in (Cb, Cr), fitted on glabella + malar
//    pixels during the first skin_fit_s seconds, refit by EMA (30 s), gated
//    by a widened YCbCr sanity box. Before the fit: sanity box only.
//  - specular: Y > 0.95 * 255 or any channel >= saturation_level.
//  - texture: gradient magnitude of the Gaussian-smoothed luma (sigma 1 px);
//    the top 15 % of candidate pixels per patch are excluded.
//  - occlusion: EMA of the patch skin ratio with hysteresis (on 0.40 / off 0.50).
// Not yet in the MVP: L* pigmentation sub-clusters, temporal-variance mask,
// 2x2 sub-patches.
import { CONFIG, PATCHES, SKIN_MODEL_CODES, type FvConfig } from '../config';
import { LOCAL_CONFIG, type LocalConfig } from '../localConfig';
import type { TraceFrame, TracePatch } from '../core/types';
import { ita, srgbToLab } from './color';
import { eyePose, patchPolygons, rasterise, type Landmarks } from './geometry';

const SKIN_IDS = new Set(PATCHES.filter((p) => SKIN_MODEL_CODES.includes(p.code)).map((p) => p.id));

export interface FrameInfo {
  t_s: number;
  face: boolean;
  valid: boolean[];
  occluded: boolean[];
  npix: number[];
  skinRatio: number[];
  luma: number;          // mean luma over all patch pixels (0..255)
  itaDeg: number;        // smoothed ITA of glabella + malar (NaN until measured)
  skinModelFitted: boolean;
}

interface SkinStats { n: number; sb: number; sr: number; sbb: number; srr: number; sbr: number }
const emptyStats = (): SkinStats => ({ n: 0, sb: 0, sr: 0, sbb: 0, srr: 0, sbr: 0 });

class SkinModel {
  fitted = false;
  mean = [0, 0];
  cov = [1, 0, 1]; // [bb, br, rr]
  private inv = [1, 0, 1];
  private acc = emptyStats();

  constructor(private cfg: FvConfig) {}

  get invCov() { return this.inv; }

  accumulate(s: SkinStats) {
    const a = this.acc;
    a.n += s.n; a.sb += s.sb; a.sr += s.sr; a.sbb += s.sbb; a.srr += s.srr; a.sbr += s.sbr;
  }

  private moments(s: SkinStats) {
    const mb = s.sb / s.n, mr = s.sr / s.n;
    return { mean: [mb, mr], cov: [s.sbb / s.n - mb * mb + 1, s.sbr / s.n - mb * mr, s.srr / s.n - mr * mr + 1] }; // +1: regularise
  }

  private setCov(cov: number[]) {
    this.cov = cov;
    const det = cov[0] * cov[2] - cov[1] * cov[1];
    this.inv = det > 1e-9 ? [cov[2] / det, -cov[1] / det, cov[0] / det] : [1, 0, 1];
  }

  fit(): boolean {
    if (this.acc.n < 200) return false;
    const m = this.moments(this.acc);
    this.mean = m.mean;
    this.setCov(m.cov);
    this.fitted = true;
    return true;
  }

  /** Slow refit (EMA with time constant skin_refit_tau_s). */
  update(s: SkinStats, dt: number) {
    if (!this.fitted || s.n < 50) return;
    const a = Math.min(1, dt / this.cfg.skin_refit_tau_s);
    const m = this.moments(s);
    this.mean = [0, 1].map((i) => (1 - a) * this.mean[i] + a * m.mean[i]);
    this.setCov(this.cov.map((v, i) => (1 - a) * v + a * m.cov[i]));
  }
}

export class FrontEnd {
  private skin: SkinModel;
  private tFirstFace: number | null = null;
  private lastT: number | null = null;
  private skinEma: number[] = [];
  private occluded: boolean[] = [];
  private prevPose: ReturnType<typeof eyePose> | null = null;
  private itaEma = NaN;

  constructor(private cfg: FvConfig = CONFIG, private local: LocalConfig = LOCAL_CONFIG) {
    this.skin = new SkinModel(cfg);
    this.reset();
  }

  reset() {
    this.skin = new SkinModel(this.cfg);
    this.tFirstFace = null;
    this.lastT = null;
    this.skinEma = PATCHES.map(() => 1);
    this.occluded = PATCHES.map(() => false);
    this.prevPose = null;
    this.itaEma = NaN;
  }

  /**
   * img: RGBA crop of the frame with origin (ox, oy); W x H: full processing
   * frame size; lm: target landmarks (normalised) or null.
   */
  process(
    img: ImageData | null, ox: number, oy: number, W: number, H: number,
    lm: Landmarks | null, t_s: number, faceStatus: number,
  ): { frame: TraceFrame; info: FrameInfo } {
    const cfg = this.cfg;
    const dt = this.lastT === null ? 1 / 30 : Math.max(1e-3, t_s - this.lastT);
    this.lastT = t_s;
    const nP = PATCHES.length;
    const info: FrameInfo = {
      t_s, face: !!lm && !!img, valid: new Array(nP).fill(false), occluded: [...this.occluded],
      npix: new Array(nP).fill(0), skinRatio: new Array(nP).fill(0), luma: NaN, itaDeg: this.itaEma,
      skinModelFitted: this.skin.fitted,
    };
    if (!lm || !img) {
      this.prevPose = null;
      return { frame: { t_s, face_status: 0, track_quality: 0, motion: [0, 0, 0, 0], n_patches: 0, patches: [] }, info };
    }
    if (this.tFirstFace === null) this.tFirstFace = t_s;

    // motion vs previous frame (README 11.1)
    const pose = eyePose(lm, W, H);
    const pp = this.prevPose;
    const motion: [number, number, number, number] = pp && pp.iod > 0
      ? [(pose.cx - pp.cx) / pp.iod, (pose.cy - pp.cy) / pp.iod, pose.iod / pp.iod - 1, pose.roll - pp.roll]
      : [0, 0, 0, 0];
    this.prevPose = pose;

    const cw = img.width, chh = img.height, d = img.data;
    // smoothed luma and its gradient magnitude over the crop
    const Y = new Float32Array(cw * chh);
    for (let i = 0, p = 0; i < Y.length; i++, p += 4) Y[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    const grad = gradientMagnitude(Y, cw, chh);

    const polys = patchPolygons(lm, W, H);
    const npixMin = cfg.npix_min_qvga * ((W * H) / (320 * 240));
    const fitStats = emptyStats();
    let lumaSum = 0, lumaN = 0;
    let itaR = 0, itaG = 0, itaB = 0, itaN = 0;
    const patches: TracePatch[] = [];

    const specY = cfg.specular_y_frac * 255, sat = cfg.saturation_level;
    const [cb0, cb1] = cfg.skin_sanity_cb, [cr0, cr1] = cfg.skin_sanity_cr;
    const fitted = this.skin.fitted, [mb, mr] = this.skin.mean, [i0, i1, i2] = this.skin.invCov;
    const tau2 = cfg.skin_mahalanobis * cfg.skin_mahalanobis;
    for (let pi = 0; pi < nP; pi++) {
      const def = PATCHES[pi];
      let idx = rasterise(polys[pi], ox, oy, cw, chh);
      if (this.local.patch_erode_px > 0) idx = erode(idx, cw, chh, this.local.patch_erode_px);
      const total = idx.length;
      const candBuf = new Int32Array(total);
      let nc = 0;
      const collect = SKIN_IDS.has(def.id);
      for (let k = 0; k < total; k++) {
        const i = idx[k], p = 4 * i;
        const r = d[p], g = d[p + 1], b = d[p + 2];
        lumaSum += Y[i]; lumaN++;
        if (Y[i] > specY || r >= sat || g >= sat || b >= sat) continue;
        const cb = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255; // = ycbcr(r, g, b)[1]
        const cr = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255;  // = ycbcr(r, g, b)[2]
        if (cb < cb0 || cb > cb1 || cr < cr0 || cr > cr1) continue; // widened sanity box
        if (fitted) {
          const db = cb - mb, dr = cr - mr;
          if (i0 * db * db + 2 * i1 * db * dr + i2 * dr * dr >= tau2) continue; // Mahalanobis
        }
        if (collect) addStats(fitStats, cb, cr);
        candBuf[nc++] = i;
      }
      const cand = candBuf.subarray(0, nc);
      // texture exclusion: drop the top texture_top_frac by gradient
      let sel: Int32Array = cand;
      if (cand.length >= 10) {
        const gs = new Float32Array(cand.length);
        for (let k = 0; k < cand.length; k++) gs[k] = grad[cand[k]];
        gs.sort();
        const thr = gs[Math.max(0, Math.ceil((1 - cfg.texture_top_frac) * gs.length) - 1)];
        sel = cand.filter((i) => grad[i] <= thr);
      }
      const rgb = trimmedMeanRgb(d, sel, cfg.trimmed_mean_frac);
      const ratio = total > 0 ? sel.length / total : 0;
      const a = Math.min(1, dt / cfg.occlusion_ema_s);
      this.skinEma[pi] = (1 - a) * this.skinEma[pi] + a * ratio;
      if (this.occluded[pi] && this.skinEma[pi] > cfg.occlusion_off) this.occluded[pi] = false;
      else if (!this.occluded[pi] && this.skinEma[pi] < cfg.occlusion_on) this.occluded[pi] = true;
      const valid = !this.occluded[pi] && sel.length >= npixMin && Number.isFinite(rgb[0]);
      if (collect && valid) {
        itaR += rgb[0] * sel.length; itaG += rgb[1] * sel.length; itaB += rgb[2] * sel.length; itaN += sel.length;
      }
      patches.push({
        id: def.id, group: def.groupIndex, cluster: 0,
        rgb_mean: valid ? rgb : [NaN, NaN, NaN], npix: sel.length, valid: valid ? 1 : 0, skin_ratio: ratio,
      });
      info.valid[pi] = valid;
      info.npix[pi] = sel.length;
      info.skinRatio[pi] = ratio;
    }
    info.occluded = [...this.occluded];

    // adaptive skin model: fit after skin_fit_s, then slow EMA refit
    if (!this.skin.fitted) {
      this.skin.accumulate(fitStats);
      if (t_s - this.tFirstFace >= cfg.skin_fit_s) this.skin.fit();
    } else {
      this.skin.update(fitStats, dt);
    }
    info.skinModelFitted = this.skin.fitted;
    info.luma = lumaN ? lumaSum / lumaN : NaN;
    if (itaN > 0) {
      const lab = srgbToLab(itaR / itaN, itaG / itaN, itaB / itaN);
      const v = ita(lab[0], lab[2]);
      const a = Math.min(1, dt / 2);
      this.itaEma = Number.isFinite(this.itaEma) ? (1 - a) * this.itaEma + a * v : v;
    }
    info.itaDeg = this.itaEma;

    return {
      frame: { t_s, face_status: faceStatus, track_quality: 1, motion, n_patches: patches.length, patches },
      info,
    };
  }
}

/** Morphological erosion (4-neighbour, r passes) of a pixel set given as crop indices. */
export function erode(idx: Int32Array, w: number, h: number, r: number): Int32Array {
  const m = new Uint8Array(w * h);
  for (const i of idx) m[i] = 1;
  let cur = idx;
  for (let pass = 0; pass < r; pass++) {
    const keep: number[] = [];
    for (const i of cur) {
      const x = i % w, y = (i - x) / w;
      if (x > 0 && x < w - 1 && y > 0 && y < h - 1 && m[i - 1] && m[i + 1] && m[i - w] && m[i + w]) keep.push(i);
    }
    const next = Int32Array.from(keep);
    for (const i of cur) m[i] = 0;
    for (const i of next) m[i] = 1;
    cur = next;
  }
  return cur;
}

function addStats(s: SkinStats, cb: number, cr: number) {
  s.n++; s.sb += cb; s.sr += cr; s.sbb += cb * cb; s.srr += cr * cr; s.sbr += cb * cr;
}

/** Gradient magnitude of the luma after a 5-tap binomial blur (sigma ~ 1 px). */
export function gradientMagnitude(Y: Float32Array, w: number, h: number): Float32Array {
  const tmp = new Float32Array(w * h), bl = new Float32Array(w * h), g = new Float32Array(w * h);
  const cx = (x: number) => (x < 0 ? 0 : x >= w ? w - 1 : x);
  const cy = (y: number) => (y < 0 ? 0 : y >= h ? h - 1 : y);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      tmp[o + x] = (Y[o + cx(x - 2)] + 4 * Y[o + cx(x - 1)] + 6 * Y[o + x] + 4 * Y[o + cx(x + 1)] + Y[o + cx(x + 2)]) / 16;
    }
  }
  for (let y = 0; y < h; y++) {
    const a = cy(y - 2) * w, b = cy(y - 1) * w, c = y * w, d = cy(y + 1) * w, e = cy(y + 2) * w;
    for (let x = 0; x < w; x++) bl[c + x] = (tmp[a + x] + 4 * tmp[b + x] + 6 * tmp[c + x] + 4 * tmp[d + x] + tmp[e + x]) / 16;
  }
  for (let y = 0; y < h; y++) {
    const o = y * w, up = cy(y - 1) * w, dn = cy(y + 1) * w;
    for (let x = 0; x < w; x++) {
      const gx = (bl[o + cx(x + 1)] - bl[o + cx(x - 1)]) / 2;
      const gy = (bl[dn + x] - bl[up + x]) / 2;
      g[o + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return g;
}

/** Per-channel trimmed mean (drop the lowest and highest frac) via 256-bin histograms. */
export function trimmedMeanRgb(d: Uint8ClampedArray, sel: ArrayLike<number>, frac: number): [number, number, number] {
  const n = sel.length;
  if (!n) return [NaN, NaN, NaN];
  const out: [number, number, number] = [0, 0, 0];
  const cut = Math.floor(frac * n);
  const keep = n - 2 * cut;
  for (let c = 0; c < 3; c++) {
    const hist = new Uint32Array(256);
    for (let k = 0; k < n; k++) hist[d[4 * sel[k] + c]]++;
    let lo = cut, hi = cut, sum = 0;
    const counts = Array.from(hist);
    for (let v = 0; v < 256 && lo > 0; v++) { const t = Math.min(lo, counts[v]); counts[v] -= t; lo -= t; }
    for (let v = 255; v >= 0 && hi > 0; v--) { const t = Math.min(hi, counts[v]); counts[v] -= t; hi -= t; }
    for (let v = 0; v < 256; v++) sum += v * counts[v];
    out[c] = sum / keep;
  }
  return out;
}
