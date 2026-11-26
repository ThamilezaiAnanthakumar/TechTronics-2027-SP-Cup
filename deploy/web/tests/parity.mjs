// Parity test: JS port (rppg.js) vs MATLAB on the traces exported by
// tools/exportParityData.m (outputs/web_parity/*.json).
//   node deploy/web/tests/parity.mjs [dir]
// 1. Offline: same RGB trace, same fps -> per-window HR must match MATLAB.
// 2. Live replay: frames streamed through LiveEstimator (as from a camera),
//    HR of the newest window compared with the contact-PPG reference.
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_CFG, processTrace, LiveEstimator } from '../rppg.js';

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(process.argv[2] ?? join(here, '..', '..', '..', 'outputs', 'web_parity'));
const files = readdirSync(dir).filter((f) => f.endsWith('.json')).sort();
if (!files.length) { console.error(`No JSON in ${dir}. Run tools/exportParityData.m first.`); process.exit(2); }

const { methods, roiSet } = DEFAULT_CFG;
const combos = methods.flatMap((m) => roiSet.map((r) => `${m}/${r}`));
const acc = Object.fromEntries(combos.map((c) => [c, { maxDiff: 0, sumDiff: 0, n: 0, minCorr: 1, errM: [], errJ: [], errLive: [] }]));
const corr = (a, b) => {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
};
const arr = (v) => (Array.isArray(v) ? v.flat() : [v]);

const t0 = performance.now();
for (const file of files) {
  const J = JSON.parse(readFileSync(join(dir, file), 'utf8'));
  const fs = J.fps, ref = J.hrRef ? arr(J.hrRef) : null, tHRm = arr(J.tHR);
  for (const roi of roiSet) {
    const rgb = J.rgb[roi];
    for (const method of methods) {
      const a = acc[`${method}/${roi}`];
      const hrM = arr(J.hr[method][roi]);
      const js = processTrace(rgb, fs, method, DEFAULT_CFG);
      if (js.hr.length !== hrM.length) throw new Error(`${file} ${method}/${roi}: ${js.hr.length} vs ${hrM.length} windows`);
      for (let i = 0; i < hrM.length; i++) {
        const d = Math.abs(js.hr[i] - hrM[i]);
        a.maxDiff = Math.max(a.maxDiff, d); a.sumDiff += d; a.n++;
        if (ref) { a.errM.push(hrM[i] - ref[i]); a.errJ.push(js.hr[i] - ref[i]); }
      }
      a.minCorr = Math.min(a.minCorr, corr(js.pf, arr(J.pf[method][roi])));
    }
  }

  // Live replay: frames at their nominal times, estimate once per second
  if (ref) {
    const base = ['forehead', 'leftCheek', 'rightCheek'];
    const est = new LiveEstimator({ fs: 30, bufferSec: 20 });
    const N = J.rgb.combined.length;
    let nextEst = DEFAULT_CFG.hrWindowSec;
    for (let i = 0; i < N; i++) {
      const t = i / fs, sums = {};
      // per-ROI means with unit pixel count (pixel counts are not exported);
      // 'combined' therefore pools ROIs equally here.
      for (const r of base) sums[r] = { s: J.rgb[r][i], n: 1 };
      est.push(t, sums);
      if (t + 1e-9 >= nextEst) {
        nextEst += 1;
        const centre = t - (DEFAULT_CFG.hrWindowSec - 1 / fs) / 2;
        let k = 0;
        for (let j = 1; j < tHRm.length; j++) if (Math.abs(tHRm[j] - centre) < Math.abs(tHRm[k] - centre)) k = j;
        for (const roi of roiSet) for (const method of methods) {
          const e = est.estimate(method, roi);
          if (e) acc[`${method}/${roi}`].errLive.push(e.hr - ref[k]);
        }
      }
    }
  }
}

const mae = (e) => (e.length ? e.reduce((s, v) => s + Math.abs(v), 0) / e.length : NaN);
const f = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '  -  ').padStart(7);
console.log(`\n${files.length} subjects from ${dir} (${((performance.now() - t0) / 1000).toFixed(1)} s)\n`);
console.log('Method/ROI            |JS-MATLAB| HR (bpm)   min corr   MAE vs contact-PPG reference (bpm)');
console.log('                         max     mean       pf         MATLAB   JS offline   JS live');
let worst = 0;
for (const c of combos) {
  const a = acc[c];
  worst = Math.max(worst, a.maxDiff);
  console.log(`${c.padEnd(20)} ${f(a.maxDiff, 3)} ${f(a.sumDiff / a.n, 4)}   ${f(a.minCorr, 5)}    ${f(mae(a.errM))}  ${f(mae(a.errJ))}     ${f(mae(a.errLive))}`);
}
const TOL = 0.5; // bpm; filter edge handling may differ slightly from MATLAB
console.log(`\nWorst per-window HR difference JS vs MATLAB: ${worst.toFixed(3)} bpm (tolerance ${TOL})`);
process.exit(worst <= TOL ? 0 : 1);
