// app.js — live webcam front end for rppg.js.
// Camera -> MediaPipe Face Landmarker (face box) -> ROI rectangles as in
// src/face/roiBoxes.m -> skin-masked RGB sums -> LiveEstimator -> UI.
import { FilesetResolver, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { DEFAULT_CFG, ROI_CFG, LiveEstimator, roiBoxes, roiSumRGB } from './rppg.js';

const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const BASE_ROIS = Object.keys(ROI_CFG.roi);
const BOX_SMOOTH = 0.4;      // EMA weight of the new face box (landmark jitter)
const STABLE_N = 5;          // median over the last N estimates
const MIN_COVERAGE = 0.8;    // fraction of the window with a face
const HIST_SEC = 120;

const $ = (id) => document.getElementById(id);
const ui = {
  video: $('video'), overlay: $('overlay'), placeholder: $('placeholder'), status: $('status'),
  start: $('startBtn'), lock: $('lockBtn'), method: $('methodSel'), roi: $('roiSel'), camInfo: $('camInfo'),
  hrStable: $('hrStable'), hrRaw: $('hrRaw'), quality: $('quality'), coverage: $('coverage'), fps: $('fps'),
  elapsed: $('elapsed'), warmup: $('warmup'), warmupBar: $('warmupBar'),
  pulse: $('pulseChart'), spec: $('specChart'), specLabel: $('specLabel'), hist: $('histChart'),
  refForm: $('refForm'), refInput: $('refInput'), refBtn: $('refBtn'), refStats: $('refStats'),
  refBody: document.querySelector('#refTable tbody'), combo: $('comboTable'), export: $('exportBtn'),
};

const state = {
  running: false, stream: null, track: null, landmarker: null, estimator: null,
  grab: null, grabCtx: null, box: null, lastTs: -1, t0: null, frameTimes: [],
  timer: null, session: null, latest: {}, stableHist: [], hist: [], locked: false,
};

function setStatus(text, kind = '') { ui.status.textContent = text; ui.status.className = `pill ${kind}`; }

// ------------------------------------------------------------- startup
async function loadLandmarker() {
  if (state.landmarker) return state.landmarker;
  setStatus('Loading face model...');
  const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: MP_MODEL, delegate }, runningMode: 'VIDEO', numFaces: 1 });
  try { state.landmarker = await FaceLandmarker.createFromOptions(fileset, opts('GPU')); }
  catch { state.landmarker = await FaceLandmarker.createFromOptions(fileset, opts('CPU')); }
  return state.landmarker;
}

async function start() {
  ui.start.disabled = true;
  try {
    await loadLandmarker();
    setStatus('Opening camera...');
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false,
    });
    state.track = state.stream.getVideoTracks()[0];
    ui.video.srcObject = state.stream;
    await ui.video.play();
    const s = state.track.getSettings();
    ui.camInfo.textContent = `${state.track.label || 'Camera'} · ${s.width}x${s.height} @ ${s.frameRate ?? '?'} fps requested`;

    const W = ui.video.videoWidth, H = ui.video.videoHeight;
    state.grab = new OffscreenCanvas(W, H);
    state.grabCtx = state.grab.getContext('2d', { willReadFrequently: true });
    ui.overlay.width = W; ui.overlay.height = H;

    state.estimator = new LiveEstimator({ cfg: DEFAULT_CFG, fs: 30, bufferSec: 20, rois: BASE_ROIS });
    Object.assign(state, { box: null, lastTs: -1, t0: null, frameTimes: [], latest: {}, stableHist: [], hist: [], locked: false });
    state.session = { frames: [], estimates: [], refs: [] };
    ui.refBody.innerHTML = ''; ui.refStats.textContent = 'No references logged.';
    state.running = true;
    ui.placeholder.hidden = true;
    ui.start.textContent = 'Stop'; ui.start.classList.remove('primary');
    ui.start.disabled = false; ui.lock.disabled = false; ui.refBtn.disabled = false; ui.export.disabled = false;
    ui.lock.textContent = 'Lock camera';
    ui.warmup.hidden = false;
    setStatus('Looking for a face...', 'warn');
    scheduleFrame();
    state.timer = setInterval(estimateTick, DEFAULT_CFG.hrStepSec * 1000);
  } catch (err) {
    console.error(err);
    setStatus(err.name === 'NotAllowedError' ? 'Camera permission denied' : `Error: ${err.message}`, 'err');
    ui.start.disabled = false;
  }
}

function stop() {
  state.running = false;
  clearInterval(state.timer);
  state.stream?.getTracks().forEach((t) => t.stop());
  ui.video.srcObject = null;
  ui.placeholder.hidden = false;
  ui.start.textContent = 'Start'; ui.start.classList.add('primary');
  ui.lock.disabled = true; ui.refBtn.disabled = true; ui.warmup.hidden = true;
  setStatus('Stopped');
}

// ---------------------------------------------------------- per frame
function scheduleFrame() {
  if (!state.running) return;
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) ui.video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame((now) => onFrame(now, null));
}

function onFrame(now, meta) {
  if (!state.running) return;
  // Capture time when the browser reports it, else presentation time.
  const tMs = meta?.captureTime ?? meta?.expectedDisplayTime ?? now;
  if (tMs <= state.lastTs) { scheduleFrame(); return; }
  state.lastTs = tMs;
  if (state.t0 === null) state.t0 = tMs;
  const t = (tMs - state.t0) / 1000;
  state.frameTimes.push(t);
  while (state.frameTimes.length && state.frameTimes[0] < t - 2) state.frameTimes.shift();

  const W = state.grab.width, H = state.grab.height;
  state.grabCtx.drawImage(ui.video, 0, 0, W, H);

  let ts = Math.round(performance.now());
  if (ts <= (state.mpTs ?? 0)) ts = state.mpTs + 1;
  state.mpTs = ts;
  const res = state.landmarker.detectForVideo(ui.video, ts);
  const lm = res.faceLandmarks?.[0];

  let rects = null, sums = null;
  if (lm) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of lm) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
    const b = { x: x0 * W, y: y0 * H, w: (x1 - x0) * W, h: (y1 - y0) * H };
    state.box = state.box ? Object.fromEntries(Object.keys(b).map((k) => [k, state.box[k] + BOX_SMOOTH * (b[k] - state.box[k])])) : b;
    rects = roiBoxes(state.box, W, H);
    sums = {};
    for (const name of BASE_ROIS) {
      const r = rects[name];
      sums[name] = r.w > 1 && r.h > 1 ? roiSumRGB(state.grabCtx.getImageData(r.x, r.y, r.w, r.h).data) : null;
    }
  }
  state.estimator.push(t, sums ?? {});
  recordFrame(t, sums);
  drawOverlay(rects, !!lm);
  scheduleFrame();
}

function drawOverlay(rects, faceOK) {
  const ctx = ui.overlay.getContext('2d');
  ctx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  if (!rects) return;
  const b = state.box;
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.strokeStyle = faceOK ? '#3fb950' : '#d29922';
  for (const r of Object.values(rects)) ctx.strokeRect(r.x, r.y, r.w, r.h);
}

// ---------------------------------------------------------- estimation
function estimateTick() {
  const est = state.estimator;
  const dur = est.duration();
  const nF = state.frameTimes.length;
  const fps = nF > 1 ? (nF - 1) / (state.frameTimes[nF - 1] - state.frameTimes[0]) : 0;
  ui.fps.textContent = fps ? fps.toFixed(1) : '--';
  ui.fps.className = fps && fps < 20 ? 'q poor' : '';
  ui.elapsed.textContent = `${dur.toFixed(0)} s`;
  ui.warmupBar.style.width = `${Math.min(100, (100 * dur) / DEFAULT_CFG.hrWindowSec)}%`;
  if (dur >= DEFAULT_CFG.hrWindowSec) ui.warmup.hidden = true;

  const t = state.estimator.t.at(-1) ?? 0;
  const results = {};
  for (const method of DEFAULT_CFG.methods) {
    for (const roi of DEFAULT_CFG.roiSet) {
      let r = null;
      try { r = est.estimate(method, roi); } catch (e) { console.warn(method, roi, e); }
      results[`${method}/${roi}`] = r;
      if (r) state.session.estimates.push([t, method, roi, r.hr, r.snr, r.coverage]);
    }
  }
  state.latest = results;
  renderCombos(results);

  const sel = results[`${ui.method.value}/${ui.roi.value}`];
  if (!sel) {
    const faceSeen = state.box !== null;
    setStatus(faceSeen ? `Warming up (${Math.max(0, DEFAULT_CFG.hrWindowSec - dur).toFixed(0)} s)` : 'Looking for a face...', 'warn');
    return;
  }
  const good = sel.coverage >= MIN_COVERAGE;
  if (good) state.stableHist.push(sel.hr);
  if (state.stableHist.length > STABLE_N) state.stableHist.shift();
  const stable = state.stableHist.length ? median(state.stableHist) : NaN;

  ui.hrStable.textContent = Number.isFinite(stable) ? stable.toFixed(0) : '--';
  ui.hrRaw.textContent = `${sel.hr.toFixed(1)} bpm`;
  ui.coverage.textContent = `${(100 * sel.coverage).toFixed(0)} %`;
  ui.coverage.className = good ? '' : 'q poor';
  const q = sel.snr > 3 ? ['good', 'Good'] : sel.snr > 0 ? ['fair', 'Fair'] : ['poor', 'Poor'];
  ui.quality.textContent = `${q[1]} (${sel.snr.toFixed(1)} dB)`;
  ui.quality.className = `q ${q[0]}`;
  setStatus(good ? 'Measuring' : 'Face lost — hold still', good ? 'ok' : 'warn');

  state.hist.push({ t, hr: sel.hr, stable });
  while (state.hist.length && state.hist[0].t < t - HIST_SEC) state.hist.shift();

  drawLine(ui.pulse, sel.pulse, { color: css('--stable') });
  ui.specLabel.textContent = `peak ${sel.hr.toFixed(1)} bpm`;
  drawSpectrum(ui.spec, sel.f, sel.P, sel.hr / 60);
  drawHistory();
}

const median = (a) => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

function renderCombos(results) {
  const { methods, roiSet } = DEFAULT_CFG;
  const head = `<thead><tr><th></th>${roiSet.map((r) => `<th>${r}</th>`).join('')}</tr></thead>`;
  const rows = methods.map((m) => `<tr><td>${m}</td>${roiSet.map((r) => {
    const v = results[`${m}/${r}`];
    const cls = [m === ui.method.value && r === ui.roi.value ? 'sel' : '', v && v.snr <= 0 ? 'low' : ''].join(' ');
    return `<td class="${cls}">${v ? `${v.hr.toFixed(0)} <span class="muted">/ ${v.snr.toFixed(0)}</span>` : '--'}</td>`;
  }).join('')}</tr>`).join('');
  ui.combo.innerHTML = head + `<tbody>${rows}</tbody>`;
}

// -------------------------------------------------------------- charts
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function prep(canvas) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '11px system-ui, sans-serif';
  return { ctx, w, h };
}

function drawLine(canvas, y, { color }) {
  const { ctx, w, h } = prep(canvas);
  let lo = Infinity, hi = -Infinity;
  for (const v of y) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = 6, sy = (v) => h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
  y.forEach((v, i) => { const x = (i / (y.length - 1)) * w; i ? ctx.lineTo(x, sy(v)) : ctx.moveTo(x, sy(v)); });
  ctx.stroke();
}

function drawSpectrum(canvas, f, P, fPeak) {
  const { ctx, w, h } = prep(canvas);
  const padB = 18, padL = 4, bpm0 = f[0] * 60, bpm1 = f.at(-1) * 60;
  const sx = (bpm) => padL + ((bpm - bpm0) / (bpm1 - bpm0)) * (w - padL - 4);
  let mx = 0; for (const v of P) mx = Math.max(mx, v);
  ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted'); ctx.lineWidth = 1;
  for (let b = 60; b <= 180; b += 30) { const x = sx(b); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - padB); ctx.stroke(); ctx.fillText(`${b}`, x - 8, h - 4); }
  ctx.strokeStyle = css('--est'); ctx.lineWidth = 1.4; ctx.beginPath();
  for (let i = 0; i < f.length; i++) { const x = sx(f[i] * 60), y = (h - padB) * (1 - P[i] / (mx || 1)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.stroke();
  ctx.strokeStyle = css('--stable'); ctx.setLineDash([4, 3]); ctx.beginPath();
  ctx.moveTo(sx(fPeak * 60), 0); ctx.lineTo(sx(fPeak * 60), h - padB); ctx.stroke(); ctx.setLineDash([]);
}

function drawHistory() {
  const { ctx, w, h } = prep(ui.hist);
  const pts = state.hist; if (!pts.length) return;
  const tEnd = pts.at(-1).t, tStart = Math.max(0, tEnd - HIST_SEC);
  const refs = state.session.refs.filter((r) => r.t >= tStart);
  const vals = [...pts.map((p) => p.hr), ...refs.map((r) => r.ref)];
  let lo = Math.floor((Math.min(...vals) - 5) / 10) * 10, hi = Math.ceil((Math.max(...vals) + 5) / 10) * 10;
  lo = Math.max(40, lo); hi = Math.min(190, Math.max(hi, lo + 20));
  const padL = 30, padB = 18;
  const sx = (t) => padL + ((t - tStart) / Math.max(1, tEnd - tStart)) * (w - padL - 6);
  const sy = (v) => (h - padB) * (1 - (v - lo) / (hi - lo));
  ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted'); ctx.lineWidth = 1;
  for (let v = lo; v <= hi; v += 10) { ctx.beginPath(); ctx.moveTo(padL, sy(v)); ctx.lineTo(w, sy(v)); ctx.stroke(); ctx.fillText(`${v}`, 2, sy(v) + 4); }
  ctx.fillText(`${tStart.toFixed(0)} s`, padL, h - 4); ctx.fillText(`${tEnd.toFixed(0)} s`, w - 34, h - 4);
  const line = (key, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); let on = false;
    for (const p of pts) { if (!Number.isFinite(p[key])) { on = false; continue; } const x = sx(p.t), y = sy(p[key]); on ? ctx.lineTo(x, y) : ctx.moveTo(x, y); on = true; }
    ctx.stroke();
  };
  line('hr', css('--est'), 1.2);
  line('stable', css('--stable'), 2.2);
  ctx.fillStyle = css('--ref');
  for (const r of refs) { ctx.beginPath(); ctx.arc(sx(r.t), sy(r.ref), 4.5, 0, 2 * Math.PI); ctx.fill(); }
}

// ----------------------------------------------------- camera locking
async function toggleLock() {
  const track = state.track; if (!track) return;
  const caps = track.getCapabilities?.() ?? {};
  const set = track.getSettings();
  const want = state.locked ? 'continuous' : 'manual';
  const adv = [], done = [];
  if (caps.exposureMode?.includes(want)) { adv.push({ exposureMode: want, ...(want === 'manual' && set.exposureTime ? { exposureTime: set.exposureTime } : {}) }); done.push('exposure'); }
  if (caps.whiteBalanceMode?.includes(want)) { adv.push({ whiteBalanceMode: want, ...(want === 'manual' && set.colorTemperature ? { colorTemperature: set.colorTemperature } : {}) }); done.push('white balance'); }
  if (!adv.length) { ui.camInfo.textContent = 'This camera/browser does not allow locking exposure or white balance. Use steady lighting.'; return; }
  try {
    await track.applyConstraints({ advanced: adv });
    state.locked = !state.locked;
    ui.lock.textContent = state.locked ? 'Unlock camera' : 'Lock camera';
    ui.camInfo.textContent = `${state.locked ? 'Locked' : 'Auto'}: ${done.join(' + ')}.`;
  } catch (e) { ui.camInfo.textContent = `Lock failed: ${e.message}`; }
}

// ---------------------------------------------------- references, export
function logReference(ev) {
  ev.preventDefault();
  const ref = Number(ui.refInput.value);
  if (!Number.isFinite(ref) || !state.running) return;
  const t = state.estimator.t.at(-1) ?? 0;
  const est = state.stableHist.length ? median(state.stableHist) : NaN;
  const entry = { t, ref, est, method: ui.method.value, roi: ui.roi.value };
  state.session.refs.push(entry);
  ui.refInput.value = '';
  const err = est - ref;
  ui.refBody.insertAdjacentHTML('afterbegin', `<tr><td>${t.toFixed(0)}</td><td>${ref}</td><td>${Number.isFinite(est) ? est.toFixed(1) : '--'}</td><td>${Number.isFinite(err) ? (err > 0 ? '+' : '') + err.toFixed(1) : '--'}</td></tr>`);
  const errs = state.session.refs.map((r) => r.est - r.ref).filter(Number.isFinite);
  if (errs.length) {
    const mae = errs.reduce((s, e) => s + Math.abs(e), 0) / errs.length;
    const within5 = (100 * errs.filter((e) => Math.abs(e) <= 5).length) / errs.length;
    ui.refStats.textContent = `${errs.length} reference(s) · MAE ${mae.toFixed(1)} bpm · ${within5.toFixed(0)}% within ±5 bpm`;
  }
  drawHistory();
}

function recordFrame(t, sums) {
  const row = [t, sums ? 1 : 0];
  for (const name of BASE_ROIS) {
    const v = sums?.[name];
    row.push(...(v && v.n ? [v.s[0] / v.n, v.s[1] / v.n, v.s[2] / v.n, v.n] : ['', '', '', 0]));
  }
  state.session.frames.push(row);
}

function download(name, header, rows) {
  const fmt = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(5)) : v);
  const text = [header.join(','), ...rows.map((r) => r.map(fmt).join(','))].join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function exportSession() {
  const s = state.session; if (!s) return;
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const hdr = ['t_s', 'faceOK', ...BASE_ROIS.flatMap((r) => [`${r}_R`, `${r}_G`, `${r}_B`, `${r}_n`])];
  download(`facevitals_${stamp}_traces.csv`, hdr, s.frames);
  download(`facevitals_${stamp}_estimates.csv`, ['t_s', 'method', 'roi', 'hr_bpm', 'snr_db', 'coverage'], s.estimates);
  download(`facevitals_${stamp}_refs.csv`, ['t_s', 'hr_ref', 'hr_est', 'method', 'roi'], s.refs.map((r) => [r.t, r.ref, r.est, r.method, r.roi]));
}

// ---------------------------------------------------------------- wiring
ui.start.addEventListener('click', () => (state.running ? stop() : start()));
ui.lock.addEventListener('click', toggleLock);
ui.refForm.addEventListener('submit', logReference);
ui.export.addEventListener('click', exportSession);
for (const el of [ui.method, ui.roi]) el.addEventListener('change', () => { state.stableHist = []; renderCombos(state.latest); });
if (!navigator.mediaDevices?.getUserMedia) setStatus('Camera API unavailable (use localhost or https)', 'err');
