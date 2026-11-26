import { useCallback, useEffect, useRef, useState } from 'react';
import { CONFIG } from '../config';
import type { CoreTick, HrRow } from '../core/types';
import { itaCategory } from '../front/color';
import type { FrameInfo } from '../front/frontEnd';
import { Pipeline, type Source } from '../pipeline';
import { STATUS_COLOR, drawBvp, drawOverlay, drawSpectrogram, type SpecColumn } from './draw';

/** Provisional C1 threshold on whole-frame mean luma (0..255); to move into shared/config.default.json. */
const LOW_LIGHT_FRAME_LUMA = 40;
/** Display-only hold of the last reliable HR through short quality dips (s). */
const HOLD_S = 10;

const SPEC_SPAN_S = 60;
const STATUS_TEXT: Record<string, string> = {
  OK: 'Measuring',
  WARMUP: 'Warming up',
  NO_FACE: 'No face',
  MASKED: 'Mask detected — using upper face',
  LOW_SNR: 'Weak signal — hold still',
  LOW_COVERAGE: 'Face partly lost',
};

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const bvpRef = useRef<HTMLCanvasElement>(null);
  const specRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pipe = useRef<Pipeline | null>(null);
  const lastTick = useRef<CoreTick | null>(null);
  const lastInfo = useRef<FrameInfo | null>(null);
  const cols = useRef<SpecColumn[]>([]);
  const rowsRef = useRef<HrRow[]>([]);
  const t0 = useRef<number | null>(null);
  const uiThrottle = useRef(0);

  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<Source>('live');
  const [lowBw, setLowBw] = useState(false);
  const [method, setMethod] = useState('POS');
  const [rows, setRows] = useState<HrRow[]>([]);
  const [tick, setTick] = useState<CoreTick | null>(null);
  const [info, setInfo] = useState<FrameInfo | null>(null);
  const [stats, setStats] = useState({ fps: 0, procMs: 0, faces: 0 });
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [frameY, setFrameY] = useState(NaN);
  const lastGood = useRef<{ hr: number; t: number } | null>(null);
  const confEma = useRef(0);
  const confRow = useRef<HrRow | null>(null);
  // whole-frame brightness, sampled once per second while running (C1 warning)
  useEffect(() => {
    if (!running) { setFrameY(NaN); return; }
    const id = window.setInterval(() => setFrameY(pipe.current?.frameLuma(true) ?? NaN), 1000);
    return () => window.clearInterval(id);
  }, [running]);

  const log = useCallback((line: string) => {
    const ts = new Date().toLocaleTimeString();
    setLogs((l) => [...l.slice(-40), `${ts}  ${line}`]);
  }, []);

  const redrawCharts = useCallback(() => {
    if (specRef.current) {
      drawSpectrogram(specRef.current, cols.current, rowsRef.current, SPEC_SPAN_S, [CONFIG.hr_grid_bpm[0], CONFIG.hr_grid_bpm[1]]);
    }
    const tk = lastTick.current;
    if (bvpRef.current) drawBvp(bvpRef.current, tk?.bvp ?? null, STATUS_COLOR[tk?.status ?? 'WARMUP']);
  }, []);

  useEffect(() => {
    const v = videoRef.current!;
    pipe.current = new Pipeline(v, {
      onFrame(fi, faces, target, lm, procMs, fps) {
        lastInfo.current = fi;
        if (t0.current === null) t0.current = fi.t_s;
        if (overlayRef.current && v.videoWidth) {
          drawOverlay(overlayRef.current, v.videoWidth, v.videoHeight, faces, target, lm, fi, lastTick.current);
        }
        const now = performance.now();
        if (now - uiThrottle.current > 250) {
          uiThrottle.current = now;
          setInfo(fi);
          setStats({ fps, procMs, faces: faces.length });
        }
      },
      onTick(tk) {
        lastTick.current = tk;
        cols.current.push({ t: tk.t_s, emission: tk.emission, track: tk.trackBpm, status: tk.status });
        while (cols.current.length && cols.current[0].t < tk.t_s - SPEC_SPAN_S - 2) cols.current.shift();
        setTick(tk);
        redrawCharts();
      },
      onRows(r) {
        rowsRef.current = [...rowsRef.current, ...r];
        setRows(rowsRef.current);
        redrawCharts();
      },
      onLog: log,
      onEnded() {
        setRunning(false);
        log('Input ended');
      },
      onReset() {
        lastGood.current = null; confEma.current = 0; confRow.current = null;
        cols.current = [];
        rowsRef.current = [];
        lastTick.current = null;
        t0.current = null;
        setRows([]);
        setTick(null);
        redrawCharts();
      },
    });
    // load MediaPipe WASM + model now, so nothing is fetched after page load
    pipe.current.init().catch((e: Error) => {
      setError(`Could not load the face model (${e.message}) — run "npm run fetch-models" first.`);
    });
    const onResize = () => redrawCharts();
    window.addEventListener('resize', onResize);
    redrawCharts();
    return () => {
      window.removeEventListener('resize', onResize);
      pipe.current?.dispose();
    };
  }, [log, redrawCharts]);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      setRunning(true);
    } catch (e) {
      const err = e as Error;
      const msg = err.name === 'NotAllowedError' ? 'Camera permission denied.' : err.message || String(e);
      setError(msg.includes('face_landmarker') || msg.includes('wasm') ? `${msg} — run "npm run fetch-models" first.` : msg);
      log(`Error: ${msg}`);
    } finally {
      setBusy(false);
    }
  };

  const startLive = () => {
    setSource('live');
    void run(() => pipe.current!.startLive(lowBw, method));
  };
  const startFile = (f: File) => {
    setSource('file');
    void run(() => pipe.current!.startFile(f, lowBw, method));
  };
  const stop = () => {
    pipe.current?.stop();
    setRunning(false);
  };

  const onStageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!running) return;
    const r = e.currentTarget.getBoundingClientRect();
    let x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    if (source === 'live') x = 1 - x; // the live view is mirrored
    pipe.current?.selectAt(x, y);
  };

  const exportCsv = () => {
    const off = source === 'live' ? t0.current ?? 0 : 0;
    const lines = ['time_s,hr_bpm,confidence,status'];
    for (const r of rowsRef.current) {
      lines.push(`${(r.time_s - off).toFixed(3)},${Number.isFinite(r.hr_bpm) ? r.hr_bpm.toFixed(2) : 'NaN'},${r.confidence.toFixed(3)},${r.status}`);
    }
    download('processed_output.csv', lines.join('\n') + '\n', 'text/csv');
  };
  const exportTrace = async () => {
    const text = await pipe.current!.exportTrace();
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    download(`facevitals_${stamp}.tracejsonl`, text, 'application/x-ndjson');
  };

  // headline numbers
  const last = rows.length ? rows[rows.length - 1] : null;
  const shownStatus = tick && (tick.status === 'NO_FACE' || tick.status === 'LOW_COVERAGE') ? tick.status : last?.status ?? tick?.status ?? null;
  const hrOk = last && (last.status === 'OK' || last.status === 'MASKED') && Number.isFinite(last.hr_bpm) && shownStatus === last.status;
  // Display only (CSV keeps NaN): hold the last reliable HR for HOLD_S seconds through
  // short LOW_SNR / LOW_COVERAGE dips, greyed and labelled; never through NO_FACE.
  if (hrOk) lastGood.current = { hr: last!.hr_bpm, t: last!.time_s };
  const heldAge = !hrOk && lastGood.current && last && shownStatus !== 'NO_FACE' ? last.time_s - lastGood.current.t : Infinity;
  const held = heldAge <= HOLD_S;
  const hrShown = hrOk ? last!.hr_bpm : held ? lastGood.current!.hr : NaN;
  // confidence bar: exponential smoothing (~3 s) of the per-second confidence, display only
  if (last && last !== confRow.current) {
    confRow.current = last;
    confEma.current = confEma.current + 0.3 * ((hrOk ? last.confidence : 0) - confEma.current);
  }
  const conf = running ? confEma.current : 0;
  // C1 warning from whole-frame brightness (scene light), not skin brightness:
  // skin luma also depends on skin tone, so a skin threshold would flag darker
  // skin as "low light" (C2 bias). LOW_LIGHT_FRAME_LUMA is a provisional web-only
  // value until it is moved into shared/config.default.json.
  const lowLight = running && Number.isFinite(frameY) && frameY < LOW_LIGHT_FRAME_LUMA;
  const itaVal = info?.itaDeg ?? NaN;
  const nOcc = info ? info.occluded.filter(Boolean).length : 0;
  const warmLeft = tick ? Math.max(0, CONFIG.window_s + CONFIG.viterbi_lag_s - (tick.t_s - (t0.current ?? tick.t_s))) : null;

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg viewBox="0 0 64 64" width="28" height="28" aria-hidden="true">
            <path d="M8 34h12l5-12 8 24 6-16 4 4h13" fill="none" stroke="currentColor" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>
            <h1>FaceVitals</h1>
            <p>Camera heart rate · TechTronics · IEEE SP Cup 2027</p>
          </div>
        </div>
        <div className="privacy" role="note">
          <span className="dot" /> On-device only. Video never leaves this device; no network requests after loading.
        </div>
      </header>

      <main className="grid">
        <section className="card video-card">
          <div className={`stage ${source === 'live' ? 'mirror' : ''}`} onClick={onStageClick}>
            <video ref={videoRef} playsInline muted />
            <canvas ref={overlayRef} className="overlay" />
            {!running && (
              <div className="placeholder">
                <p>Start the camera or open a video file.</p>
                <p className="small">Face the camera in even light, about 50 cm away, and keep still.</p>
              </div>
            )}
          </div>
          {running && stats.faces > 1 && <div className="hint">{stats.faces} faces — click a face to make it the target.</div>}
          {lowLight && (
            <div className="warn" role="alert">
              Low light: frame brightness {frameY.toFixed(0)}/255. Add light in front of the face for a reliable reading.
            </div>
          )}
          <div className="controls">
            {!running ? (
              <>
                <button className="primary" disabled={busy} onClick={startLive}>{busy ? 'Loading…' : 'Start camera'}</button>
                <button disabled={busy} onClick={() => fileRef.current?.click()}>Open video file</button>
                <input ref={fileRef} type="file" accept="video/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) startFile(f); e.target.value = ''; }} />
              </>
            ) : (
              <button className="danger" onClick={stop}>Stop</button>
            )}
            <label className="toggle" title="320x240 at 10 fps (C4/C5)">
              <input type="checkbox" checked={lowBw} disabled={running} onChange={(e) => setLowBw(e.target.checked)} /> Low bandwidth (320×240, 10 fps)
            </label>
            <label className="select">
              Method
              <select value={method} disabled={running} onChange={(e) => setMethod(e.target.value)}>
                <option value="POS">POS</option>
                <option value="CHROM">CHROM</option>
              </select>
            </label>
          </div>
          {error && <div className="error" role="alert">{error}</div>}
        </section>

        <section className="card hr-card">
          <div className="hr">
            <span className="value" style={held ? { opacity: 0.45 } : undefined}>{Number.isFinite(hrShown) ? hrShown.toFixed(0) : '--'}</span>
            <span className="unit">bpm{held ? ` · held, ${heldAge.toFixed(0)} s ago` : ''}</span>
          </div>
          {shownStatus && (
            <div className="badge" style={{ background: STATUS_COLOR[shownStatus] }}>
              {shownStatus}
              <span>{STATUS_TEXT[shownStatus]}{shownStatus === 'WARMUP' && warmLeft !== null ? ` · ${warmLeft.toFixed(0)} s` : ''}</span>
            </div>
          )}
          <div className="conf">
            <div className="conf-label"><span>Confidence</span><span>{running && (hrOk || held) ? `${(100 * conf).toFixed(0)} %` : '--'}</span></div>
            <div className="bar"><div style={{ width: `${100 * conf}%`, background: STATUS_COLOR[shownStatus ?? 'WARMUP'] }} /></div>
          </div>
          <dl className="facts">
            <dt>Skin tone (ITA)</dt>
            <dd>{Number.isFinite(itaVal) ? `${itaVal.toFixed(1)}° · ${itaCategory(itaVal)}` : '--'}</dd>
            <dt>Frame brightness</dt>
            <dd className={lowLight ? 'bad' : ''}>{running && Number.isFinite(frameY) ? `${frameY.toFixed(0)} / 255` : '--'}</dd>
            <dt>Face skin brightness</dt>
            <dd>{info && Number.isFinite(info.luma) ? `${info.luma.toFixed(0)} / 255` : '--'}</dd>
            <dt>Patches used</dt>
            <dd>{last && hrOk ? `${last.n_patches_used} of 10` : '--'}{nOcc ? ` · ${nOcc} occluded` : ''}</dd>
            <dt>Skin model</dt>
            <dd>{info ? (info.skinModelFitted ? 'adapted to this face' : 'learning (first 3 s)') : '--'}</dd>
            <dt>Frame rate</dt>
            <dd>{stats.fps ? `${stats.fps.toFixed(1)} fps · ${stats.procMs.toFixed(1)} ms/frame` : '--'}</dd>
          </dl>
          <div className="exports">
            <button disabled={!rows.length} onClick={exportCsv}>processed_output.csv</button>
            <button disabled={!rows.length && !running} onClick={() => void exportTrace()}>.tracejsonl</button>
          </div>
        </section>

        <section className="card chart-card">
          <h2>Pulse wave (fused BVP, last {CONFIG.window_s} s)</h2>
          <canvas ref={bvpRef} className="chart bvp" />
        </section>

        <section className="card chart-card">
          <h2>Fused spectrum over time <span className="legend"><i className="lg-hr" /> HR track (Viterbi, {CONFIG.viterbi_lag_s} s lag) <i className="lg-now" /> live state</span></h2>
          <canvas ref={specRef} className="chart spec" />
        </section>

        <section className="card log-card">
          <h2>Session log</h2>
          <pre>{logs.length ? logs.join('\n') : 'Nothing yet.'}</pre>
          <p className="note">
            Interim TypeScript signal core (POS/CHROM, Kalman, top-{CONFIG.top_k} spectral fusion, harmonic Viterbi). Parity with the MATLAB
            reference is pending golden vectors. Not a medical device.
          </p>
        </section>
      </main>
    </div>
  );
}
