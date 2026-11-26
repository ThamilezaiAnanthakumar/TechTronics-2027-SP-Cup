// pipeline.ts — main-thread capture loop: video -> MediaPipe FaceLandmarker
// (numFaces 4) -> target selection (C3) -> ImageBitmap + landmarks to the
// worker. Timestamps come from requestVideoFrameCallback (captureTime for a
// camera, mediaTime for a file), never frame_index / fps.
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { CONFIG } from './config';
import type { CoreTick, HrRow } from './core/types';
import type { FrameInfo } from './front/frontEnd';
import { landmarksBox, type Box, type Landmarks } from './front/geometry';
import type { FromWorker, ToWorker } from './front/messages';

export type Source = 'live' | 'file';

export interface PipelineEvents {
  onFrame(info: FrameInfo, faces: Box[], target: number, lm: Landmarks | null, procMs: number, fps: number): void;
  onTick(tick: CoreTick): void;
  onRows(rows: HrRow[]): void;
  onLog(line: string): void;
  onEnded(): void;
  onReset(): void;
}

interface CameraLock { ae_locked: boolean; awb_locked: boolean; focus_locked: boolean; fps_fixed: boolean; settings: Record<string, unknown> }

const assetUrl = (p: string) => new URL(p, document.baseURI).href;

export class Pipeline {
  private landmarker: FaceLandmarker | null = null;
  private worker: Worker;
  private video: HTMLVideoElement;
  private stream: MediaStream | null = null;
  private running = false;
  private source: Source = 'live';
  private lowBw = false;
  private method = 'POS';
  private mpTs = 0;
  private inFlight = 0;
  private lastProcT = -Infinity;
  private tStart: number | null = null;
  private targetBox: Box | null = null;
  private targetLost = false;
  private designated = false;
  private clickPoint: [number, number] | null = null;
  private frameTimes: number[] = [];
  private lastFaces: Box[] = [];
  private lastTarget = -1;
  private lastLm: Landmarks | null = null;
  private lockTimer: number | null = null;
  lock: CameraLock = { ae_locked: false, awb_locked: false, focus_locked: false, fps_fixed: false, settings: {} };
  fileName = '';

  constructor(video: HTMLVideoElement, private ev: PipelineEvents) {
    this.video = video;
    this.worker = new Worker(new URL('./front/worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.onWorker(e.data);
  }

  get processingSize(): [number, number] {
    const vw = this.video.videoWidth || 640, vh = this.video.videoHeight || 480;
    const maxW = this.lowBw ? 320 : 640;
    const s = Math.min(1, maxW / vw);
    return [Math.round(vw * s), Math.round(vh * s)];
  }

  private initPromise: Promise<void> | null = null;

  init(): Promise<void> {
    this.initPromise ??= this.load().catch((e) => { this.initPromise = null; throw e; });
    return this.initPromise;
  }

  private async load(): Promise<void> {
    if (this.landmarker) return;
    const fileset = await FilesetResolver.forVisionTasks(assetUrl('mediapipe/wasm'));
    const make = (delegate: 'GPU' | 'CPU') =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: assetUrl('models/face_landmarker.task'), delegate },
        runningMode: 'VIDEO',
        numFaces: 4,
      });
    try {
      this.landmarker = await make('GPU');
      this.ev.onLog('FaceLandmarker ready (GPU delegate, 4 faces)');
    } catch {
      this.landmarker = await make('CPU');
      this.ev.onLog('FaceLandmarker ready (CPU delegate, 4 faces)');
    }
  }

  async startLive(lowBw: boolean, method: string) {
    await this.init();
    this.stop();
    this.source = 'live';
    this.lowBw = lowBw;
    this.method = method;
    const fps = lowBw ? 10 : 30;
    const [w, h] = lowBw ? [320, 240] : [640, 480];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: fps, max: fps } },
    });
    const track = this.stream.getVideoTracks()[0];
    this.video.srcObject = this.stream;
    this.video.muted = true;
    await this.video.play();
    this.lock = { ae_locked: false, awb_locked: false, focus_locked: false, fps_fixed: false, settings: {} };
    const s = track.getSettings();
    this.ev.onLog(`Camera: ${track.label || 'camera'} ${s.width}x${s.height} @ ${s.frameRate ?? '?'} fps`);
    try {
      await track.applyConstraints({ frameRate: { exact: fps } });
      this.lock.fps_fixed = true;
      this.ev.onLog(`Frame rate fixed at ${fps} fps (applyConstraints exact)`);
    } catch (e) {
      this.ev.onLog(`Fixed frame rate not supported: ${(e as Error).name}`);
    }
    // README 11.2.1: lock AE / AWB / focus after a 2 s convergence
    this.lockTimer = window.setTimeout(() => void this.lockCamera(track), CONFIG.camera_lock_after_s * 1000);
    this.begin();
  }

  async startFile(file: File, lowBw: boolean, method: string) {
    await this.init();
    this.stop();
    this.source = 'file';
    this.lowBw = lowBw;
    this.method = method;
    this.fileName = file.name;
    this.video.srcObject = null;
    this.video.src = URL.createObjectURL(file);
    this.video.muted = true;
    this.video.loop = false;
    this.video.onended = () => this.finish();
    await this.video.play();
    this.lock = { ae_locked: false, awb_locked: false, focus_locked: false, fps_fixed: false, settings: { note: 'file input' } };
    this.ev.onLog(`File: ${file.name} ${this.video.videoWidth}x${this.video.videoHeight}`);
    this.begin();
  }

  /**
   * Mean luma (0..255) of the current video frame on an 80x60 thumbnail: the whole
   * frame (scene lighting, independent of skin tone) or only its central half.
   */
  frameLuma(full = false): number {
    const v = this.video;
    if (!v.videoWidth) return NaN;
    const c = document.createElement('canvas');
    c.width = 80; c.height = 60;
    const g = c.getContext('2d', { willReadFrequently: true });
    if (!g) return NaN;
    if (full) g.drawImage(v, 0, 0, 80, 60);
    else g.drawImage(v, v.videoWidth / 4, v.videoHeight / 4, v.videoWidth / 2, v.videoHeight / 2, 0, 0, 80, 60);
    const d = g.getImageData(0, 0, 80, 60).data;
    let s = 0;
    for (let p = 0; p < d.length; p += 4) s += 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    return s / (d.length / 4);
  }

  private async lockCamera(track: MediaStreamTrack) {
    if (!this.running || track.readyState !== 'live') return;
    const caps = (track.getCapabilities?.() ?? {}) as Record<string, unknown>;
    const cur = track.getSettings() as Record<string, unknown>;
    const has = (k: string, v: string) => Array.isArray(caps[k]) && (caps[k] as string[]).includes(v);
    const adv: Record<string, unknown> = {};
    if (has('exposureMode', 'manual')) {
      adv.exposureMode = 'manual';
      if (typeof cur.exposureTime === 'number') adv.exposureTime = cur.exposureTime;
    }
    if (has('whiteBalanceMode', 'manual')) {
      adv.whiteBalanceMode = 'manual';
      if (typeof cur.colorTemperature === 'number') adv.colorTemperature = cur.colorTemperature;
    }
    if (has('focusMode', 'manual')) {
      adv.focusMode = 'manual';
      if (typeof cur.focusDistance === 'number') adv.focusDistance = cur.focusDistance;
    }
    if (!Object.keys(adv).length) {
      this.ev.onLog('Exposure / white-balance lock not supported by this camera or browser (auto stays on)');
      return;
    }
    // Some drivers report a stale/default exposureTime while in auto mode; copying it
    // into manual mode can freeze the camera on a much shorter exposure (dark image).
    // Guard: compare frame brightness before and after the lock, revert if it dropped.
    const yBefore = this.frameLuma();
    this.ev.onLog(`Before lock: exposureTime ${cur.exposureTime ?? '?'}, colorTemperature ${cur.colorTemperature ?? '?'}, frame luma ${yBefore.toFixed(0)}/255`);
    try {
      await track.applyConstraints({ advanced: [adv as MediaTrackConstraintSet] });
    } catch (e) {
      this.ev.onLog(`Camera lock failed: ${(e as Error).message}`);
    }
    await new Promise((r) => window.setTimeout(r, 1500));
    if (!this.running || track.readyState !== 'live') return;
    const yAfter = this.frameLuma();
    const sl = track.getSettings() as Record<string, unknown>;
    this.ev.onLog(`After lock: exposureTime ${sl.exposureTime ?? '?'}, frame luma ${yAfter.toFixed(0)}/255`);
    if (Number.isFinite(yBefore) && Number.isFinite(yAfter) && yAfter < 0.75 * yBefore) {
      const back: Record<string, unknown> = {};
      if (adv.exposureMode) back.exposureMode = 'continuous';
      if (adv.whiteBalanceMode) back.whiteBalanceMode = 'continuous';
      if (adv.focusMode) back.focusMode = 'continuous';
      try {
        await track.applyConstraints({ advanced: [back as MediaTrackConstraintSet] });
      } catch { /* keep going; status below reports what the camera accepted */ }
      this.ev.onLog(`Lock darkened the image (${yBefore.toFixed(0)} -> ${yAfter.toFixed(0)}); reverted to auto exposure / white balance`);
    }
    const s = track.getSettings() as Record<string, unknown>;
    this.lock = {
      ...this.lock,
      ae_locked: s.exposureMode === 'manual',
      awb_locked: s.whiteBalanceMode === 'manual',
      focus_locked: s.focusMode === 'manual',
      settings: {
        exposureMode: s.exposureMode, exposureTime: s.exposureTime, whiteBalanceMode: s.whiteBalanceMode,
        colorTemperature: s.colorTemperature, focusMode: s.focusMode, frameRate: s.frameRate,
      },
    };
    this.ev.onLog(`Lock: AE ${this.lock.ae_locked ? 'locked' : 'not locked'}, AWB ${this.lock.awb_locked ? 'locked' : 'not locked'}, focus ${this.lock.focus_locked ? 'locked' : 'not locked'}`);
  }

  private begin() {
    this.post({ type: 'reset', method: this.method });
    this.ev.onReset();
    Object.assign(this, {
      running: true, inFlight: 0, lastProcT: -Infinity, tStart: null, targetBox: null, targetLost: false,
      designated: false, clickPoint: null, frameTimes: [],
    });
    this.video.requestVideoFrameCallback((n, m) => this.onVideoFrame(n, m));
  }

  /** Designate the target face at a normalised video point (C3). */
  selectAt(x: number, y: number) {
    this.clickPoint = [x, y];
  }

  stop() {
    if (this.lockTimer !== null) clearTimeout(this.lockTimer);
    this.lockTimer = null;
    if (this.running) this.finish();
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.video.src) {
      this.video.pause();
      URL.revokeObjectURL(this.video.src);
      this.video.removeAttribute('src');
    }
    this.video.srcObject = null;
  }

  /** Stop and release the worker (component unmount). */
  dispose() {
    this.stop();
    this.worker.terminate();
    this.landmarker?.close();
    this.landmarker = null;
  }

  private finish() {
    if (!this.running) return;
    this.running = false;
    this.post({ type: 'finish' });
    this.ev.onEnded();
  }

  exportTrace(): Promise<string> {
    const [pw, ph] = this.processingSize;
    const header = {
      version: 1,
      source: this.source === 'live' ? 'web-live' : `web-file:${this.fileName}`,
      resolution: [this.video.videoWidth, this.video.videoHeight],
      proc_resolution: [pw, ph],
      fps_nominal: this.lowBw ? 10 : 30,
      lux: null,
      device: navigator.userAgent,
      ae_locked: this.lock.ae_locked,
      awb_locked: this.lock.awb_locked,
      focus_locked: this.lock.focus_locked,
      fps_fixed: this.lock.fps_fixed,
      camera_settings: this.lock.settings,
      mode: this.lowBw ? 'lowbw' : 'default',
      core: 'interim-ts',
      method: this.method,
      patches: 'shared/patches.json v1 (MediaPipe polygons, Sec 4.2b)',
      created: new Date().toISOString(),
    };
    return new Promise((resolve) => {
      const h = (e: MessageEvent<FromWorker>) => {
        if (e.data.type !== 'export') return;
        this.worker.removeEventListener('message', h);
        resolve(e.data.trace);
      };
      this.worker.addEventListener('message', h);
      this.post({ type: 'export', header });
    });
  }

  private post(m: ToWorker, transfer: Transferable[] = []) {
    this.worker.postMessage(m, transfer);
  }

  private onWorker(m: FromWorker) {
    if (m.type === 'frame') {
      this.inFlight = Math.max(0, this.inFlight - 1);
      const n = this.frameTimes.length;
      const fps = n > 1 ? (n - 1) / (this.frameTimes[n - 1] - this.frameTimes[0]) : 0;
      this.ev.onFrame(m.info, this.lastFaces, this.lastTarget, this.lastLm, m.procMs, fps);
    } else if (m.type === 'tick') {
      this.ev.onTick(m.tick);
      if (m.tick.row) this.ev.onRows([m.tick.row]);
    } else if (m.type === 'rows') {
      this.ev.onRows(m.rows);
    }
  }

  /** Pick the target among detected faces (README 4.1 / 11.2.4). */
  private chooseTarget(boxes: Box[], t: number): { idx: number; status: number } {
    const centre = (b: Box) => [b.x + b.w / 2, b.y + b.h / 2];
    if (this.clickPoint) {
      const [px, py] = this.clickPoint;
      this.clickPoint = null;
      let best = -1, bd = Infinity;
      boxes.forEach((b, i) => {
        const [cx, cy] = centre(b);
        const inside = px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h;
        const dd = Math.hypot(px - cx, py - cy) - (inside ? 10 : 0);
        if (dd < bd) { bd = dd; best = i; }
      });
      if (best >= 0) {
        this.designated = true;
        this.targetBox = boxes[best];
        this.post({ type: 'reset', method: this.method }); // new person: restart skin model and core
        this.ev.onReset();
        this.ev.onLog('Target designated by click');
        return { idx: best, status: 2 };
      }
    }
    if (!boxes.length) {
      if (this.targetBox) this.targetLost = true;
      return { idx: -1, status: 0 };
    }
    const largest = boxes.reduce((bi, b, i) => (b.w * b.h > boxes[bi].w * boxes[bi].h ? i : bi), 0);
    if (this.tStart === null) this.tStart = t;
    if (!this.targetBox || (!this.designated && t - this.tStart < CONFIG.target_first_s)) {
      this.targetBox = boxes[largest];
      return { idx: largest, status: 1 };
    }
    // re-acquire only near the last target position
    const [tx, ty] = centre(this.targetBox);
    let best = -1, bd = Infinity;
    boxes.forEach((b, i) => {
      const [cx, cy] = centre(b);
      const dd = Math.hypot(cx - tx, cy - ty);
      if (dd < bd) { bd = dd; best = i; }
    });
    if (bd > CONFIG.reacquire_radius_frac * Math.max(this.targetBox.w, this.targetBox.h)) {
      this.targetLost = true;
      return { idx: -1, status: 0 };
    }
    this.targetBox = boxes[best];
    const status = this.targetLost ? 2 : 1;
    this.targetLost = false;
    return { idx: best, status };
  }

  private onVideoFrame(now: number, meta: VideoFrameCallbackMetadata) {
    if (!this.running) return;
    this.video.requestVideoFrameCallback((n, m) => this.onVideoFrame(n, m));
    const tS = this.source === 'file'
      ? meta.mediaTime
      : (meta.captureTime ?? meta.expectedDisplayTime ?? now) / 1000;
    const minDt = this.lowBw ? 1 / 10 - 0.01 : 0;
    if (tS <= this.lastProcT || tS - this.lastProcT < minDt) return;
    if (this.inFlight >= 2) return; // worker busy: drop this frame (timestamps keep the timing honest)
    this.lastProcT = tS;
    this.frameTimes.push(tS);
    while (this.frameTimes.length > 2 && this.frameTimes[0] < tS - 2) this.frameTimes.shift();

    let ts = this.source === 'file' ? Math.round(meta.mediaTime * 1000) : Math.round(performance.now());
    if (ts <= this.mpTs) ts = this.mpTs + 1;
    this.mpTs = ts;
    const res = this.landmarker!.detectForVideo(this.video, ts);
    const all = (res.faceLandmarks ?? []).map((pts) => {
      const a = new Float32Array(pts.length * 2);
      pts.forEach((p, i) => { a[2 * i] = p.x; a[2 * i + 1] = p.y; });
      return a;
    });
    const boxes = all.map(landmarksBox);
    const { idx, status } = this.chooseTarget(boxes, tS);
    const lm = idx >= 0 ? all[idx] : null;
    this.lastFaces = boxes;
    this.lastTarget = idx;
    this.lastLm = lm;

    const [pw, ph] = this.processingSize;
    this.inFlight++;
    createImageBitmap(this.video, { resizeWidth: pw, resizeHeight: ph, resizeQuality: 'medium' })
      .then((bitmap) => this.post({ type: 'frame', bitmap, lm, t_s: tS, faceStatus: status }, [bitmap]))
      .catch(() => { this.inFlight = Math.max(0, this.inFlight - 1); });
  }
}
