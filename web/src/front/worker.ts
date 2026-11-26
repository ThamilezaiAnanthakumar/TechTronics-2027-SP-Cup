/// <reference lib="webworker" />
// worker.ts — pixel selection (FrontEnd) + signal core, off the main thread.
// Receives ImageBitmaps + target landmarks; emits per-frame info and 1 Hz
// core ticks. Keeps the TraceFrame log for .tracejsonl export.
import { CONFIG } from '../config';
import { InterimCore } from '../core/interimCore';
import type { HrRow, SignalCore, TraceFrame } from '../core/types';
import { FrontEnd } from './frontEnd';
import { patchPolygons } from './geometry';
import type { ToWorker } from './messages';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let canvas: OffscreenCanvas | null = null;
let g: OffscreenCanvasRenderingContext2D | null = null;
let front = new FrontEnd();
let core: SignalCore = new InterimCore();
let trace: TraceFrame[] = [];
let rows: HrRow[] = [];

function makeCore(method: string): SignalCore {
  // fvcore-WASM will replace this line (README Section 16, interim rule).
  return new InterimCore({ ...CONFIG, method });
}

ctx.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  if (m.type === 'reset') {
    front = new FrontEnd();
    core = makeCore(m.method);
    trace = [];
    rows = [];
    return;
  }
  if (m.type === 'finish') {
    const tail = core.finish();
    rows.push(...tail);
    ctx.postMessage({ type: 'rows', rows: tail });
    return;
  }
  if (m.type === 'export') {
    const lines = [JSON.stringify(m.header), ...trace.map((f) => JSON.stringify(f))];
    ctx.postMessage({ type: 'export', trace: lines.join('\n') + '\n', nFrames: trace.length });
    return;
  }
  if (m.type === 'frame') {
    const t0 = performance.now();
    const { bitmap, lm, t_s, faceStatus } = m;
    const W = bitmap.width, H = bitmap.height;
    let img: ImageData | null = null, ox = 0, oy = 0;
    if (lm) {
      if (!canvas || canvas.width !== W || canvas.height !== H) {
        canvas = new OffscreenCanvas(W, H);
        g = canvas.getContext('2d', { willReadFrequently: true });
      }
      g!.drawImage(bitmap, 0, 0);
      // crop = union of the patch polygons (+3 px margin for the gradient)
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const poly of patchPolygons(lm, W, H)) {
        for (let i = 0; i < poly.length; i += 2) {
          x0 = Math.min(x0, poly[i]); x1 = Math.max(x1, poly[i]);
          y0 = Math.min(y0, poly[i + 1]); y1 = Math.max(y1, poly[i + 1]);
        }
      }
      ox = Math.max(0, Math.floor(x0) - 3); oy = Math.max(0, Math.floor(y0) - 3);
      const cw = Math.min(W, Math.ceil(x1) + 3) - ox, ch = Math.min(H, Math.ceil(y1) + 3) - oy;
      if (cw > 4 && ch > 4) img = g!.getImageData(ox, oy, cw, ch);
    }
    bitmap.close();
    const { frame, info } = front.process(img, ox, oy, W, H, img ? lm : null, t_s, img ? faceStatus : 0);
    trace.push(frame);
    const ticks = core.push(frame);
    for (const tk of ticks) if (tk.row) rows.push(tk.row);
    ctx.postMessage({ type: 'frame', info, procMs: performance.now() - t0 });
    for (const tick of ticks) ctx.postMessage({ type: 'tick', tick });
  }
};

export {}; // module worker
