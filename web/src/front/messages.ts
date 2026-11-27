import type { CoreTick, HrRow } from '../core/types';

/** Signal core selection: CP3 = frozen Round-1 chain (default), interim = README 4.x interim core. */
export type CoreKind = 'cp3' | 'interim';
import type { FrameInfo } from './frontEnd';

export type ToWorker =
  | { type: 'reset'; method: string; core: CoreKind }
  | { type: 'frame'; bitmap: ImageBitmap; lm: Float32Array | null; t_s: number; faceStatus: number }
  | { type: 'finish' }
  | { type: 'export'; header: Record<string, unknown> };

export type FromWorker =
  | { type: 'frame'; info: FrameInfo; procMs: number; diag?: { bitmapLuma: number; cropLuma: number; size: string; crop: string } }
  | { type: 'tick'; tick: CoreTick }
  | { type: 'rows'; rows: HrRow[] }
  | { type: 'export'; trace: string; nFrames: number };
