// Writes a synthetic-face .y4m clip for Chrome's fake camera (no real faces).
import { writeFileSync } from 'node:fs';
import { gaussian, mulberry32 } from '../tests/synth';
import { renderCrop } from '../tests/faceRender';

export function writeSyntheticY4m(path: string, seconds = 20, fps = 15, W = 320, H = 240) {
  const g = gaussian(mulberry32(5));
  const parts: Buffer[] = [Buffer.from(`YUV4MPEG2 W${W} H${H} F${fps}:1 Ip A1:1 C420jpeg\n`)];
  for (let i = 0; i < seconds * fps; i++) {
    const t = i / fps;
    const pulse = 0.004 / 0.77 * Math.sin(2 * Math.PI * 1.2 * t);
    const img = renderCrop({ cx: W / 2, cy: H * 0.42, iod: 0.17 * W, roll: 0 }, 0, 0, W, H, pulse, g, 2);
    const d = img.data;
    const Y = Buffer.alloc(W * H), U = Buffer.alloc((W / 2) * (H / 2)), V = Buffer.alloc((W / 2) * (H / 2));
    for (let p = 0; p < W * H; p++) Y[p] = Math.round(0.299 * d[4 * p] + 0.587 * d[4 * p + 1] + 0.114 * d[4 * p + 2]);
    for (let y = 0; y < H / 2; y++) for (let x = 0; x < W / 2; x++) {
      let r = 0, gg = 0, b = 0;
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
        const p = 4 * ((2 * y + dy) * W + 2 * x + dx);
        r += d[p] / 4; gg += d[p + 1] / 4; b += d[p + 2] / 4;
      }
      const k = y * (W / 2) + x;
      U[k] = Math.max(0, Math.min(255, Math.round(128 - 0.168736 * r - 0.331264 * gg + 0.5 * b)));
      V[k] = Math.max(0, Math.min(255, Math.round(128 + 0.5 * r - 0.418688 * gg - 0.081312 * b)));
    }
    parts.push(Buffer.from('FRAME\n'), Y, U, V);
  }
  writeFileSync(path, Buffer.concat(parts));
}
