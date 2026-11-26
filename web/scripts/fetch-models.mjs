// Copies the MediaPipe Tasks Vision WASM runtime from node_modules into
// public/mediapipe/wasm and downloads the Face Landmarker model into
// public/models, so the app (and its service worker) never needs a CDN.
// Run once after `npm ci`: npm run fetch-models
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const wasmSrc = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmDst = join(root, 'public', 'mediapipe', 'wasm');
mkdirSync(wasmDst, { recursive: true });
for (const f of readdirSync(wasmSrc)) {
  copyFileSync(join(wasmSrc, f), join(wasmDst, f));
  console.log(`wasm   ${f} (${(statSync(join(wasmDst, f)).size / 1e6).toFixed(1)} MB)`);
}

const modelDst = join(root, 'public', 'models', 'face_landmarker.task');
mkdirSync(dirname(modelDst), { recursive: true });
if (existsSync(modelDst) && statSync(modelDst).size > 1e6) {
  console.log(`model  already present (${(statSync(modelDst).size / 1e6).toFixed(1)} MB)`);
} else {
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length < 1e6) throw new Error(`model download too small (${buf.length} bytes)`);
  writeFileSync(modelDst, buf);
  console.log(`model  face_landmarker.task (${(buf.length / 1e6).toFixed(1)} MB)`);
}
