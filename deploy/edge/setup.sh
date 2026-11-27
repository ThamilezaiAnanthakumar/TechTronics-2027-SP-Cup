#!/usr/bin/env bash
# Set up the edge runtime on a Raspberry Pi 4 (64-bit Raspberry Pi OS).
# Run from anywhere:  bash deploy/edge/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"

if [ "$(getconf LONG_BIT)" != 64 ]; then
    # uname -m is not enough: a Pi 4 boots a 64-bit kernel under the 32-bit OS too (uname -m = aarch64)
    echo "error: 32-bit userland. mediapipe has no 32-bit ARM wheels: flash the 64-bit Raspberry Pi OS." >&2
    exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
    echo "warning: $(uname -m), not aarch64. Continuing (useful for testing on a laptop)." >&2
fi

# mediapipe 0.10.18 has wheels for Python 3.9-3.12 only.
PY=""
for v in 3.12 3.11 3.10 3.9; do
    if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
done
if [ -z "$PY" ]; then
    echo "error: need Python 3.9-3.12 for mediapipe 0.10.18 (found: $(python3 --version 2>&1))." >&2
    echo "Raspberry Pi OS Bookworm ships 3.11. On Trixie (3.13), install 3.12 with uv:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12" >&2
    exit 1
fi

echo "using $PY ($($PY --version))"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

# Check that the pipeline modules import and the API face.py needs is there.
PYTHONPATH="$HERE/../../src/python" "$VENV/bin/python" - <<'EOF'
import cv2, numpy, mediapipe as mp
from facevitals import face, rppg
mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5).close()
print("ok: mediapipe", mp.__version__, "| opencv", cv2.__version__, "| numpy", numpy.__version__)
EOF

echo "done. activate with: source $VENV/bin/activate"
