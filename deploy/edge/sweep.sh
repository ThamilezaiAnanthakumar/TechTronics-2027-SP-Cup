#!/usr/bin/env bash
# Timing sweep on one video: frame size x face-detection period.
#   bash deploy/edge/sweep.sh --ubfc 1
#   bash deploy/edge/sweep.sh outputs/edge/synthetic/vid.avi --gt outputs/edge/synthetic/ground_truth.txt
# Results: outputs/edge/*.json, plus a summary table at the end.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

for w in 640 320; do
    for n in 1 2 3 5; do
        echo "=== width $w, detect every $n"
        "$PY" "$HERE/bench_video.py" "$@" --max-width "$w" --det-every "$n" | grep -E 'fps|^total|^pos|throttled'
        if command -v vcgencmd >/dev/null; then
            # let the Pi cool down between runs so each one starts from the same temperature;
            # at most 3 min: a Pi without a heatsink can idle above 60 C and would wait forever
            for _ in $(seq 18); do t=$(vcgencmd measure_temp | tr -dc '0-9.'); (( ${t%.*} > 60 )) || break; sleep 10; done
        fi
    done
done

"$PY" "$HERE/summary.py"
