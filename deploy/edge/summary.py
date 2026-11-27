"""Table of all bench_video.py results in outputs/edge/*.json (also saved as outputs/edge/summary.csv)."""

import csv
import json
from pathlib import Path

OD = Path(__file__).resolve().parents[2] / "outputs" / "edge"

rows = []
for p in sorted(OD.glob("*.json")):
    r = json.loads(p.read_text())
    if "stages_ms" not in r:
        continue
    row = dict(name=r["name"], device=r["device"]["model"][:30], size="x".join(map(str, r["frame_size"])),
               det_every=r["args"]["det_every"], fps=round(r["fps"], 1), ms_frame=round(r["total_ms"], 1),
               detect_ms=round(r["detect_call_ms"], 1), face_pct=round(r["face_found_pct"], 1),
               temp_max=r["temp_max_c"], throttled=r["throttled_end"])
    for m, h in r["hr"].items():
        row[f"{m}_mae"] = round(h["mae"], 2) if "mae" in h else None
    rows.append(row)

if not rows:
    raise SystemExit(f"no results in {OD}")
keys = list(dict.fromkeys(k for r in rows for k in r))
widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
print("  ".join(k.ljust(widths[k]) for k in keys))
for r in rows:
    print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))
with open(OD / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)
print(f"\nsaved {OD / 'summary.csv'}")
