"""
Dataset loaders. Each yields Subject records with the video(s) and the reference.

Reference kinds:
  "ppg"  contact PPG waveform + times -> reference HR per 10 s window (UBFC, MPU)
  "mean" one HR value for the whole recording (AiPEX: Masimo mean HR, CLBP: spot reading)
  None   no ground truth (SRMC-N)

Default root is the folder that contains the repo (D:/SPCUP), where the datasets were
downloaded. Override with --data-root or the FACEVITALS_DATA environment variable.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .face import REPO

UBFC_EXCLUDE = {11, 18, 20, 24, 27}   # 27: no image consent; others: pulse-oximeter dropouts (README)


@dataclass
class Subject:
    dataset: str
    subject: str
    kind: str                    # "face" (full frame) or "crop" (pre-cropped ROI clips)
    video: object                # Path, or {region: Path} for crops
    ref: str | None = None       # "ppg", "hr", "mean" or None
    meta: dict = field(default_factory=dict)

    def reference(self, trace_t=None, fps=None):
        """('ppg', t, ppg) | ('hr', t, hr) | ('mean', hr) | None."""
        if self.ref == "ppg":
            load = {"UBFC": load_ubfc_ppg, "MPU": load_mpu_ppg}[self.dataset]
            return ("ppg",) + load(self.meta["gt_path"], trace_t, fps)
        if self.ref == "hr":
            return ("hr",) + load_mpu_hr(self.meta["gt_path"], trace_t, fps)
        if self.ref == "mean":
            return ("mean", float(self.meta["hr_ref"]))
        return None


def data_root(root=None):
    return Path(root or os.environ.get("FACEVITALS_DATA") or REPO.parent)


def _num(s):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else 0


# --------------------------------------------------------------- UBFC-rPPG (DATASET_2)

def load_ubfc_ppg(gt_path, trace_t=None, fps=None):
    lines = [l for l in Path(gt_path).read_text().splitlines() if l.strip()]
    ppg = np.array(lines[0].split(), float)
    t = np.array(lines[2].split(), float)
    return t, ppg


def ubfc(root):
    vids = list((root / "UBFC-rPPG").glob("*/subject*/vid.avi"))           # Google-Drive download layout
    vids += list((REPO / "data/raw/UBFC/DATASET_2").glob("subject*/vid.avi"))  # repo layout
    seen = set()
    for v in sorted(vids, key=lambda p: _num(p.parent.name)):
        sid = _num(v.parent.name)
        gt = v.parent / "ground_truth.txt"
        if sid in UBFC_EXCLUDE or sid in seen or not gt.exists():
            continue
        seen.add(sid)
        yield Subject("UBFC", f"subject{sid}", "face", v, "ppg", dict(gt_path=gt))


# --------------------------------------------------------------- MPU-rPPG

def load_mpu_ppg(csv, trace_t=None, fps=None):
    d = pd.read_csv(csv)
    n = len(d)
    # one CSV row per video frame (Count = frame index), oximeter resampled to the video
    if trace_t is not None and len(trace_t) >= n:
        t = np.asarray(trace_t[:n], float)
    else:
        t = d["Count"].to_numpy(float) / (fps or 60.0)
    return t, d["PPG"].to_numpy(float)


def load_mpu_hr(csv, trace_t=None, fps=None):
    """Oximeter HR column (bpm), one value per video frame.

    Used as the MPU reference: HR computed from the PPG column disagrees with the device
    HR by 11-45 bpm on recordings 8, 9 and 10 (likely harmonics/artefacts in the exported
    waveform), while the device HR is stable and matches on 5 and the root recording.
    """
    t, _ = load_mpu_ppg(csv, trace_t, fps)
    return t, pd.read_csv(csv)["HR"].to_numpy(float)


def mpu(root):
    base = root / "MPU rppg"
    for d in sorted(base.glob("*/"), key=lambda p: _num(p.name)):
        v, c = d / "Output.mp4", d / "Output.csv"
        if v.exists() and c.exists():
            yield Subject("MPU", f"mpu_{d.name}", "face", v, "hr", dict(gt_path=c))
    v, c = base / "Output.mkv", base / "Output.csv"
    if v.exists() and c.exists():
        yield Subject("MPU", "mpu_root_mkv", "face", v, "hr", dict(gt_path=c))


# --------------------------------------------------------------- CLBP-400 (samples)

def clbp(root):
    base = root / "CLBP-400_Samples"
    spec = pd.read_excel(base / "CLBP400_Dataset_Specifications.xlsx")
    spec["Subject ID"] = spec["Subject ID"].astype(str).str.strip()
    spec = spec.set_index("Subject ID")
    for v in sorted(base.glob("*.mov")):
        sid = v.stem.split("_")[0]
        if sid not in spec.index:
            continue
        row = spec.loc[sid]
        yield Subject("CLBP", sid, "face", v, "mean",
                      dict(hr_ref=row["Heart Rate"], age=row["Age"], gender=row["Gender"],
                           lux=row["Light intensity (Lux)"]))


# --------------------------------------------------------------- AiPEX rppg_biases

def aipex(root):
    base = root / "rppg_biases-main"
    gts = {}
    for country in ("India", "Sierra_Leone"):
        g = pd.read_csv(base / "Ground_Truth" / f"{country}_GT.csv", encoding="utf-8-sig")
        g.columns = [c.strip() for c in g.columns]
        for _, r in g.iterrows():
            gts[(country, str(r["Subject"]).strip())] = float(r["Heart_Rate(BPM)"])
    clips = {}
    for v in (base / "Data").glob("*/*.avi"):
        m = re.match(r"(.+)_(forehead|leftcheek|rightcheek)$", v.stem)
        if m:
            clips.setdefault((v.parent.name, m[1]), {})[m[2]] = v
    for (country, rec) in sorted(clips, key=lambda k: (k[0], _num(k[1]))):
        if (country, rec) in gts:
            yield Subject("AiPEX", f"{country}_{rec}", "crop", clips[(country, rec)], "mean",
                          dict(hr_ref=gts[(country, rec)], country=country))


# --------------------------------------------------------------- SRMC-N (organizer, no GT)

def srmc(root):
    by = {}
    for v in sorted((root / "SRMC_N").iterdir()):
        m = re.match(r"S_N(\d+)_(.*)\.(mp4|mov)$", v.name, re.I)
        if not m:
            continue
        sid, tag = int(m[1]), m[2].lower()
        rank = 0 if tag == "resting" else (1 if "bright" in tag else 2)
        if sid not in by or rank < by[sid][0]:
            by[sid] = (rank, v)
    for sid in sorted(by):
        yield Subject("SRMC", f"S_N{sid}", "face", by[sid][1], None, dict(file=by[sid][1].name))


LOADERS = dict(UBFC=ubfc, MPU=mpu, CLBP=clbp, AiPEX=aipex, SRMC=srmc)


def subjects(names, root=None):
    root = data_root(root)
    for n in names:
        yield from LOADERS[n](root)
