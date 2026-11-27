"""
ITA (Individual Typology Angle) survey of every local dataset (constraint C2).

Fast skin-tone overview without running the HR pipeline: N frames are sampled evenly
per video, the face is found, the ROI boxes of config/face_config.json are placed and
the repo skin mask selects skin pixels (same functions as run_c2.py, facevitals.face).
AiPEX clips are already ROI crops, so the mask is applied to the whole crop.

    python src/python/ita_survey.py
    python src/python/ita_survey.py --datasets SRMC --samples 30

Outputs (gitignored):
    outputs/tables/ita_survey_regions.csv   one row per video x region
    outputs/tables/ita_survey.csv           one row per video
    outputs/tables/ita_survey_summary.csv   class counts per dataset
    outputs/figures/ita_thumbs/<dataset>/<subject>.jpg  face + ROIs + ITA label
"""

import mediapipe  # noqa: F401  # must load before scipy/pandas on Windows (DLL init conflict)
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facevitals import datasets as dsets                                  # noqa: E402
from facevitals.face import (ITA_CLASSES, REPO, FaceBox, crop, load_face_config,  # noqa: E402
                             region_lab, roi_rects, skin_mask, summarize_ita)

MAX_WIDTH = 960


def sample_frames(path, n):
    """Yield RGB frames at n evenly spaced positions."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.unique(np.linspace(0, max(total - 1, 0), n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        if w > MAX_WIDTH:
            bgr = cv2.resize(bgr, (MAX_WIDTH, int(round(h * MAX_WIDTH / w))), interpolation=cv2.INTER_AREA)
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()


def survey_face(path, roi, skin, n, facebox):
    names = list(roi["names"])
    samples = {r: [] for r in names}
    n_face, thumb = 0, None
    for rgb in sample_frames(path, n):
        box = facebox(rgb)
        if box is None:
            continue
        n_face += 1
        rects = roi_rects(box, roi, names)
        for r, rect in rects.items():
            patch = crop(rgb, rect)
            if patch.size:
                v = region_lab(patch, skin_mask(patch, skin))
                if v is not None:
                    samples[r].append(v)
        if thumb is None or n_face == n // 2:
            thumb = (rgb, box, rects)
    return samples, n_face, thumb


def survey_crop(paths, skin, n):
    samples = {r: [] for r in paths}
    for r, p in paths.items():
        for rgb in sample_frames(p, n):
            v = region_lab(rgb, skin_mask(rgb, skin))
            if v is not None:
                samples[r].append(v)
    return samples, n, None


def save_thumb(thumb, path, lines):
    rgb, (x, y, w, h), rects = thumb
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for rect in rects.values():
        cv2.rectangle(img, rect[:2], rect[2:], (0, 255, 0), 1)
    pad = int(0.25 * w)
    H, W = img.shape[:2]
    face = img[max(0, int(y) - pad):min(H, int(y + h) + pad), max(0, int(x) - pad):min(W, int(x + w) + pad)]
    if face.size == 0:
        return
    face = cv2.resize(face, (240, int(240 * face.shape[0] / face.shape[1])))
    for i, txt in enumerate(lines):
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(face, txt, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), face)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(dsets.LOADERS), choices=list(dsets.LOADERS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--samples", type=int, default=20, help="frames sampled per video")
    ap.add_argument("--no-thumbs", action="store_true")
    args = ap.parse_args()

    roi, skin, _ = load_face_config()
    facebox = FaceBox()
    tables = REPO / "outputs" / "tables"
    thumbs = REPO / "outputs" / "figures" / "ita_thumbs"
    tables.mkdir(parents=True, exist_ok=True)

    reg_rows, vid_rows = [], []
    subs = list(dsets.subjects(args.datasets, args.data_root))
    print(f"{len(subs)} videos")
    for k, s in enumerate(subs, 1):
        try:
            if s.kind == "crop":
                samples, n_face, thumb = survey_crop(s.video, skin, args.samples)
            else:
                samples, n_face, thumb = survey_face(s.video, roi, skin, args.samples, facebox)
        except Exception as e:  # keep going on unreadable files
            print(f"[{k}/{len(subs)}] {s.dataset} {s.subject}: ERROR {e}")
            continue
        regions, summ = summarize_ita(samples)
        reg_rows += [dict(dataset=s.dataset, subject=s.subject, **r) for r in regions]
        vid_rows.append(dict(dataset=s.dataset, subject=s.subject, frames_with_face=n_face, **summ))
        ita = summ["ITA"]
        print(f"[{k}/{len(subs)}] {s.dataset} {s.subject}: ITA={ita:6.1f}  {summ['ITA_class']:<12} "
              f"regions={summ['ita_regions'] or '-'}  spread={summ['ita_spread']:.1f}")
        if thumb is not None and not args.no_thumbs:
            save_thumb(thumb, thumbs / s.dataset / f"{s.subject}.jpg",
                       [f"{s.dataset} {s.subject}",
                        f"ITA {ita:.0f} {summ['ITA_class']}" if np.isfinite(ita) else "ITA n/a"])
    facebox.close()

    R, V = pd.DataFrame(reg_rows), pd.DataFrame(vid_rows)
    R.to_csv(tables / "ita_survey_regions.csv", index=False)
    V.to_csv(tables / "ita_survey.csv", index=False)

    order = [c[1] for c in ITA_CLASSES] + ["Unknown"]
    S = (V.groupby("dataset")["ITA_class"].value_counts().unstack(fill_value=0)
         .reindex(columns=order, fill_value=0))
    stats = V.groupby("dataset")["ITA"].agg(["count", "median", "min", "max"]).round(1)
    stats["inconsistent"] = (~V["ita_consistent"] & V["ITA"].notna()).groupby(V["dataset"]).sum()
    stats["wb_warning"] = V["wb_warning"].groupby(V["dataset"]).sum()
    stats["median_b"] = V.groupby("dataset")["b"].median().round(1)
    S = S.join(stats)
    S.to_csv(tables / "ita_survey_summary.csv")
    pd.set_option("display.width", 200)
    print("\n" + S.to_string())
    print(f"\nTables in {tables}\nThumbnails in {thumbs}")


if __name__ == "__main__":
    main()
