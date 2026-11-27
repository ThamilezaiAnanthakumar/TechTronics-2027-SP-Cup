"""
C2 evaluation: the same HR pipeline on every dataset, error stratified by skin tone (ITA).

    python src/python/run_c2.py                                  # all datasets with ground truth, one process
    python src/python/run_c2.py --shard 1/4   (... up to 4/4)    # 4 parallel processes, start each separately
    python src/python/run_c2.py --analyze-only                   # merge shard tables + statistics
    python src/python/run_c2.py --datasets UBFC --max-frames 600 # quick test

MediaPipe on Windows is unreliable when one process spawns several workers that import it
at once, so parallelism is by independent processes (--shard), not a process pool.

Per subject: face traces (cached in outputs/traces_py/) -> POS HR in 10 s windows every 1 s
-> error against the reference, and ITA from the same skin pixels.
  ppg reference  (UBFC, MPU):   per-window error, subject MAE = mean over its windows
  mean reference (AiPEX, CLBP): video HR = median of the window HRs, error vs the single reference

Outputs in outputs/tables/:
  c2_windows.csv   one row per window (ppg datasets)
  c2_subjects.csv  one row per subject: dataset, ITA, class, MAE, ...
  c2_groups.csv    per ITA group / class: N, MAE with bootstrap 95 % CI, deviation, pass/fail
  c2_regression.csv  OLS  subject MAE ~ ITA + dataset
"""

import mediapipe  # noqa: F401  # must load before scipy/pandas on Windows (DLL init conflict)
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facevitals import datasets as dsets                      # noqa: E402
from facevitals.face import (REPO, crop_traces, face_traces,    # noqa: E402
                             load_face_config, summarize_ita, tracked_face_traces)
from facevitals.rppg import (default_centers, error_stats, kalman_denoise, ppg_hr,  # noqa: E402
                             sanity_hr, sanity_hr_bg, sanity_hr_snr)

TABLES = REPO / "outputs" / "tables"
TRACES = REPO / "outputs" / "traces_py"
GT_DATASETS = ["UBFC", "MPU", "CLBP", "AiPEX"]


def load_c2_config():
    return json.loads((REPO / "config" / "c2_config.json").read_text())


# --------------------------------------------------------------- traces (cached)

_DETECTOR = None


def get_traces(s, cfg, max_frames=None, rerun=False, module="tracker"):
    """module "tracker": port of the MATLAB face module (RetinaFace, Kalman/Hungarian, KLT,
    One-Euro, occlusion hysteresis). "simple": MediaPipe largest face (earlier development
    version). Crop datasets (AiPEX) skip face detection, so both use the same traces."""
    global _DETECTOR
    tag = f"{s.dataset}_{s.subject}" + (f"_first{max_frames}" if max_frames else "")
    tdir = TRACES if (s.kind == "crop" or module == "simple") else TRACES / "tracker"
    npz, js = tdir / f"{tag}.npz", tdir / f"{tag}_ita.json"
    if npz.exists() and js.exists() and not rerun:
        d = dict(np.load(npz, allow_pickle=False))
        # face traces from before the background reference was added are recomputed
        if s.kind == "crop" or "rgb_bg" in d:
            d["ita_samples"] = {k: [tuple(v) for v in vs] for k, vs in json.loads(js.read_text()).items()}
            d["fps"] = float(d["fps"])
            return d
    roi, skin, occl = load_face_config()
    if s.kind == "crop":
        d = crop_traces(s.video, skin, ita_every=cfg["ita_every"])
    elif module == "tracker":
        if _DETECTOR is None:
            from facevitals.retinaface import RetinaFaceDetector
            from facevitals.tracker import face_params
            fp = face_params()
            _DETECTOR = RetinaFaceDetector(fp["det"]["threshold"], fp["det"]["minSize"])
        d = tracked_face_traces(s.video, detector=_DETECTOR, ita_every=cfg["ita_every"],
                                max_frames=max_frames, max_width=cfg.get("tracker_max_width"))
    else:
        d = face_traces(s.video, roi, skin, occl, max_width=cfg["max_width"],
                        ita_every=cfg["ita_every"], max_frames=max_frames)
    tdir.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **{k: v for k, v in d.items() if k != "ita_samples"})
    js.write_text(json.dumps(d["ita_samples"]))
    return d


# --------------------------------------------------------------- one subject

def evaluate_subject(s, cfg, max_frames=None, rerun=False, module="tracker"):
    t0 = time.time()
    d = get_traces(s, cfg, max_frames, rerun, module)
    _, ita = summarize_ita(d["ita_samples"])
    q = np.asarray(d["quality"])
    row = dict(dataset=s.dataset, subject=s.subject, ref=s.ref or "none",
               face_module="crop" if s.kind == "crop" else module, frames=len(q),
               face_found_pct=100 * float((q > 0).mean()) if len(q) else np.nan, **ita)
    row.update({k: v for k, v in s.meta.items() if k in ("age", "gender", "lux", "country")})
    win, step = cfg["window_s"], cfg["step_s"]
    windows = []

    ref = s.reference(d["t"], d["fps"])
    bg = d.get("rgb_bg")
    names = [str(n) for n in np.atleast_1d(d["roi_names"])]
    regions = [d["rgb_" + n] for n in names if "rgb_" + n in d]
    snr_w = {}

    # Kalman denoising (team method, thamilezai_dev): rgb_face weighted by its skin pixels,
    # each region by its own skin pixels (crops carry no pixel counts -> unweighted)
    fps = float(d["fps"]) if np.isfinite(float(d["fps"])) and float(d["fps"]) > 0 else 30.0
    npix = np.asarray(d["npix"], float) if "npix" in d else None
    if npix is not None and npix.ndim == 2 and "occluded" in d:
        n_face = np.where(np.asarray(d["occluded"]) > 0, 0, npix).sum(axis=1)
    else:
        n_face = None
    kf_cache = {}

    def kf_face():
        if "face" not in kf_cache:
            kf_cache["face"] = kalman_denoise(d["rgb_face"], fps, npix=n_face)
        return kf_cache["face"]

    def kf_regions():
        if "regions" not in kf_cache:
            kf_cache["regions"] = [kalman_denoise(x, fps, npix=npix[:, r] if npix is not None and npix.ndim == 2
                                                  and r < npix.shape[1] else None)
                                   for r, x in enumerate(regions)]
        return kf_cache["regions"]

    def snr_method(c, use_bg, key, power=None, regs=None):
        power = cfg.get("snr_power", 1.0) if power is None else power
        hr, _, w = sanity_hr_snr(d["t"], regions if regs is None else regs, win, step, centers=c,
                                 rgb_bg=bg if use_bg else None, power=power, return_weights=True)
        snr_w[key] = w
        return hr

    methods = {"": lambda c: sanity_hr(d["t"], d["rgb_face"], win, step, centers=c)[0],
               "bg_": lambda c: sanity_hr_bg(d["t"], d["rgb_face"], bg, win, step, centers=c,
                                             kappa=cfg.get("bg_kappa", 1.0))[0],
               "snr_": lambda c: snr_method(c, False, "snr_"),
               "snrbg_": lambda c: snr_method(c, True, "snrbg_"),
               # control: same region-spectrum fusion with equal weights (SNR ** 0 = 1)
               "eq_": lambda c: snr_method(c, False, "eq_", power=0.0),
               # Kalman denoising before POS (team method), alone and stacked with SNR fusion
               "kf_": lambda c: sanity_hr(d["t"], kf_face(), win, step, centers=c)[0],
               "kfsnr_": lambda c: snr_method(c, False, "kfsnr_", regs=kf_regions()),
               "kfsnrbg_": lambda c: snr_method(c, True, "kfsnrbg_", regs=kf_regions())}
    if ref is None:
        pass
    elif ref[0] in ("ppg", "hr"):
        if ref[0] == "ppg":
            _, tg, ppg = ref
            hr_ref, tc = ppg_hr(tg, ppg, win, step)
        else:
            # device HR series (MPU oximeter): reference = mean device HR inside each window
            _, tg, hrg = ref
            ok = np.isfinite(tg) & np.isfinite(hrg) & (hrg > 0)
            tg, hrg = tg[ok], hrg[ok]
            tc = default_centers(tg, win, step)
            hr_ref = np.array([hrg[(tg >= c - win / 2) & (tg < c + win / 2)].mean() for c in tc])
        row["hr_ref_mean"] = float(np.nanmean(hr_ref))
        est = {}
        for pre, fn in methods.items():
            est[pre] = fn(tc)
            row.update({pre + k: v for k, v in error_stats(est[pre], hr_ref).items()})
        windows = [dict(dataset=s.dataset, subject=s.subject, tc=c, hr_ref=r,
                        **{"hr_est" + ("_" + pre.rstrip("_") if pre else ""): est[pre][i] for pre in est})
                   for i, (c, r) in enumerate(zip(tc, hr_ref))]
    else:
        hr_ref = ref[1]
        row["hr_ref_mean"] = hr_ref
        for pre, fn in methods.items():
            hr_w = fn(None)
            e_hr = float(np.nanmedian(hr_w)) if np.isfinite(hr_w).any() else np.nan
            e_mean = float(np.nanmean(hr_w)) if np.isfinite(hr_w).any() else np.nan
            err = e_hr - hr_ref
            row.update({pre + "n_windows": int(np.isfinite(hr_w).sum()), pre + "mae": abs(err),
                        pre + "rmse": abs(err), pre + "r": np.nan, pre + "bias": err,
                        pre + "suc3_pct": 100.0 * (abs(err) < 3), pre + "suc5_pct": 100.0 * (abs(err) < 5),
                        pre + "hr_est": e_hr,
                        # mean of the window HRs (as in aipex_hr.m) instead of the median
                        pre + "hr_est_meanagg": e_mean, pre + "mae_meanagg": abs(e_mean - hr_ref)})
    for key, w in snr_w.items():       # mean share of the fused spectrum per region
        if key == "eq_":
            continue                   # equal weights by construction
        for n, x in zip(names, w):
            row[f"{key}w_{n}"] = float(x)
    row["has_bg"] = bg is not None
    row["seconds"] = round(time.time() - t0, 1)
    return row, windows


# --------------------------------------------------------------- statistics

def bootstrap_ci(x, n, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def group_table(S, by, cfg, rng):
    pop = S["mae"].mean()
    rows = []
    for g, G in S.groupby(by, sort=False):
        lo, hi = bootstrap_ci(G["mae"], cfg["bootstrap"], rng)
        dev = G["mae"].mean() - pop
        rows.append({"grouping": by, "group": g, "n_subjects": len(G),
                     "datasets": ", ".join(f"{k}:{v}" for k, v in G["dataset"].value_counts().items()),
                     "ITA_median": G["ITA"].median(), "mae": G["mae"].mean(), "mae_ci_lo": lo, "mae_ci_hi": hi,
                     "mae_median": G["mae"].median(), "rmse": float(np.sqrt((G["rmse"] ** 2).mean())),
                     "suc5_pct": G["suc5_pct"].mean(), "population_mae": pop, "deviation": dev,
                     "tau": cfg["tau_bpm"], "pass": abs(dev) <= cfg["tau_bpm"]})
    return pd.DataFrame(rows)


def ols(S):
    """subject MAE ~ ITA + dataset (dummies). Returns coefficient table."""
    ds = sorted(S["dataset"].unique())
    X = [np.ones(len(S)), S["ITA"].to_numpy(float)]
    names = ["intercept", "ITA (bpm per degree)"]
    for d in ds[1:]:
        X.append((S["dataset"] == d).to_numpy(float))
        names.append(f"dataset={d} (vs {ds[0]})")
    X = np.column_stack(X)
    y = S["mae"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - X.shape[1]
    res = y - X @ beta
    s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    tval = beta / se
    p = 2 * stats.t.sf(np.abs(tval), dof)
    return pd.DataFrame(dict(term=names, coef=beta, se=se, t=tval, p=p, n=len(y), dof=dof))


def analyze(S, cfg):
    rng = np.random.default_rng(cfg["seed"])
    S = S[np.isfinite(S["mae"]) & np.isfinite(S["ITA"])].copy()
    out = []
    order3 = ["I-II", "III-IV", "V-VI"]
    order6 = ["Very light", "Light", "Intermediate", "Tan", "Brown", "Dark"]
    for name, sub in [("all", S), ("ita_consistent", S[S["ita_consistent"]])]:
        for by, order in [("ITA_group", order3), ("ITA_class", order6)]:
            g = group_table(sub, by, cfg, rng)
            if g.empty:
                continue
            g["subset"] = name
            g["_o"] = g["group"].map({k: i for i, k in enumerate(order)})
            out.append(g.sort_values("_o").drop(columns="_o"))
    G = pd.concat(out, ignore_index=True)
    reg = ols(S) if S["dataset"].nunique() >= 1 and len(S) > S["dataset"].nunique() + 2 else pd.DataFrame()

    within = []
    for d, D in S.groupby("dataset"):
        if len(D) >= 5:
            rho, p = stats.spearmanr(D["ITA"], D["mae"])
            within.append(dict(dataset=d, n=len(D), spearman_rho=rho, p=p,
                               ITA_min=D["ITA"].min(), ITA_max=D["ITA"].max(), mae=D["mae"].mean()))
    return G, reg, pd.DataFrame(within)


METHOD_PREFIX = {"pos": "", "pos_bg": "bg_", "pos_eq": "eq_", "pos_snr": "snr_", "pos_snr_bg": "snrbg_",
                 "pos_kf": "kf_", "pos_kf_snr": "kfsnr_", "pos_kf_snr_bg": "kfsnrbg_"}


def select_method(S, method):
    """Copy the chosen method's columns into mae / rmse / suc5_pct used by the statistics."""
    pre = METHOD_PREFIX[method]
    if pre:
        S = S.copy()
        for k in ("mae", "rmse", "suc5_pct", "bias", "r", "n_windows"):
            if pre + k in S:
                S[k] = S[pre + k]
    return S


def report(S, cfg, method="pos"):
    S = select_method(S, method)
    print(f"\n##### method: {method} #####")
    G, reg, within = analyze(S, cfg)
    tag = "" if method == "pos" else "_" + method
    G.to_csv(TABLES / f"c2_groups{tag}.csv", index=False)
    reg.to_csv(TABLES / f"c2_regression{tag}.csv", index=False)
    within.to_csv(TABLES / f"c2_within_dataset{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["group", "n_subjects", "datasets", "ITA_median", "mae", "mae_ci_lo", "mae_ci_hi", "deviation", "pass"]
    print(f"\n=== Per dataset ===")
    print(S.groupby("dataset").agg(n=("mae", "size"), ITA_median=("ITA", "median"), mae=("mae", "mean"),
                                   rmse=("rmse", lambda x: np.sqrt((x ** 2).mean())),
                                   suc5_pct=("suc5_pct", "mean")).round(2).to_string())
    for sub in ["all", "ita_consistent"]:
        for by in ["ITA_group", "ITA_class"]:
            T = G[(G["subset"] == sub) & (G["grouping"] == by)]
            if len(T):
                print(f"\n=== {by}  ({sub} subjects, tau = {cfg['tau_bpm']} bpm, "
                      f"population MAE = {T['population_mae'].iloc[0]:.2f}) ===")
                print(T[cols].round(2).to_string(index=False))
    if len(reg):
        print("\n=== OLS: subject MAE ~ ITA + dataset ===")
        print(reg.round(4).to_string(index=False))
    if len(within):
        print("\n=== Within-dataset Spearman(ITA, MAE) ===")
        print(within.round(3).to_string(index=False))


def compare(S):
    """Baseline POS vs background reference vs SNR-weighted region fusion."""
    pd.set_option("display.width", 250)
    S = S[np.isfinite(S["mae"])]
    meth = [(m, p) for m, p in METHOD_PREFIX.items() if p + "mae" in S and S[p + "mae"].notna().any()]
    rows = []
    for d, D in S.groupby("dataset"):
        r = {"dataset": d, "n": len(D)}
        for m, p in meth:
            r[f"mae_{m}"] = D[p + "mae"].mean()
            r[f"suc5_{m}"] = D[p + "suc5_pct"].mean()
        rows.append(r)
    agg = [(m, p) for m, p in meth if p + "mae_meanagg" in S and S[p + "mae_meanagg"].notna().any()]
    if agg:
        print("\n=== Recording-level datasets: MAE with median (ours) vs mean (aipex_hr.m) of window HRs ===")
        R = S[S["ref"] == "mean"]
        print(R.groupby("dataset").agg(**{f"{m}|median": (p + "mae", "mean") for m, p in agg},
                                       **{f"{m}|mean": (p + "mae_meanagg", "mean") for m, p in agg}
                                       ).T.round(2).to_string())
    print("\n=== Mean subject MAE (bpm) and SUC@5 (%) per method ===")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    cols = ["dataset", "subject", "ITA"] + [p + "mae" for _, p in meth]
    wcols = [c for c in S.columns if c.startswith("snr_w_")]
    T = S[cols + wcols].copy()
    print("\n=== Per subject MAE (snr_w_* = mean share of each region in the SNR fusion) ===")
    print(T.round(2).to_string(index=False))
    T.to_csv(TABLES / "c2_method_compare.csv", index=False)
    if "ITA_group" in S:
        print("\n=== Mean MAE per ITA group and method ===")
        print(S.groupby("ITA_group")[[p + "mae" for _, p in meth]].mean().round(2).to_string())


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=GT_DATASETS, choices=list(dsets.LOADERS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="quick test: first N frames per video")
    ap.add_argument("--shard", default="1/1",
                    help="i/n: process every n-th subject starting at i. Start n processes "
                         "(one per shard) to run in parallel; then run --analyze-only to merge")
    ap.add_argument("--rerun", action="store_true", help="ignore cached traces")
    ap.add_argument("--analyze-only", action="store_true", help="merge all shard tables and report")
    ap.add_argument("--method", default="pos", choices=list(METHOD_PREFIX) + ["compare"],
                    help="HR method used for the skin-tone statistics; 'compare' prints baseline vs "
                         "background-reference per dataset and subject")
    ap.add_argument("--subjects", nargs="+", default=None, help="only these subject names (quick tests)")
    ap.add_argument("--face-module", default=None, choices=["tracker", "simple"],
                    help="face module: 'tracker' = port of the MATLAB module (default, config "
                         "face_module), 'simple' = MediaPipe largest face (earlier version)")
    args = ap.parse_args()
    cfg = load_c2_config()
    module = args.face_module or cfg.get("face_module", "tracker")
    TABLES.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{args.max_frames}" if args.max_frames else ""

    if args.analyze_only:
        parts = sorted(TABLES.glob(f"c2_subjects{suffix}_shard*.csv"))
        parts = [q for q in parts if q.stem.endswith("_simple") == (module == "simple")]
        if not parts:
            raise SystemExit(f"no c2_subjects{suffix}_shard*.csv in {TABLES}")
        S = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["dataset", "subject"], keep="last")
        S = S.sort_values(["dataset", "subject"])
        S.to_csv(TABLES / f"c2_subjects{suffix}.csv", index=False)
        wparts = [q for q in sorted(TABLES.glob(f"c2_windows{suffix}_shard*.csv"))
                  if q.stem.endswith("_simple") == (module == "simple")]
        if wparts:
            pd.concat([pd.read_csv(p) for p in wparts]).to_csv(TABLES / f"c2_windows{suffix}.csv", index=False)
        print(f"merged {len(parts)} shard(s): {len(S)} subjects")
        if args.method == "compare":
            compare(S)
        else:
            report(S, cfg, args.method)
        return

    i, n = (int(x) for x in args.shard.split("/"))
    subs = list(dsets.subjects(args.datasets, args.data_root))
    if args.subjects:
        subs = [x for x in subs if x.subject in args.subjects]
    subs.sort(key=lambda s: (s.dataset != "MPU", s.dataset, s.subject))   # longest videos spread first
    subs = subs[i - 1::n]
    tag = f"{suffix}_shard{i}of{n}_" + "-".join(args.datasets) + ("_simple" if module == "simple" else "") + ("_" + "-".join(args.subjects) if args.subjects else "")
    print(f"shard {i}/{n}: {len(subs)} subjects, face module: {module}", flush=True)
    rows, wins = [], []
    for k, s in enumerate(subs, 1):
        try:
            row, w = evaluate_subject(s, cfg, args.max_frames, args.rerun, module)
        except Exception:
            print(f"[{k}/{len(subs)}] ERROR {s.dataset} {s.subject}\n{traceback.format_exc()}", flush=True)
            continue
        rows.append(row)
        wins += w
        print(f"[{k}/{len(subs)}] {s.dataset:5s} {s.subject:22s} ITA={row['ITA']:6.1f} "
              f"{row['ITA_class']:<12s} MAE={row.get('mae', np.nan):6.2f}  MAE_bg={row.get('bg_mae', np.nan):6.2f}  MAE_eq={row.get('eq_mae', np.nan):6.2f}  MAE_snr={row.get('snr_mae', np.nan):6.2f}  "
              f"face={row['face_found_pct']:5.1f}%  {row['seconds']}s", flush=True)
        # write after every subject so a crash loses nothing
        pd.DataFrame(rows).to_csv(TABLES / f"c2_subjects{tag}.csv", index=False)
        if wins:
            pd.DataFrame(wins).to_csv(TABLES / f"c2_windows{tag}.csv", index=False)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
