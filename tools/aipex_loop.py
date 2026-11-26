"""Helpers for the AiPEX iteration loop (README 8.5), operating on outputs/aipex_iter/.

  python tools/aipex_loop.py finalize EXP DECISION UBFC_DEV_MAE [--set-best]
      Fill ubfc_dev_mae/decision of the last log row of EXP; optionally make EXP the new best.
  python tools/aipex_loop.py nested EXP "hypothesis" PARAM VAL1:EXPa VAL2:EXPb ... [--seed 2027]
      Nested 5-fold CV over candidate runs that differ only in PARAM: for each outer fold, pick the
      value with the lowest MAE on the other 4 folds, score it on the held-out fold. Compares the
      resulting per-video errors with the current best (paired bootstrap, 5,000 resamples) and
      appends one log row for the whole hypothesis group.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IT = ROOT / "outputs" / "aipex_iter"
LOG = IT / "log.csv"
BEST = IT / "best_per_video.csv"


def git_hash():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def finalize(exp, decision, ubfc, set_best):
    L = pd.read_csv(LOG, dtype=str, keep_default_na=False)
    idx = L.index[L.exp_id == exp]
    if not len(idx):
        sys.exit(f"{exp} not in log")
    i = idx[-1]
    L.loc[i, "decision"] = decision
    L.loc[i, "ubfc_dev_mae"] = ubfc
    L.to_csv(LOG, index=False)
    if set_best:
        pd.read_csv(IT / exp / "per_video.csv").to_csv(BEST, index=False)
    print(f"{exp}: decision={decision}, ubfc_dev_mae={ubfc}, best={'yes' if set_best else 'unchanged'}")


def paired(ae_new, ae_best, seed=2):
    rng = np.random.default_rng(seed)
    g = ae_best - ae_new
    n = len(g)
    bg = np.array([g[rng.integers(0, n, n)].mean() for _ in range(5000)])
    return ae_new.mean() - ae_best.mean(), np.percentile(bg, [2.5, 97.5]), float((bg <= 0).mean())


def nested(exp, hyp, param, cands):
    tabs = {}
    for c in cands:
        val, e = c.split(":")
        tabs[val] = pd.read_csv(IT / e / "per_video.csv").sort_values(["country", "video"]).reset_index(drop=True)
    base = next(iter(tabs.values()))
    fold = base.fold.to_numpy()
    ae = {v: (t.est - t.ref).abs().to_numpy() for v, t in tabs.items()}
    err = {v: (t.est - t.ref).to_numpy() for v, t in tabs.items()}
    chosen, nest_ae, nest_err = [], np.zeros(len(base)), np.zeros(len(base))
    for f in range(1, 6):
        tr, te = fold != f, fold == f
        v = min(ae, key=lambda k: np.nanmean(ae[k][tr]))
        chosen.append(v)
        nest_ae[te] = ae[v][te]
        nest_err[te] = err[v][te]
    cv = np.mean([nest_ae[fold == f].mean() for f in range(1, 6)])
    rng = np.random.default_rng(1)
    ci = np.percentile([nest_ae[rng.integers(0, len(base), len(base))].mean() for _ in range(5000)], [2.5, 97.5])
    b = pd.read_csv(BEST).sort_values(["country", "video"]).reset_index(drop=True)
    ae_best = (b.est - b.ref).abs().to_numpy()
    d, dci, p = paired(nest_ae, ae_best)
    india = nest_ae[base.country == "India"].mean()
    sl = nest_ae[base.country == "Sierra_Leone"].mean()
    gap_best = abs(ae_best[b.country == "India"].mean() - ae_best[b.country == "Sierra_Leone"].mean())
    keep = d < 0 and dci[0] > 0 and abs(india - sl) <= gap_best + 0.5
    per_val = {v: round(float(np.nanmean(a)), 2) for v, a in ae.items()}
    row = dict(exp_id=exp, date=pd.Timestamp.now().strftime("%Y-%m-%d"), git_hash=git_hash(), hypothesis=hyp,
               change=f"nested CV over {param}", config_diff=f"{param} in {list(tabs)}",
               cv_mae=round(cv, 3), cv_mae_ci95=f"[{ci[0]:.2f}, {ci[1]:.2f}]", delta_vs_best=round(d, 3),
               p_paired_bootstrap=round(p, 4), india_mae=round(india, 3), sl_mae=round(sl, 3),
               within5_pct=round(100 * (nest_ae < 5).mean(), 1), bias=round(nest_err.mean(), 3), ubfc_dev_mae="",
               decision="keep (pending UBFC check)" if keep else "reject",
               notes=f"chosen per outer fold {chosen}; plain MAE per value {per_val}; gain CI [{dci[0]:.2f}, {dci[1]:.2f}]")
    L = pd.read_csv(LOG, dtype=str, keep_default_na=False)
    pd.concat([L, pd.DataFrame([row]).astype(str)]).to_csv(LOG, index=False)
    print(f"{exp}: nested CV MAE {cv:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | chosen {chosen} | per value {per_val} | "
          f"delta {d:+.2f} gain CI [{dci[0]:.2f}, {dci[1]:.2f}] | India {india:.2f} SL {sl:.2f} -> {row['decision']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "finalize":
        finalize(a[1], a[2], a[3], "--set-best" in a)
    elif a[0] == "nested":
        nested(a[1], a[2], a[3], [x for x in a[4:] if not x.startswith("--")])
