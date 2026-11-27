"""
U11/U12 analysis of the teammate's SNR-fusion result (contrib/snr_fusion, M. I. Nadha).

Reads the per-video table written by their code (run_c2.py --analyze-only -> c2_subjects.csv),
does NOT recompute any HR, and writes:
  outputs/snr_fusion/per_video.csv      one row per AiPEX video: ids, split, ITA, ref, pos / eq / snr estimates and errors
  outputs/snr_fusion/summary.csv        claim table per subset (all 88, ITA-valid 85, dev 62, test 26, countries)
  outputs/snr_fusion/paired_stats.csv   like-for-like pos vs snr: paired bootstrap CI (5,000 resamples over videos), sign test, Wilcoxon
  outputs/snr_fusion/ita_groups.csv     ITA groups with their own population definition (finite ITA)

Usage:
  python tools/snr_fusion_analysis.py <c2_subjects.csv> [out_dir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEED = 2027
NBOOT = 5000


def split_ids(name):
    lines = (ROOT / "shared" / "splits" / f"aipex_{name}.txt").read_text().split()
    return {l.strip().replace("/", "_") for l in lines if l.strip()}


def describe(D, pre):
    e = D[pre + "bias"].to_numpy(float)
    a = np.abs(e)
    ok = np.isfinite(a)
    return dict(n=int(ok.sum()), mae=a[ok].mean(), median_ae=np.median(a[ok]),
                within5_pct=100 * (a[ok] < 5).mean(), bias=e[ok].mean())


def paired(D, a="", b="snr_", rng=None):
    """b - a per video (negative = b better)."""
    x, y = D[a + "mae"].to_numpy(float), D[b + "mae"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = y[ok] - x[ok]
    n = len(d)
    idx = rng.integers(0, n, size=(NBOOT, n))
    bm = d[idx].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    better, worse = int((d < 0).sum()), int((d > 0).sum())
    b1, w1 = int((d < -1).sum()), int((d > 1).sum())
    p_sign = stats.binomtest(better, better + worse, 0.5).pvalue if better + worse else np.nan
    p_sign1 = stats.binomtest(b1, b1 + w1, 0.5).pvalue if b1 + w1 else np.nan
    p_wil = stats.wilcoxon(y[ok], x[ok]).pvalue if n > 0 else np.nan
    return dict(n=n, mae_a=x[ok].mean(), mae_b=y[ok].mean(), diff_mean=d.mean(), ci_lo=lo, ci_hi=hi,
                better=better, worse=worse, ties=n - better - worse, p_sign=p_sign,
                better_gt1=b1, worse_gt1=w1, p_sign_gt1=p_sign1, p_wilcoxon=p_wil)


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "snr_fusion"
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(src)
    S = S[S["dataset"] == "AiPEX"].copy()
    dev, test = split_ids("dev"), split_ids("test")
    S["split"] = np.where(S["subject"].isin(dev), "dev", np.where(S["subject"].isin(test), "test", "none"))
    S["ita_valid"] = np.isfinite(S["ITA"])

    keep = ["subject", "country", "split", "ITA", "ITA_class", "ITA_group", "ita_valid", "hr_ref_mean"]
    for pre in ["", "eq_", "snr_"]:
        keep += [pre + k for k in ("hr_est", "bias", "mae", "n_windows", "hr_est_meanagg", "mae_meanagg")]
    keep += [c for c in S.columns if c.startswith("snr_w_")]
    P = S[[c for c in keep if c in S.columns]].rename(columns={"bias": "pos_bias", "mae": "pos_mae", "n_windows": "pos_n_windows", "hr_est_meanagg": "pos_hr_est_meanagg", "mae_meanagg": "pos_mae_meanagg",
                                                              "hr_est": "pos_hr_est"})
    P.to_csv(out / "per_video.csv", index=False)

    subsets = {"all_88": S, "ita_valid_85": S[S["ita_valid"]], "dev_62": S[S["split"] == "dev"],
               "test_26": S[S["split"] == "test"], "India": S[S["country"] == "India"],
               "Sierra_Leone": S[S["country"] == "Sierra_Leone"]}
    rows = []
    for name, D in subsets.items():
        for meth, pre in [("pos", ""), ("pos_eq", "eq_"), ("pos_snr", "snr_")]:
            rows.append(dict(subset=name, method=meth, **describe(D, pre)))
    T = pd.DataFrame(rows)
    T.to_csv(out / "summary.csv", index=False)

    rows = []
    for name, D in subsets.items():
        for a, b in [("", "snr_"), ("", "eq_"), ("eq_", "snr_")]:
            rng = np.random.default_rng(SEED)
            rows.append(dict(subset=name, a=a or "pos_", b=b, **paired(D, a, b, rng)))
    Q = pd.DataFrame(rows)
    Q.to_csv(out / "paired_stats.csv", index=False)

    V = S[S["ita_valid"]]
    rows = []
    for meth, pre in [("pos", ""), ("pos_snr", "snr_")]:
        pop = V[pre + "mae"].mean()
        for g in ["I-II", "III-IV", "V-VI"]:
            G = V[V["ITA_group"] == g]
            rows.append(dict(method=meth, group=g, n=len(G), mae=G[pre + "mae"].mean(), population_mae=pop,
                             deviation=G[pre + "mae"].mean() - pop))
    pd.DataFrame(rows).to_csv(out / "ita_groups.csv", index=False)

    pd.set_option("display.width", 200)
    print(T.round(2).to_string(index=False))
    print(Q.round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nvideos without ITA:", ", ".join(S.loc[~S["ita_valid"], "subject"]))
    print("videos with NaN estimate:", {pre or "pos_": list(S.loc[~np.isfinite(S[pre + "mae"]), "subject"])
                                        for pre in ["", "eq_", "snr_"]})
    print("split counts:", S["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
