"""C2 skin-tone reporting for AiPEX (README 5 C2 row, README_UPDATE_03 U14).

  python tools/c2_aipex.py PER_VIDEO_CSV OUT_PREFIX [LABEL]

PER_VIDEO_CSV: columns country, video, ref, est (one row per video; e.g. outputs/aipex_iter/E019/per_video.csv
for dev CV, outputs/round1/aipex_test_<exp>/per_video.csv for the single test run).
ITA per video: outputs/repro/aipex_ita.csv (computed by Thamil's aipex_hr from the skin pixels of the three
crops: mean RGB -> rgb2lab (MATLAB default sRGB, D65) -> ITA = atan2(L*-50, b*)*180/pi; a camera-derived,
illumination-dependent proxy, not a clinical Fitzpatrick rating).
Groups: I-II (Very light, Light), III-IV (Intermediate, Tan), V-VI (Brown, Dark).
Outputs: <OUT_PREFIX>_groups.csv, _crosstab.csv, _country.csv, _within_country.csv and a printed summary.
tau = max(1.0 bpm, 25 % of the population MAE), pre-specified in README Section 5.
Pass = the 95 % bootstrap CI of the group's deviation from the overall MAE lies within +-tau;
otherwise "inconclusive" (never "fail" on these sample sizes).
"""
import sys

import numpy as np
import pandas as pd

GROUP = {"Very light (I)": "I-II", "Light (II)": "I-II", "Intermediate (III)": "III-IV", "Tan (IV)": "III-IV",
         "Brown (V)": "V-VI", "Dark (VI)": "V-VI"}
ORDER = ["I-II", "III-IV", "V-VI"]
B = 5000


def main(per_video, prefix, label=""):
    d = pd.read_csv(per_video)
    ita = pd.read_csv("outputs/repro/aipex_ita.csv")
    d = d.merge(ita, on=["country", "video"], how="left")
    d["group"] = d.ITA_class.map(GROUP)
    d["ae"] = (d.est - d.ref).abs()
    n_all = len(d)
    d = d[np.isfinite(d.ae)]
    overall = d.ae.mean()
    tau = max(1.0, 0.25 * overall)
    rng = np.random.default_rng(2027)
    rows = []
    for g in ORDER:
        a = d.ae[d.group == g].to_numpy()
        rest = d.ae.to_numpy()
        if not len(a):
            continue
        bm, bd = np.empty(B), np.empty(B)
        for i in range(B):                     # resample within the group and the population
            s = a[rng.integers(0, len(a), len(a))]
            p = rest[rng.integers(0, len(rest), len(rest))]
            bm[i] = s.mean(); bd[i] = s.mean() - p.mean()
        lo, hi = np.percentile(bd, [2.5, 97.5])
        rows.append(dict(group=g, n=len(a), mae=a.mean(), mae_ci_lo=np.percentile(bm, 2.5),
                         mae_ci_hi=np.percentile(bm, 97.5), deviation=a.mean() - overall, dev_ci_lo=lo, dev_ci_hi=hi,
                         verdict="pass" if (lo >= -tau and hi <= tau) else "inconclusive"))
    G = pd.DataFrame(rows)
    # permutation test: range of group MAEs under shuffled labels
    lab = d.group.to_numpy(); ae = d.ae.to_numpy()
    stat = lambda L: np.ptp([ae[L == g].mean() for g in ORDER if (L == g).any()])
    obs = stat(lab)
    perm = np.array([stat(rng.permutation(lab)) for _ in range(B)])
    p_perm = float((perm >= obs).mean())
    X = pd.crosstab(d.group, d.country).reindex(ORDER)
    Cn = d.groupby("country").ae.agg(["count", "mean"]).rename(columns={"count": "n", "mean": "mae"})
    W = d.groupby(["country", "group"]).ae.agg(["count", "mean"]).rename(columns={"count": "n", "mean": "mae"})
    W = W[W.n >= 5]
    G.to_csv(prefix + "_groups.csv", index=False); X.to_csv(prefix + "_crosstab.csv")
    Cn.to_csv(prefix + "_country.csv"); W.to_csv(prefix + "_within_country.csv")
    print(f"C2 {label}: {len(d)}/{n_all} videos, overall MAE {overall:.2f}, tau {tau:.2f}, "
          f"permutation p (range of group MAEs) {p_perm:.3f}")
    print(G.round(2).to_string(index=False)); print(X.to_string()); print(Cn.round(2).to_string()); print(W.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
