"""Video-level AiPEX metrics with bootstrap CIs (README 7 / 8.5 final evaluation).

  python tools/aipex_metrics.py PER_VIDEO_CSV OUT_CSV [LABEL]

MAE, median AE, RMSE, Pearson r, bias, 95 % limits of agreement (one value per video, so standard
Bland-Altman), within 3 / 5 bpm, coverage; 95 % bootstrap CI of MAE (5,000 resamples of videos);
the same per country.
"""
import sys

import numpy as np
import pandas as pd


def metrics(est, ref, rng):
    ok = np.isfinite(est) & np.isfinite(ref)
    e, r = est[ok], ref[ok]
    d = e - r
    ae = np.abs(d)
    boot = [ae[rng.integers(0, len(ae), len(ae))].mean() for _ in range(5000)]
    return dict(n=int(ok.sum()), coverage=100 * ok.mean(), MAE=ae.mean(), MAE_ci_lo=np.percentile(boot, 2.5),
                MAE_ci_hi=np.percentile(boot, 97.5), MedianAE=np.median(ae), RMSE=np.sqrt((d ** 2).mean()),
                r=np.corrcoef(e, r)[0, 1] if len(e) > 2 else np.nan, bias=d.mean(),
                LoA_low=d.mean() - 1.96 * d.std(ddof=1), LoA_high=d.mean() + 1.96 * d.std(ddof=1),
                within3=100 * (ae < 3).mean(), within5=100 * (ae < 5).mean())


def main(path, out, label=""):
    t = pd.read_csv(path)
    rng = np.random.default_rng(2027)
    rows = [dict(group="All", **metrics(t.est.to_numpy(), t.ref.to_numpy(), rng))]
    for c, g in t.groupby("country"):
        rows.append(dict(group=c, **metrics(g.est.to_numpy(), g.ref.to_numpy(), rng)))
    M = pd.DataFrame(rows)
    M.to_csv(out, index=False)
    print(label); print(M.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
