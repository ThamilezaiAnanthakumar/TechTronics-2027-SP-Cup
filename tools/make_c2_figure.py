"""C2 figure (README_UPDATE_03 U15): per-ITA-group MAE with 95 % bootstrap CIs, +-tau band around the
overall MAE, AiPEX dev (CV predictions) and AiPEX-test (frozen CP3). Inputs: outputs/round1/c2_*_groups.csv
written by tools/c2_aipex.py. Output: docs/report/figures/c2_ita.pdf."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
sets = [("AiPEX-dev (CV)", "outputs/round1/c2_dev_CP3_groups.csv", BLUE),
        ("AiPEX-test", "outputs/round1/c2_test_CP3_groups.csv", ORANGE)]
fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
for i, (name, path, col) in enumerate(sets):
    g = pd.read_csv(path)
    overall = (g.mae - g.deviation).iloc[0]
    tau = max(1.0, 0.25 * overall)
    x = [j + (i - 0.5) * 0.28 for j in range(len(g))]
    ax.errorbar(x, g.mae, yerr=[g.mae - g.mae_ci_lo, g.mae_ci_hi - g.mae], fmt="o", ms=3.5, color=col,
                capsize=2, lw=1, label=name)
    for xi, n in zip(x, g.n):
        ax.text(xi, 0.6, f"n={n}", ha="center", fontsize=5.5, color=INK2)
    ax.axhspan(overall - tau, overall + tau, color=col, alpha=0.10, lw=0)
    ax.axhline(overall, color=col, lw=0.8, ls="--")
ax.set_xticks(range(3)); ax.set_xticklabels(["ITA I–II", "III–IV", "V–VI"])
ax.set_ylabel("Video-level MAE (bpm)"); ax.set_ylim(0, 18); ax.grid(axis="y", color=GRID, lw=0.5)
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig("docs/report/figures/c2_ita.pdf"); print("written docs/report/figures/c2_ita.pdf")
