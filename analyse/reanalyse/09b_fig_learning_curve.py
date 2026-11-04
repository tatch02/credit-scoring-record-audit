# -*- coding: utf-8 -*-
"""
Figure for the first learning curve (09_learning_curve.py).

Usage: python 09b_fig_learning_curve.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = config.OUTPUTS_DIR / "figures" / "fig7_learning_curve.png"


def main():
    df = pd.read_csv(config.OUTPUTS_DIR / "learning_curve.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.errorbar(df["n"], df["AUC_full_mean"], yerr=df["AUC_full_sd"],
                marker="o", ms=6, lw=2, capsize=4, color="#2f5b8c",
                label="Full model (24 variables)")
    ax.plot(df["n"], df["AUC_hist2var_mean"], marker="s", ms=6, lw=2,
            ls="--", color="#b1442b",
            label="Prior-credit history only (2 variables)")

    ax.axhline(0.5, color="#94a3b8", lw=1, ls=":")
    ax.text(df["n"].iloc[-1], 0.515, "chance level", fontsize=8.5,
            color="#64748b", ha="right")

    ax.set_xlabel("Training sample size (n)", fontsize=11)
    ax.set_ylabel("Out-of-fold ROC-AUC", fontsize=11)
    ax.set_title("Performance does not improve with sample size",
                 fontsize=12.5, fontweight="bold", loc="left")
    ax.set_ylim(0.45, 1.02)
    ax.set_xticks(df["n"])
    ax.grid(alpha=0.25, lw=0.7)
    ax.legend(frameon=False, fontsize=9.5, loc="center left", bbox_to_anchor=(0.02, 0.42))

    d = df["AUC_full_mean"].iloc[-1] - df["AUC_full_mean"].iloc[0]
    ax.annotate(f"$\\Delta$AUC = {d:+.3f} from n={int(df['n'].iloc[0])} to n={int(df['n'].iloc[-1])}",
                xy=(df["n"].iloc[-1], df["AUC_full_mean"].iloc[-1]),
                xytext=(-8, -34), textcoords="offset points",
                fontsize=9.5, color="#2f5b8c", ha="right")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("written:", OUT)


if __name__ == "__main__":
    main()
