# -*- coding: utf-8 -*-
"""
Figure of the learning curve (Figure S2), from courbe_apprentissage.csv.

Outputs: outputs/figures/fig7_learning_curve.png,
         outputs/figures/dp/supp_fig_s2_learning_curve.pdf
Usage: python 15b_fig_courbe.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # polices TrueType incorporées
import matplotlib.pyplot as plt
import pandas as pd
import config

LARGEUR_PT = 0.8 * 455.2          # \textwidth * 0.8, comme dans main.tex
POUCES = LARGEUR_PT / 72.27


def main():
    src = config.OUTPUTS_DIR / "courbe_apprentissage.csv"
    d = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(POUCES, POUCES * 0.55), dpi=200)
    ax.fill_between(d.n_entrainement, d.AUC_p05, d.AUC_p95,
                    color="#8C2F39", alpha=0.12, linewidth=0,
                    label="5th–95th percentile across replicates")
    ax.plot(d.n_entrainement, d.AUC_moyenne, "-o", color="#8C2F39",
            markersize=4, linewidth=1.6, label="Mean ROC-AUC (40 replicates)")

    a, b = d.AUC_moyenne.iloc[0], d.AUC_moyenne.iloc[-1]
    ax.annotate(f"{a:.3f}", (d.n_entrainement.iloc[0], a),
                textcoords="offset points", xytext=(4, 9), fontsize=8, color="#16181D")
    ax.annotate(f"{b:.3f}", (d.n_entrainement.iloc[-1], b),
                textcoords="offset points", xytext=(-6, -14), fontsize=8, color="#16181D")

    ax.set_xlabel("Training subsample size (files)", fontsize=9)
    ax.set_ylabel("ROC-AUC on held-out set", fontsize=9)
    ax.set_ylim(0.45, 1.02)                 # la ligne de base à 0,5 reste visible
    ax.axhline(0.5, color="#5D6470", linewidth=0.8, linestyle=":")
    ax.text(d.n_entrainement.iloc[0], 0.515, "no-information baseline (0.5)",
            ha="left", fontsize=7.5, color="#5D6470")
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center right")   # hors de la ligne de base
    fig.tight_layout()

    for dest in (config.OUTPUTS_DIR / "figures" / "fig7_learning_curve.png",
                 config.PROJECT_ROOT / "article" / "figures" / "fig7_learning_curve.png"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        print(f"  écrit : {dest.relative_to(config.PROJECT_ROOT)}")
    # supplément D&P : PDF vectoriel à la taille exacte d'insertion (pas de recadrage)
    supp = config.OUTPUTS_DIR / "figures" / "dp" / "supp_fig_s2_learning_curve.pdf"
    supp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(supp, facecolor="white")
    print(f"  écrit : {supp.relative_to(config.PROJECT_ROOT)}")
    plt.close(fig)
    print(f"  courbe : {a:.4f} à n={d.n_entrainement.iloc[0]} -> "
          f"{b:.4f} à n={d.n_entrainement.iloc[-1]}  (Δ {b - a:+.4f})")


if __name__ == "__main__":
    main()
