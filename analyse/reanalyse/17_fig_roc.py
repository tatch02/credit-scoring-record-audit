# -*- coding: utf-8 -*-
"""
Figure: ROC, precision-recall and calibration of the reference model (Figure S1).

Out-of-fold probabilities averaged over repetitions; produced at the exact
insertion width.

Outputs: outputs/figures/fig2_roc_pr_calibration.png,
         outputs/figures/dp/supp_fig_s1_roc_pr_calibration.pdf
Usage: python 17_fig_roc.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # polices TrueType incorporées
import matplotlib.pyplot as plt
import numpy as np
import core
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)
from sklearn.calibration import calibration_curve

LARGEUR_PT = 455.2
POUCES = LARGEUR_PT / 72.27
ACCENT, GRIS = "#8C2F39", "#5D6470"


def main():
    df, X, y, roles = core.load_data("matures")
    yv = y.astype(int).values
    print("[17] Probabilités out-of-fold (CV imbriquée répétée)...")
    proba = core.oof_proba(core.logreg_l2(X), X, y)

    fig, axes = plt.subplots(1, 3, figsize=(POUCES, POUCES * 0.36), dpi=300)

    # --- ROC
    fpr, tpr, _ = roc_curve(yv, proba)
    auc = roc_auc_score(yv, proba)
    ax = axes[0]
    ax.plot(fpr, tpr, color=ACCENT, lw=1.6)
    ax.plot([0, 1], [0, 1], color=GRIS, lw=0.8, ls=":")
    ax.set_xlabel("False positive rate", fontsize=7.5)
    ax.set_ylabel("True positive rate", fontsize=7.5)
    ax.set_title(f"ROC  (AUC = {auc:.2f})", fontsize=8)

    # --- précision-rappel
    pr, rc, _ = precision_recall_curve(yv, proba)
    ap = average_precision_score(yv, proba)
    ax = axes[1]
    ax.plot(rc, pr, color=ACCENT, lw=1.6)
    ax.axhline(yv.mean(), color=GRIS, lw=0.8, ls=":")
    ax.text(0.02, yv.mean() + 0.02, "prevalence", fontsize=7.2, color=GRIS)
    ax.set_xlabel("Recall", fontsize=7.5)
    ax.set_ylabel("Precision", fontsize=7.5)
    ax.set_title(f"Precision–recall  (AP = {ap:.2f})", fontsize=8)

    # --- calibration
    frac, mean_pred = calibration_curve(yv, proba, n_bins=6, strategy="quantile")
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=GRIS, lw=0.8, ls=":")
    ax.plot(mean_pred, frac, "-o", color=ACCENT, lw=1.4, markersize=3.4)
    ax.set_xlabel("Mean predicted probability", fontsize=7.5)
    ax.set_ylabel("Observed frequency", fontsize=7.5)
    ax.set_title("Calibration", fontsize=8)

    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=7.2)
        ax.grid(alpha=0.2, linewidth=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.6, w_pad=1.4)
    for dest in (core.config.OUTPUTS_DIR / "figures" / "fig2_roc_pr_calibration.png",
                 core.config.PROJECT_ROOT / "article" / "figures" / "fig2_roc_pr_calibration.png"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  écrit : {dest.relative_to(core.config.PROJECT_ROOT)}")
    # supplément D&P : PDF vectoriel à la taille exacte d'insertion (pas de recadrage)
    supp = core.config.OUTPUTS_DIR / "figures" / "dp" / "supp_fig_s1_roc_pr_calibration.pdf"
    supp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(supp, facecolor="white")
    print(f"  écrit : {supp.relative_to(core.config.PROJECT_ROOT)}")
    plt.close(fig)
    print(f"  AUC={auc:.4f}  AP={ap:.4f}  (à comparer au Tableau des résultats)")


if __name__ == "__main__":
    main()
