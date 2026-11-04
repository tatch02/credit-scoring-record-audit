# -*- coding: utf-8 -*-
"""
Figures built from the aggregate outputs (English labels).

Reads outputs/*.csv and recomputes the reference model's out-of-fold
probabilities where a figure needs them.

Outputs: outputs/figures/*.png
Usage: python 06_figures.py
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
sys.stdout.reconfigure(encoding="utf-8")

FIG = core.config.OUTPUTS_DIR / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "axes.titlesize": 11})
BLUE, ORANGE, GREY = "#1f5fa8", "#e08214", "#888888"

# ---- Traductions FR -> EN ----
MODEL_EN = {"Baseline majoritaire": "Majority baseline", "Scorecard (LogReg simple)": "Scorecard (logistic)",
            "LogReg L2 (C validé)": "Logistic (L2, tuned)", "LogReg L1 / LASSO": "Logistic (L1/LASSO)",
            "Random Forest": "Random Forest", "SVM RBF": "SVM (RBF)"}
ABL_EN = {"Complet (référence)": "Full (reference)", "SANS historique de crédit": "Without credit history",
          "SANS indicateurs de manque": "Without missingness flags",
          "SANS historique NI manque": "Without history nor missingness",
          "Historique de crédit SEUL": "Credit history only",
          "SANS moralité (jugement d'agent)": "Without reputation"}
GROUP_EN = {"FEMININ": "Female", "MASCULIN": "Male", "CELIBATAIRE": "Single", "MARIE": "Married"}
FEAT_EN = {
    "num__apprec_credit_ord": "Prior credit rating", "num__miss_Revenu_glo": "Income missing",
    "num__miss_Taux_d_end": "Debt ratio missing", "num__contexte_ord": "Market context",
    "num__sans_historique_credit": "No prior credit", "num__Durée (Mois)": "Loan duration",
    "num__Ancienneté dans la profession (Année)": "Years in profession", "num__Age": "Age",
    "cat__Justificatifs des activités ?_NON": "No activity proof",
    "cat__Justificatifs des activités ?_OUI": "Has activity proof",
    "cat__Actif dans culture pendant les 03 dernières saisons ?_NON": "Not active (3 seasons)",
    "cat__Actif dans culture pendant les 03 dernières saisons ?_OUI": "Active (3 seasons)",
    "cat__Moralité par rapport au marché_Mauvaise": "Poor reputation",
    "cat__Moralité par rapport au marché_Bonne": "Good reputation",
    "cat__Propriétaire du logement_OUI": "Home owner"}


def parse_ci(s):
    m = re.match(r"\s*([\d.]+)\s*\[\s*([\d.]+)\s*[–-]\s*([\d.]+)\s*\]", str(s))
    return tuple(float(x) for x in m.groups()) if m else (float(s), np.nan, np.nan)


def feat_label(raw):
    if raw in FEAT_EN:
        return FEAT_EN[raw]
    return raw.replace("num__", "").replace("cat__", "")


def fig_forest():
    df = pd.read_csv(core.config.OUTPUTS_DIR / "results_pipeline.csv")
    df = df[df["Modèle"] != "Baseline majoritaire"]
    labels = [MODEL_EN.get(m, m) for m in df["Modèle"]]
    pts = [parse_ci(v) for v in df["AUC-ROC"]]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for yi, (p, lo, hi) in zip(y, pts):
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2)
        ax.plot(p, yi, "o", color=BLUE, ms=7)
    ax.axvline(0.5, color=GREY, ls="--", lw=1, label="Chance (0.50)")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("AUC-ROC (out-of-fold, 95% CI)"); ax.set_xlim(0.45, 1.02)
    ax.set_title("Discrimination (AUC-ROC): models are indistinguishable")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig1_forest_auc.png", dpi=300); plt.close(fig)


def fig_roc_pr_calib(proba, yv):
    fpr, tpr, _ = roc_curve(yv, proba)
    prec, rec, _ = precision_recall_curve(yv, proba)
    frac_pos, mean_pred = calibration_curve(yv, proba, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(fpr, tpr, color=BLUE, lw=2); ax[0].plot([0, 1], [0, 1], ls="--", color=GREY)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC curve")
    ax[1].plot(rec, prec, color=BLUE, lw=2)
    ax[1].axhline(yv.mean(), ls="--", color=GREY, label=f"Prevalence ({yv.mean():.2f})")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision–Recall curve")
    ax[1].legend(loc="lower left")
    ax[2].plot(mean_pred, frac_pos, "o-", color=ORANGE, lw=2)
    ax[2].plot([0, 1], [0, 1], ls="--", color=GREY, label="Perfect calibration")
    ax[2].set_xlabel("Mean predicted probability"); ax[2].set_ylabel("Observed default frequency")
    ax[2].set_title("Calibration curve"); ax[2].legend(loc="upper left")
    fig.suptitle("Reference model (Logistic L2), out-of-fold", y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "fig2_roc_pr_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_confusion(proba, yv):
    cm = confusion_matrix(yv, (proba >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No default", "Default"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No default", "Default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (reference model, threshold 0.5)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(FIG / "fig6_confusion_matrix.png", dpi=300); plt.close(fig)


def fig_ablation():
    df = pd.read_csv(core.config.OUTPUTS_DIR / "ablation.csv")
    labels = [ABL_EN.get(m, m) for m in df["Modèle (variables)"]]
    pts = [parse_ci(v) for v in df["AUC-ROC"]]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for yi, (p, lo, hi) in zip(y, pts):
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2)
        ax.plot(p, yi, "o", color=BLUE, ms=7)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("AUC-ROC (out-of-fold, 95% CI)"); ax.set_xlim(0.75, 1.02)
    ax.set_title("Ablations: the predictive signal is redundant")
    fig.tight_layout(); fig.savefig(FIG / "fig3_ablation.png", dpi=300); plt.close(fig)


def fig_selection():
    s = pd.read_csv(core.config.OUTPUTS_DIR / "selection_frequency.csv", index_col=0)
    s = s["freq"].head(12)[::-1]
    labels = [feat_label(i) for i in s.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(np.arange(len(s)), s.values, color=BLUE)
    ax.set_yticks(np.arange(len(s))); ax.set_yticklabels(labels)
    ax.set_xlabel("LASSO selection frequency across folds"); ax.set_xlim(0, 1)
    ax.set_title("Variable selection frequency (LASSO)")
    fig.tight_layout(); fig.savefig(FIG / "fig4_selection_frequency.png", dpi=300); plt.close(fig)


def fig_equity():
    df = pd.read_csv(core.config.OUTPUTS_DIR / "equite.csv")
    groups = [GROUP_EN.get(g, g) for g in df["Groupe"]]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w/2, df["taux défaut réel"], w, label="Actual default rate", color=GREY)
    ax.bar(x + w/2, df["taux prédiction défaut"], w, label="Predicted default rate", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1)
    ax.set_title("Equity: model reproduces group disparities in default")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig5_equity.png", dpi=300); plt.close(fig)


def main():
    fig_forest(); print("  OK fig1_forest_auc")
    _, X, y, _ = core.load_data("matures")
    proba = core.oof_proba(core.logreg_l2(X), X, y); yv = y.values   # calculé une fois
    fig_roc_pr_calib(proba, yv); print("  OK fig2_roc_pr_calibration")
    fig_confusion(proba, yv); print("  OK fig6_confusion_matrix")
    fig_ablation(); print("  OK fig3_ablation")
    fig_selection(); print("  OK fig4_selection_frequency")
    fig_equity(); print("  OK fig5_equity")
    print(f"\nFigures (EN) -> {FIG}")


if __name__ == "__main__":
    main()
