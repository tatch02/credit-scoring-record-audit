# -*- coding: utf-8 -*-
"""
Exploratory data analysis of the loan records.

Class balance of the outcome, missing values, descriptive statistics,
correlations with the outcome, and default rates by subgroup.

Outputs: outputs/eda_*.csv and outputs/figures/eda*.png (aggregates only).
Usage: python 00_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config, datamodel as dm, core
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

FIG = config.OUTPUTS_DIR / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "axes.titlesize": 11})
BLUE, ORANGE, GREY = "#1f5fa8", "#e08214", "#888888"

TR = {"Age": "Age", "Revenu global": "Total income", "Traite mensuelle": "Monthly installment",
      "Charges fixes": "Fixed charges", "Taux d'endettement": "Debt ratio",
      "Montant du prêt demandé": "Requested loan amount", "Montant du prêt accordé": "Granted amount",
      "Durée (Mois)": "Loan duration (months)", "Ancienneté dans la profession (Année)": "Years in profession",
      "Ancienneté à l'institution": "Tenure at MFI", "Mouvements confiés sur les 12 derniers mois": "Funds entrusted (12mo)",
      "Historique de crédit": "Credit history", "superficie_ha": "Farm size (ha)", "contexte_ord": "Market context",
      "apprec_credit_ord": "Prior credit rating", "sans_historique_credit": "No prior credit",
      "Traite chez confrère": "Installment elsewhere", "Engagement Chez Confrère": "Commitment elsewhere",
      "Moralité par rapport au marché": "Market reputation", "Justificatifs des activités ?": "Activity proof",
      "Actif dans culture pendant les 03 dernières saisons ?": "Active (3 seasons)",
      "Nature garantie": "Guarantee type", "Nature de la culture": "Crop type",
      "Propriétaire du logement": "Home ownership", "Autre activité": "Other activity",
      "miss_Revenu_glo": "Income missing", "miss_Taux_d_end": "Debt ratio missing"}

VAL = {"OUI": "Yes", "NON": "No", "Bonne": "Good", "Mauvaise": "Poor", "mais": "maize",
       "Tomates": "tomatoes", "Tomates et pastèques": "tomatoes+melon", "aucune": "none",
       "Garantie financière": "financial", "Garantie réelle et personnelle": "real+personal",
       "Garantie personnelle et Financière": "personal+financial"}


def tr(name):
    if name in TR:
        return TR[name]
    for fr in sorted(TR, key=len, reverse=True):
        if name.startswith(fr + "_"):
            val = name[len(fr) + 1:]
            return f"{TR[fr]} = {VAL.get(val, val)}"
    return name


def main():
    # --- données : brut nettoyé, restreint aux prêts matures ---
    df = dm.clean_and_encode(dm.load_raw())
    tgt_col = dm.find(df, "situation de credit")
    df["target"] = df[tgt_col].map(dm.target_matures)
    mat = df[df["target"].notna()].copy(); mat["target"] = mat["target"].astype(int)
    _, allr = df[df[tgt_col].map(dm.target_all).notna()], df[tgt_col].map(dm.target_all).dropna().astype(int)

    print(f"[EDA] matured loans n={len(mat)} | target(default=1): {mat['target'].value_counts().to_dict()}")

    # === 0. Construction de la cible (comment le label est bâti) ===
    STAT_EN = {"En souffrance": "In default",
               "Totalement remboursé sans difficultés": "Fully repaid (no difficulty)",
               "Totalement remboursé avec difficultés": "Repaid (with difficulty)",
               "En cours en souffrance": "In progress, distressed",
               "En cours sains": "In progress, performing"}

    def klass(v):
        s = str(v).lower()
        if s.startswith("en cours"):
            return "Excluded (in progress)"
        return "Default (1)" if "souffrance" in s else "Sound (0)"

    sit = df[tgt_col].value_counts()
    tc = pd.DataFrame({"original_status": [STAT_EN.get(i, i) for i in sit.index],
                       "n": sit.values, "mapped_class": [klass(i) for i in sit.index]})
    tc.to_csv(config.OUTPUTS_DIR / "eda_target_construction.csv", index=False)
    print("\nTarget construction (5 statuses -> binary label):")
    print(tc.to_string(index=False))

    cmap = {"Default (1)": ORANGE, "Sound (0)": BLUE, "Excluded (in progress)": GREY}
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(8, 3.2))
    yv = np.arange(len(tc))[::-1]
    ax.barh(yv, tc["n"], color=[cmap[c] for c in tc["mapped_class"]])
    ax.set_yticks(yv); ax.set_yticklabels(tc["original_status"])
    ax.set_xlabel("Number of loans")
    ax.set_title("Target construction — 5 statuses -> default / sound / excluded (censored)")
    ax.legend(handles=[mpatches.Patch(color=c, label=k) for k, c in cmap.items()],
              loc="lower right", fontsize=9)
    for yi, n in zip(yv, tc["n"]):
        ax.text(n + 0.4, yi, str(n), va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "eda0_target_construction.png", dpi=300); plt.close(fig)

    # === 0b. Les 11 prêts « en cours » exclus — l'exclusion biaise-t-elle ? ===
    inprog = df[df[tgt_col].astype(str).str.lower().str.startswith("en cours")].copy()
    sexc = dm.find(df, "sexe")
    cmp_vars = [(dm.find(df, "Montant du prêt demandé"), "Requested loan amount"),
                ("superficie_ha", "Farm size (ha)"),
                (dm.find(df, "Ancienneté dans la profession"), "Years in profession"),
                ("apprec_credit_ord", "Prior credit rating (0-2)"),
                ("contexte_ord", "Market context (0-2)")]
    cmp_rows = [{"variable": name, "matured (n=89)": round(mat[c].mean(), 2),
                 "in-progress (n=11)": round(inprog[c].mean(), 2)} for c, name in cmp_vars]
    cmp_rows.append({"variable": "% female", "matured (n=89)": round((mat[sexc] == "FEMININ").mean(), 2),
                     "in-progress (n=11)": round((inprog[sexc] == "FEMININ").mean(), 2)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(config.OUTPUTS_DIR / "eda_inprogress_vs_matured.csv", index=False)
    print("\nExcluded in-progress loans (n=11: 6 distressed, 5 performing) vs matured — profile:")
    print(cmp.to_string(index=False))

    # === 1. Valeurs manquantes ===
    miss = (mat.isna().mean().sort_values(ascending=False) * 100).round(1)
    miss = miss[miss > 0]
    miss.to_csv(config.OUTPUTS_DIR / "eda_missingness.csv", header=["pct_missing"])
    print("\nMissingness (% of matured loans):")
    print(miss.to_string())

    # === 2. Stats descriptives (numériques) ===
    num_cols = [dm.find(df, k) or k for k in
                ["Age", "Revenu global", "Traite mensuelle", "Charges fixes", "taux d'endettement",
                 "Montant du prêt demandé", "Durée", "Ancienneté dans la profession", "Ancienneté à l'institution",
                 "Mouvements", "Historique de crédit"]]
    num_cols = [c for c in num_cols if c in mat.columns] + ["superficie_ha", "apprec_credit_ord", "contexte_ord"]
    desc = mat[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
    desc.to_csv(config.OUTPUTS_DIR / "eda_descriptive_numeric.csv")
    print("\nDescriptive statistics (numeric) saved.")

    # === 3. Corrélations avec la cible (données encodées du modèle) ===
    _, Xd, yd, _ = core.load_data("matures")
    Xenc = pd.get_dummies(Xd, drop_first=True).astype(float)
    corr = Xenc.corrwith(yd).abs().sort_values(ascending=False).head(12)
    corr.to_csv(config.OUTPUTS_DIR / "eda_correlations_target.csv", header=["abs_corr"])

    # === 4. Taux de défaut par sous-groupe ===
    rows = []
    for col in [dm.find(df, "sexe"), dm.find(df, "situation matrimoniale")]:
        for v, g in mat.groupby(col):
            rows.append({"variable": col, "group": v, "n": len(g), "default_rate": round(g["target"].mean(), 2)})
    grp = pd.DataFrame(rows)
    grp.to_csv(config.OUTPUTS_DIR / "eda_default_by_group.csv", index=False)
    print("\nDefault rate by subgroup:")
    print(grp.to_string(index=False))

    # ========================= FIGURES (EN) =========================
    # eda1 — missingness
    fig, ax = plt.subplots(figsize=(6.5, 3))
    y = np.arange(len(miss))[::-1]
    ax.barh(y, miss.values, color=[ORANGE if v > 50 else BLUE for v in miss.values])
    ax.set_yticks(y); ax.set_yticklabels([tr(i) for i in miss.index])
    ax.set_xlabel("% missing"); ax.set_xlim(0, 100)
    ax.set_title(f"Missing values — financial fundamentals absent for ~{miss.max():.0f}% of matured loans")
    for yi, v in zip(y, miss.values):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "eda1_missingness.png", dpi=300); plt.close(fig)

    # eda2 — target distribution (matured loans) : effectifs + % clairs
    counts = mat["target"].value_counts().reindex([1, 0])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(["Default (1)", "Sound (0)"], counts.values, color=[ORANGE, BLUE], width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v}  ({v/len(mat)*100:.0f}%)",
                ha="center", fontsize=11)
    ax.set_ylabel("Number of matured loans"); ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Target distribution — 55% default prevalence (n=89)")
    fig.tight_layout(); fig.savefig(FIG / "eda2_target.png", dpi=300); plt.close(fig)

    # eda3 — correlations with target
    fig, ax = plt.subplots(figsize=(6.5, 4))
    yv = np.arange(len(corr))[::-1]
    ax.barh(yv, corr.values, color=BLUE)
    ax.set_yticks(yv); ax.set_yticklabels([tr(i) for i in corr.index])
    ax.set_xlabel("|correlation| with default"); ax.set_xlim(0, 1)
    ax.set_title("Association with default (top variables)")
    fig.tight_layout(); fig.savefig(FIG / "eda3_correlations.png", dpi=300); plt.close(fig)

    # eda4 — distributions of key numeric variables
    keyn = [c for c in [dm.find(df, "Age"), dm.find(df, "Montant du prêt demandé"), "superficie_ha",
                        dm.find(df, "Durée"), dm.find(df, "Ancienneté dans la profession"),
                        dm.find(df, "Historique de crédit")] if c in mat.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for a, c in zip(axes.ravel(), keyn):
        a.hist(mat[c].dropna(), bins=12, color=BLUE, edgecolor="white")
        a.set_title(tr(c), fontsize=10); a.set_ylabel("count")
    fig.suptitle("Distributions of key numeric variables (most non-normal)", y=1.01)
    fig.tight_layout(); fig.savefig(FIG / "eda4_numeric_distributions.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # eda5 — default rate by subgroup
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    g2 = grp.copy()
    g2["label"] = g2["group"].map({"MASCULIN": "Male", "FEMININ": "Female",
                                   "MARIE": "Married", "CELIBATAIRE": "Single", "VEUF": "Widowed"}).fillna(g2["group"])
    x = np.arange(len(g2))
    ax.bar(x, g2["default_rate"], color=[ORANGE if r > 0.55 else BLUE for r in g2["default_rate"]])
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(g2["label"], g2["n"])], fontsize=9)
    ax.set_ylabel("Default rate"); ax.set_ylim(0, 1)
    ax.axhline(mat["target"].mean(), ls="--", color=GREY, label=f"Overall ({mat['target'].mean():.2f})")
    ax.set_title("Observed default rate by subgroup (small n — caution)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "eda5_default_by_group.png", dpi=300); plt.close(fig)

    print(f"\n[EDA] tables -> outputs/eda_*.csv | figures -> {FIG} (eda1..eda5)")


if __name__ == "__main__":
    main()
