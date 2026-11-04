# -*- coding: utf-8 -*-
"""
Where does the signal come from? Ablations by block of variables.

Re-estimates the reference model on subsets of variables to measure the
contribution of each block: the credit history (prior-loan appraisal and the
no-prior-credit indicator), the indicators of missing income and debt ratio,
the officer's rating of market reputation, and what remains without the
credit history.

Output: outputs/ablation.csv
Usage: python 03_ablation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

KEYS = ["AUC-ROC", "PR-AUC", "Bal.acc", "Recall(défaut)", "Spécificité"]


def main():
    _, X, y, _ = core.load_data()

    hist = [c for c in X.columns if "apprec_credit_ord" in c or "sans_historique" in c]
    miss = [c for c in X.columns if c.startswith("miss_")]
    moral = [c for c in X.columns if "oralit" in c.lower()]

    subsets = {
        "Complet (référence)": X.columns.tolist(),
        "SANS historique de crédit": [c for c in X.columns if c not in hist],
        "SANS indicateurs de manque": [c for c in X.columns if c not in miss],
        "SANS historique NI manque": [c for c in X.columns if c not in hist + miss],
        "Historique de crédit SEUL": hist,
        "SANS moralité (jugement d'agent)": [c for c in X.columns if c not in moral],
    }

    print(f"[03] Ablations (modèle de référence : LogReg L2, nested CV) — n={len(y)}")
    print(f"     historique={hist} | manque={miss} | moralité={moral}\n")

    rows = []
    for label, cols in subsets.items():
        Xs = X[cols]
        point, ci = core.evaluate(core.logreg_l2(Xs), Xs, y)
        rows.append({"Modèle (variables)": label, "n_var": len(cols), **core.fmt(point, ci, KEYS)})
        print(f"  OK {label} ({len(cols)} var)")

    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "ablation.csv", index=False, encoding="utf-8")
    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n=== Ablations (AUC etc. en point [IC 95 %]) ===")
    print(res[["Modèle (variables)", "n_var", "AUC-ROC", "PR-AUC", "Bal.acc", "Recall(défaut)"]].to_string(index=False))


if __name__ == "__main__":
    main()
