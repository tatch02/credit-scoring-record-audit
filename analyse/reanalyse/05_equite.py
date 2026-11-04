# -*- coding: utf-8 -*-
"""
Subgroup audit by sex and marital status (Table S5).

Neither attribute enters the model. For each group: observed and predicted
default rates, AUC, recall and specificity. Groups of fewer than three files
are skipped; small groups give unstable estimates.

Output: outputs/equite.csv
Usage: python 05_equite.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
sys.stdout.reconfigure(encoding="utf-8")


def sub_metrics(y, proba, mask):
    yy, pp = y[mask], proba[mask]
    pred = (pp >= 0.5).astype(int)
    auc = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
    return {"n": int(mask.sum()),
            "taux défaut réel": round(float(yy.mean()), 2),
            "AUC": round(auc, 2) if not np.isnan(auc) else np.nan,
            "Recall(défaut)": round(recall_score(yy, pred, pos_label=1, zero_division=0), 2),
            "Spécificité": round(recall_score(yy, pred, pos_label=0, zero_division=0), 2),
            "taux prédiction défaut": round(float(pred.mean()), 2)}


def main():
    df, X, y, roles = core.load_data("matures")
    proba = core.oof_proba(core.logreg_l2(X), X, y)
    y = y.values

    rows = []
    for col in roles["holdout_equity"]:
        vals = df[col].astype(str)
        for v in sorted(vals.unique()):
            mask = (vals == v).values
            if mask.sum() < 3:
                continue
            rows.append({"Variable sensible": col, "Groupe": v, **sub_metrics(y, proba, mask)})

    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "equite.csv", index=False, encoding="utf-8")
    pd.set_option("display.max_columns", None, "display.width", 220)
    print("=== Équité par sous-groupe — modèle SANS variables sensibles (biais INDIRECT) ===")
    print("Note: ILLUSTRATIF : n faibles, aucune conclusion ferme.\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
