# -*- coding: utf-8 -*-
"""
Role of the recorded loan term.

The median term is one month, shorter than any crop cycle. Tests whether the
term drives performance by removing it from the model.

Output: outputs/duration_role.csv
Usage: python 11_role_duree.py
"""
import sys; sys.path.insert(0,'.')
import core, config, pandas as pd, numpy as np
_, X, y, _ = core.load_data("matures")
dcol = [c for c in X.columns if "Dur" in c][0]
print("Colonne duree :", repr(dcol), "\n")

KEYS = ["AUC-ROC", "PR-AUC", "Bal.acc", "Recall(défaut)"]
tests = {
    "Full model (reference)":      X.columns.tolist(),
    "Duration ALONE (1 variable)": [dcol],
    "Model WITHOUT duration":      [c for c in X.columns if c != dcol],
}
rows = []
for lab, cols in tests.items():
    Xs = X[cols]
    pt, ci = core.evaluate(core.logreg_l2(Xs), Xs, y)
    rows.append({"Model": lab, "n_var": len(cols), **core.fmt(pt, ci, KEYS)})
    print(" ok", lab)

res = pd.DataFrame(rows)
pd.set_option("display.max_columns", None, "display.width", 200)
print("\n=== Role de la duree ===")
print(res[["Model","n_var"]+KEYS].to_string(index=False))
res.to_csv(config.OUTPUTS_DIR/"duration_role.csv", index=False, encoding="utf-8")

# separation brute : 1 mois vs reste
d = pd.to_numeric(X[dcol], errors="coerce")
short = d <= 1
print("\n=== 1 mois vs plus long ===")
print(f"  1 mois      : n={short.sum():3d}  taux defaut={y[short].mean():.2f}")
print(f"  > 1 mois    : n={(~short).sum():3d}  taux defaut={y[~short].mean():.2f}")
from sklearn.metrics import roc_auc_score
print(f"  AUC de la seule regle 'duree=1' : {roc_auc_score(y, short.astype(int)):.3f}")
