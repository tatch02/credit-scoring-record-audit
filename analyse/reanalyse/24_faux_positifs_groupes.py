# -*- coding: utf-8 -*-
"""
False-positive and false-negative rates by group (Table S6).

Errors of the reference model at a threshold of 0.5 on its out-of-fold
probabilities, by sex and marital status, with Wilson 95% intervals. Groups
and denominators smaller than five are not reported.

Output: outputs/faux_positifs_groupes.csv
Usage: python 24_faux_positifs_groupes.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core

SEUIL_GROUPE = 5


def taux(k, n):
    if n < SEUIL_GROUPE:                         # dénominateur trop petit : non rapporté
        return np.nan, np.nan, np.nan
    z, p = 1.959964, k / n                       # intervalle de Wilson, 95 %
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    df, X, y, roles = core.load_data("matures")
    yv = y.astype(int).values
    proba = core.oof_proba(core.logreg_l2(X), X, y)
    pred = (proba >= 0.5).astype(int)
    assert ((pred == 1) & (yv == 0)).sum() == 2 and ((pred == 0) & (yv == 1)).sum() == 3, \
        "la matrice ne correspond plus à celle de l'article (FP 2, FN 3)"

    rows = []
    for col in roles["holdout_equity"]:
        vals = df[col].astype(str).where(df[col].notna(), "non renseigné")
        for g in sorted(vals.unique()):
            m = (vals == g).values
            n = int(m.sum())
            if n < SEUIL_GROUPE:
                rows.append({"variable": col, "groupe": g, "n": f"< {SEUIL_GROUPE}"})
                continue
            neg, pos = m & (yv == 0), m & (yv == 1)
            fp_k, fn_k = int((pred[neg] == 1).sum()), int((pred[pos] == 0).sum())
            fpr, fpr_lo, fpr_hi = taux(fp_k, int(neg.sum()))
            fnr, fnr_lo, fnr_hi = taux(fn_k, int(pos.sum()))
            rows.append({
                "variable": col, "groupe": g, "n": n,
                "remboursés": int(neg.sum()), "faux positifs": fp_k,
                "taux FP": round(fpr, 3) if not np.isnan(fpr) else "",
                "IC95 FP": f"[{fpr_lo:.2f}-{fpr_hi:.2f}]" if not np.isnan(fpr) else "",
                "en défaut": int(pos.sum()), "faux négatifs": fn_k,
                "taux FN": round(fnr, 3) if not np.isnan(fnr) else "",
                "IC95 FN": f"[{fnr_lo:.2f}-{fnr_hi:.2f}]" if not np.isnan(fnr) else "",
            })

    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "faux_positifs_groupes.csv", index=False,
               encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", None)
    print("[24] Erreurs du modèle de référence par groupe (seuil 0,5, hors pli)\n")
    print(res.to_string(index=False))
    print("\n  Écrit : faux_positifs_groupes.csv")


if __name__ == "__main__":
    main()
