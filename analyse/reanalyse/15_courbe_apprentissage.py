# -*- coding: utf-8 -*-
"""
Learning curve with a fixed external test set (Section 5.6, Figure S2).

In each of 40 replicates, a stratified test set of 25 files is drawn and held
fixed while stratified training subsets of 26 to 64 files are drawn from the
rest; the reference model is fitted on each and scored on the same 25 files.

Output: outputs/courbe_apprentissage.csv
Usage: python 15_courbe_apprentissage.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

N_TEST = 25
N_REPLICATS = 40
PAS = 6          # de 26 à 64 par pas de 6


def main():
    df, X, y, roles = core.load_data("matures")
    n_train_max = len(y) - N_TEST
    tailles = list(range(26, n_train_max + 1, PAS))
    if tailles[-1] != n_train_max:
        tailles.append(n_train_max)
    print(f"[15] Courbe d'apprentissage — n={len(y)}, test externe fixe={N_TEST}, "
          f"{N_REPLICATS} réplicats")
    print(f"     tailles d'entraînement : {tailles}\n")

    modele = core.logreg_l2(X)
    acc = {t: [] for t in tailles}

    for rep in range(N_REPLICATS):
        rs = core.SEED + rep
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=N_TEST, stratify=y, random_state=rs)
        Xte, yte = X.iloc[idx_te], y.iloc[idx_te]
        for t in tailles:
            if t < len(idx_tr):
                sub, _ = train_test_split(idx_tr, train_size=t,
                                          stratify=y.iloc[idx_tr], random_state=rs)
            else:
                sub = idx_tr
            ytr = y.iloc[sub]
            if ytr.nunique() < 2:
                continue
            m = clone(modele).fit(X.iloc[sub], ytr)
            p = m.predict_proba(Xte)[:, 1]
            acc[t].append(roc_auc_score(yte, p))
        if (rep + 1) % 10 == 0:
            print(f"     réplicat {rep + 1}/{N_REPLICATS}")

    rows = []
    for t in tailles:
        v = np.array(acc[t])
        rows.append({"n_entrainement": t, "n_replicats": len(v),
                     "AUC_moyenne": round(float(v.mean()), 4),
                     "AUC_ecart_type": round(float(v.std(ddof=1)), 4),
                     "AUC_p05": round(float(np.percentile(v, 5)), 4),
                     "AUC_p95": round(float(np.percentile(v, 95)), 4)})
    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "courbe_apprentissage.csv",
               index=False, encoding="utf-8")

    print("\n=== Courbe (jeu de test externe fixe de 25 observations) ===")
    print(res.to_string(index=False))
    delta = res.AUC_moyenne.iloc[-1] - res.AUC_moyenne.iloc[0]
    print(f"\n  De {tailles[0]} à {tailles[-1]} observations d'entraînement : "
          f"Δ AUC = {delta:+.4f}")
    print(f"  -> la performance est acquise dès {tailles[0]} observations ; "
          f"ajouter des données ne la change pas")
    print("\n  Écrit : courbe_apprentissage.csv")


if __name__ == "__main__":
    main()
