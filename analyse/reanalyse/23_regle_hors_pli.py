# -*- coding: utf-8 -*-
"""
The two-condition rule chosen within each fold and scored out of fold (Table 4).

Inside each training fold, among all conditions "field equals value", choose
the disjoint pair (A -> default, B -> repaid) that maximises the AUC of the
score 1 under A, 0 under B and the fold prevalence otherwise; apply it to the
held-out fold. Same resampling scheme and seeds as the models.

Output: outputs/regle_hors_pli.csv
Usage: python 23_regle_hors_pli.py
"""
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score


def conditions(X):
    """Toutes les égalités champ = valeur, en matrices booléennes (n x k)."""
    noms, cols = [], []
    for c in X.columns:
        v = X[c].astype(str).where(X[c].notna(), "<manquant>")
        for val in sorted(v.unique()):
            masque = (v == val).values
            if 0 < masque.sum() < len(v):
                noms.append(f"{c[:45]} = {val[:25]}")
                cols.append(masque)
    return noms, np.column_stack(cols)


def auc_trois_niveaux(y, A, B):
    """AUC exacte du score {1 sous A, 0 sous B, p ailleurs} par comptage.

    A et B sont des matrices (n x m) de paires candidates ; on renvoie m AUC.
    """
    pos, neg = (y == 1)[:, None], (y == 0)[:, None]
    M = ~(A | B)
    p1, p0, pm = (A & pos).sum(0), (B & pos).sum(0), (M & pos).sum(0)
    n1, n0, nm = (A & neg).sum(0), (B & neg).sum(0), (M & neg).sum(0)
    P, N = pos.sum(), neg.sum()
    gagne = p1 * (nm + n0) + pm * n0
    egal = p1 * n1 + p0 * n0 + pm * nm
    return (gagne + 0.5 * egal) / (P * N)


def meilleure_paire(y, C):
    k = C.shape[1]
    ia, ib = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    ia, ib = ia.ravel(), ib.ravel()
    garde = ia != ib
    ia, ib = ia[garde], ib[garde]
    disj = ~(C[:, ia] & C[:, ib]).any(0)
    ia, ib = ia[disj], ib[disj]
    auc = auc_trois_niveaux(y, C[:, ia], C[:, ib])
    j = int(np.argmax(auc))
    return ia[j], ib[j], auc[j]


def main():
    df, X, y, roles = core.load_data("matures")
    yv = y.astype(int).values
    noms, C = conditions(X)
    print(f"[23] {C.shape[1]} conditions candidates sur {X.shape[1]} prédicteurs, n={len(yv)}")

    # rappel : sélection sur les 89 dossiers (ce que fait le texte actuel)
    a, b, auc_in = meilleure_paire(yv, C)
    print(f"  Paire choisie sur tout l'échantillon : AUC {auc_in:.4f}")
    print(f"    A -> défaut : {noms[a]}\n    B -> sain   : {noms[b]}")

    s, c = np.zeros(len(yv)), np.zeros(len(yv))
    choix, auc_rep = Counter(), []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=core.N_REPEATS,
                                   random_state=core.SEED)
    s_rep, r = np.zeros(len(yv)), 0
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        a, b, _ = meilleure_paire(yv[tr], C[tr])
        p = yv[tr].mean()
        score = np.full(len(te), p)
        score[C[te, a]] = 1.0
        score[C[te, b] & ~C[te, a]] = 0.0
        s[te] += score; c[te] += 1; s_rep[te] = score
        choix[(noms[a], noms[b])] += 1
        if (i + 1) % 5 == 0:                    # fin d'une répétition
            auc_rep.append(roc_auc_score(yv, s_rep))
    oof = s / c
    auc_oof = roc_auc_score(yv, oof)
    ci = core.bootstrap_ci(yv, oof)["AUC-ROC"]

    print(f"\n  AUC HORS PLI (scores moyennés sur 10 répétitions) : {auc_oof:.4f} "
          f"[{ci[0]:.2f}-{ci[1]:.2f}]")
    print(f"  AUC par répétition : moyenne {np.mean(auc_rep):.4f}, "
          f"min {np.min(auc_rep):.4f}, max {np.max(auc_rep):.4f}")
    print("\n  Paires retenues dans les 50 plis :")
    for (pa, pb), k in choix.most_common(6):
        print(f"    {k:2d} x  A: {pa}  |  B: {pb}")

    pd.DataFrame([
        {"évaluation": "choisie et notée sur les 89 dossiers", "AUC": round(auc_in, 4)},
        {"évaluation": "choisie dans le pli, notée hors pli (moyenne des scores)",
         "AUC": round(auc_oof, 4), "IC95_bas": round(ci[0], 3), "IC95_haut": round(ci[1], 3)},
        {"évaluation": "hors pli, moyenne des 10 répétitions", "AUC": round(np.mean(auc_rep), 4)},
    ]).to_csv(core.config.OUTPUTS_DIR / "regle_hors_pli.csv", index=False, encoding="utf-8")
    print("\n  Écrit : regle_hors_pli.csv")


if __name__ == "__main__":
    main()
