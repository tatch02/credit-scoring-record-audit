# -*- coding: utf-8 -*-
"""
What a rule with no learning achieves (Table 4).

Two conditions read directly from the record:
  A. previous loan appraised as in arrears  -> default
  B. declared income recorded               -> repaid
scored 1 under A, 0 under B and the sample prevalence otherwise; and a single
condition, "active in cultivation = no", whose confusion matrix is compared
with that of the 24-variable model.

Output: outputs/regle_triviale.csv
Usage: python 13_regle_triviale.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core
from sklearn.metrics import roc_auc_score, confusion_matrix


def trouver(X, motif, libelle):
    hits = [c for c in X.columns if motif.lower() in c.lower()]
    assert len(hits) == 1, f"{libelle!r} : {len(hits)} correspondance(s) pour {motif!r} -> {hits}"
    return hits[0]


def main():
    df, X, y, roles = core.load_data("matures")
    yv = y.astype(int).values
    prev = yv.mean()
    print(f"[13] Règles sans apprentissage — n={len(yv)}, prévalence={prev:.3f}\n")

    apprec = pd.to_numeric(X["apprec_credit_ord"], errors="coerce")
    miss = pd.to_numeric(X[trouver(X, "miss_Revenu", "manque du revenu")], errors="coerce")
    act = X[trouver(X, "Actif dans culture", "actif 3 saisons")].astype(str).str.upper()

    A = (apprec == 0).values            # « Impayés » sur le crédit précédent
    B = (miss == 0).values              # revenu déclaré présent
    print(f"  Règle A « crédit précédent = Impayés » : n={A.sum()}, "
          f"taux de défaut={yv[A].mean():.3f}")
    print(f"  Règle B « revenu déclaré présent »     : n={B.sum()}, "
          f"taux de défaut={yv[B].mean():.3f}")
    print(f"  Chevauchement A∩B : {int((A & B).sum())}  ·  "
          f"couverture A∪B : {int((A | B).sum())}/{len(yv)}")
    assert (A & B).sum() == 0, "les deux règles se chevauchent — the two conditions overlap"

    # score à deux conditions : aucun paramètre estimé
    score2 = np.full(len(yv), prev, dtype=float)
    score2[A] = 1.0
    score2[B] = 0.0
    auc2 = roc_auc_score(yv, score2)

    # règle à une condition, en score et en décision
    une = (act.str.startswith("NON")).values
    auc1 = roc_auc_score(yv, une.astype(float))
    tn, fp, fn, tp = confusion_matrix(yv, une.astype(int)).ravel()

    print(f"\n  AUC de la règle à DEUX conditions : {auc2:.4f}")
    print(f"  AUC de la règle à UNE condition   : {auc1:.4f}")
    print(f"  Matrice de la règle à une condition : TN={tn} FP={fp} FN={fn} TP={tp}")

    # référence : le modèle complet, protocole inchangé
    print("\n  Référence : modèle à 24 variables (LogReg L2, CV imbriquée répétée)...")
    proba = core.oof_proba(core.logreg_l2(X), X, y)
    aucm = roc_auc_score(yv, proba)
    m = confusion_matrix(yv, (proba >= 0.5).astype(int)).ravel()
    print(f"    AUC={aucm:.4f}   TN={m[0]} FP={m[1]} FN={m[2]} TP={m[3]}")

    if (tn, fp, fn, tp) == tuple(int(v) for v in m):
        print("    -> la règle à UNE condition reproduit EXACTEMENT la matrice du modèle")

    rows = [
        {"prédicteur": "Règle à deux conditions (aucun apprentissage)", "n_param": 0,
         "AUC-ROC": round(auc2, 4), "TN": "", "FP": "", "FN": "", "TP": ""},
        {"prédicteur": "Règle à une condition (« actif 3 saisons = NON »)", "n_param": 0,
         "AUC-ROC": round(auc1, 4), "TN": tn, "FP": fp, "FN": fn, "TP": tp},
        {"prédicteur": "Modèle à 24 variables (LogReg L2, CV imbriquée)", "n_param": 36,
         "AUC-ROC": round(aucm, 4), "TN": m[0], "FP": m[1], "FN": m[2], "TP": m[3]},
    ]
    pd.DataFrame(rows).to_csv(core.config.OUTPUTS_DIR / "regle_triviale.csv",
                              index=False, encoding="utf-8")
    print("\n  Écrit : regle_triviale.csv")


if __name__ == "__main__":
    main()
