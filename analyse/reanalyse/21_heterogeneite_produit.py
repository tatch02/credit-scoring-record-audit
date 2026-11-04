# -*- coding: utf-8 -*-
"""
Product heterogeneity: does loan term explain the result? (Section 5.5)

If discrimination came from separating one-month loans from seasonal ones, it
would collapse within strata of loan term. The reference model is refitted
within each stratum under the same protocol.

Output: outputs/heterogeneite_produit.csv
Usage: python 21_heterogeneite_produit.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core

KEYS = ["AUC-ROC", "Bal.acc", "Recall(défaut)"]


def trouver(X, motif):
    h = [c for c in X.columns if motif.lower() in c.lower()]
    assert h, motif
    return h[0]


def main():
    df, X, y, roles = core.load_data("matures")
    yv = y.astype(int).values
    dur = pd.to_numeric(X[trouver(X, "ur")], errors="coerce") if False else None

    # la selection par motif large mordait sur « Mouvements confies sur les 12
    # derniers MOIS » : on exige explicitement la duree du pret.
    col_dur = [c for c in X.columns if c.lower().startswith(("dur", "duree", "durée"))]
    print(f"[21] Colonnes de durée candidates : {col_dur}")
    assert col_dur, "aucune colonne de durée trouvée"
    d = pd.to_numeric(X[col_dur[0]], errors="coerce")
    print(f"     retenue : {col_dur[0]!r}")
    print(f"     distribution : {d.value_counts().sort_index().to_dict()}\n")

    act = X[trouver(X, "Actif dans culture")].astype(str).str.upper()
    court = (d <= 1).values
    print(f"  Prêts d'un mois ou moins : {int(court.sum())} sur {len(yv)}")
    print(f"    taux de défaut court : {yv[court].mean():.3f}"
          f"  · long : {yv[~court].mean():.3f}")
    n_non_court = int((court & act.str.startswith("NON").values).sum())
    print(f"    dont « non actif »   : {n_non_court}/{int(court.sum())}")

    rows = []
    for lib, m in [("Tous", np.ones(len(yv), bool)),
                   ("Durée <= 1 mois", court),
                   ("Durée >= 2 mois", ~court)]:
        ys, Xs = y[m], X[m]
        n, dd = len(ys), int(ys.sum())
        if ys.nunique() < 2:
            print(f"  {lib:18s} n={n:3d} défauts={dd:3d}  CLASSE UNIQUE")
            rows.append({"strate": lib, "n": n, "defauts": dd, "AUC-ROC": "classe unique"})
            continue
        point, ci = core.evaluate(core.logreg_l2(Xs), Xs, ys)
        print(f"  {lib:18s} n={n:3d} défauts={dd:3d}  AUC={point['AUC-ROC']:.3f}")
        rows.append({"strate": lib, "n": n, "defauts": dd, **core.fmt(point, ci, KEYS)})

    # la durée explique-t-elle à elle seule ?
    from sklearn.metrics import roc_auc_score
    auc_dur = roc_auc_score(yv, court.astype(float))
    print(f"\n  AUC de la SEULE règle « durée <= 1 mois » : {auc_dur:.4f}")

    # et « actif en culture » est-il un indicateur de produit ?
    ct = pd.crosstab(court, act.str.startswith("NON"))
    print(f"\n  croisement durée courte x « non actif » :\n{ct.to_string()}")
    accord = float((court == act.str.startswith("NON").values).mean())
    print(f"  accord entre les deux : {accord:.3f}")

    pd.DataFrame(rows).to_csv(core.config.OUTPUTS_DIR / "heterogeneite_produit.csv",
                              index=False, encoding="utf-8")
    print("\n  Écrit : heterogeneite_produit.csv")


if __name__ == "__main__":
    main()
