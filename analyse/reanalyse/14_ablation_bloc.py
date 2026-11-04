# -*- coding: utf-8 -*-
"""
Ablation of the redundant block and stratified analysis (Table 5).

Removes the fields most associated with default one at a time, in groups,
and all together, and estimates the model within crop strata.

Outputs: outputs/ablation_bloc.csv, outputs/stratifie.csv
Usage: python 14_ablation_bloc.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core

SEUIL_CIBLE = 0.70      # au-delà, la variable appartient au bloc suspect
KEYS = ["AUC-ROC", "PR-AUC", "Bal.acc", "Recall(défaut)"]


def trouver(X, motif):
    hits = [c for c in X.columns if motif.lower() in c.lower()]
    assert hits, f"aucune colonne pour {motif!r}"
    return hits


def main():
    df, X, y, roles = core.load_data("matures")
    print(f"[14] Ablation du bloc redondant — n={len(y)}, {X.shape[1]} variables\n")

    # ---- constitution du bloc, à partir de la corrélation variable-cible
    num, cat = core.split_types(X)
    r = {}
    for c in X.columns:
        if c in num:
            v = pd.to_numeric(X[c], errors="coerce")
            r[c] = abs(v.corr(y.astype(float)))
        else:
            d = pd.get_dummies(X[c].astype(str)).astype(float)
            r[c] = float(d.corrwith(y.astype(float)).abs().max())
    r = pd.Series(r).sort_values(ascending=False)
    bloc = [c for c in r.index if r[c] >= SEUIL_CIBLE]
    print(f"  Bloc suspect (|r| à la cible >= {SEUIL_CIBLE}) — {len(bloc)} variables :")
    for c in bloc:
        print(f"     {r[c]:.4f}   {c}")

    # ---- échelle d'ablations
    subsets = {"Modèle complet (référence)": list(X.columns)}
    for c in bloc:                                  # une à une
        subsets[f"Sans « {c[:44]} »"] = [k for k in X.columns if k != c]
    subsets["Sans le bloc entier"] = [k for k in X.columns if k not in bloc]

    rows = []
    for label, cols in subsets.items():
        Xs = X[cols]
        point, ci = core.evaluate(core.logreg_l2(Xs), Xs, y)
        rows.append({"Modèle": label, "n_var": len(cols), **core.fmt(point, ci, KEYS)})
        print(f"  OK {label} ({len(cols)} var) — AUC {point['AUC-ROC']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "ablation_bloc.csv", index=False, encoding="utf-8")
    print("\n=== Ablations (point [IC 95 %]) ===")
    pd.set_option("display.max_columns", None, "display.width", 240)
    print(res[["Modèle", "n_var", "AUC-ROC"]].to_string(index=False))

    # ---- stratification par culture
    cult = trouver(X, "Nature de la culture")[0]
    strate = X[cult].astype(str).str.lower().str.contains("mais|maïs", regex=True)
    print(f"\n=== Stratification par culture ({cult!r}) ===")
    srows = []
    for nom, masque in [("maïs", strate), ("hors maïs", ~strate)]:
        ys, Xs = y[masque], X[masque]
        n, d = len(ys), int(ys.sum())
        if ys.nunique() < 2:
            print(f"  {nom} : n={n}, défauts={d} — UNE SEULE CLASSE, AUC indéfinie")
            srows.append({"strate": nom, "n": n, "défauts": d, "AUC-ROC": "indéfinie (classe unique)"})
            continue
        point, ci = core.evaluate(core.logreg_l2(Xs), Xs, ys)
        print(f"  {nom} : n={n}, défauts={d}, AUC={point['AUC-ROC']:.3f}")
        srows.append({"strate": nom, "n": n, "défauts": d,
                      **core.fmt(point, ci, KEYS)})
    pd.DataFrame(srows).to_csv(core.config.OUTPUTS_DIR / "stratifie.csv",
                               index=False, encoding="utf-8")
    print("\n  Écrit : ablation_bloc.csv · stratifie.csv")


if __name__ == "__main__":
    main()
