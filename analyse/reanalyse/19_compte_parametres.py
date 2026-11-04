# -*- coding: utf-8 -*-
"""
How many parameters does the model estimate? (Section 4.2)

Counts the encoded columns produced by the preprocessing actually used in the
pipeline. Rare categories are pooled per fold, so the count is reported on the
full sample and across folds.

Output: outputs/parametres_encodes.csv
Usage: python 19_compte_parametres.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core
from sklearn.model_selection import RepeatedStratifiedKFold


def main():
    df, X, y, roles = core.load_data("matures")
    print(f"[19] Paramètres encodés — n={len(y)}, {X.shape[1]} variables déclarées\n")

    pre = core.make_preprocessor(X)
    Z = pre.fit_transform(X)
    n_col = Z.shape[1]
    print(f"  Colonnes produites par le préprocesseur (échantillon complet) : {n_col}")
    print(f"  + 1 constante (l'ordonnée à l'origine)")
    print(f"  = {n_col + 1} paramètres estimés\n")

    num, cat = core.split_types(X)
    print(f"  dont numériques : {len(num)}")
    print(f"       catégorielles : {len(cat)} variables -> {n_col - len(num)} colonnes")

    # stabilité entre plis : min_frequency dépend de l'échantillon d'entraînement
    tailles = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=core.SEED)
    for tr, _ in rskf.split(X, y):
        p = core.make_preprocessor(X)
        tailles.append(p.fit_transform(X.iloc[tr]).shape[1])
    tailles = np.array(tailles)
    print(f"\n  Entre plis d'entraînement : min={tailles.min()}, "
          f"médiane={int(np.median(tailles))}, max={tailles.max()}")

    # ATTENTION : y.sum() compte la classe POSITIVE (le défaut), qui est ici la
    # classe MAJORITAIRE — 49 défauts contre 40 sains. La convention EPV retient
    # le plus petit des deux groupes, donc 40. Confondre les deux fait passer
    # l'EPV de 1,1 à 1,3 et affaiblit à tort notre propre argument.
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    evenements = min(n_pos, n_neg)
    print(f"\n  Classe positive (défaut) : {n_pos} · classe négative (sain) : {n_neg}")
    print(f"  Événements au sens de l'EPV (le plus petit groupe) : {evenements}")
    for lib, d in [("par paramètre estimé", n_col + 1),
                   ("par colonne encodée", n_col),
                   ("par variable déclarée", X.shape[1])]:
        print(f"    EPV {lib:24s} = {evenements}/{d} = {evenements / d:.2f}")

    pd.DataFrame([{
        "n_variables_declarees": X.shape[1],
        "n_colonnes_encodees": n_col,
        "n_parametres_estimes": n_col + 1,
        "n_evenements": evenements,
        "EPV_par_parametre": round(evenements / (n_col + 1), 3),
        "colonnes_min_entre_plis": int(tailles.min()),
        "colonnes_max_entre_plis": int(tailles.max()),
    }]).to_csv(core.config.OUTPUTS_DIR / "parametres_encodes.csv",
               index=False, encoding="utf-8")
    print("\n  Écrit : parametres_encodes.csv")


if __name__ == "__main__":
    main()
