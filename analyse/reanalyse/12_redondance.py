# -*- coding: utf-8 -*-
"""
Structural redundancy of the analysis table (Table 3, Section 5.2).

  - which variables induce the same partition of the 89 files, i.e. are the
    same content under two names;
  - the correlation matrix among predictors;
  - the contradiction between "active in cultivation, last three seasons"
    and years in the occupation.

Identity is tested on the original variables, not on one-hot columns: two
categories of one variable are anti-correlated by construction. Missing values
are treated as a category of their own. The one-hot encoding here describes
the table; it is not a model evaluation, so no leakage question arises.

Outputs (aggregates only):
  outputs/redondance_variables.csv      groups of identical variables
  outputs/redondance_matrice.csv        correlation among predictors
  outputs/redondance_modalites.csv      identical categories across variables
  outputs/redondance_cible.csv          correlation of each column with the outcome
  outputs/contradiction_anciennete.csv  occupation-tenure cross-tabulation
Usage: python 12_redondance.py
"""
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core

MANQUANT = "@@NA@@"     # les manquants forment une modalité, ils ne sont pas ignorés


def codes(serie):
    """Partition des lignes induite par une variable, manquants compris."""
    s = serie.astype(object).where(serie.notna(), MANQUANT).astype(str)
    return pd.factorize(s)[0]


def meme_partition(a, b):
    """Vrai si a et b découpent les 89 lignes exactement de la même façon.

    Test par bijection : le croisement ne doit avoir qu'une case non nulle par
    ligne et par colonne. Insensible au nommage des modalités (OUI/NON contre
    1/0), ce qui est bien la question posée : le CONTENU est-il le même ?
    """
    ct = pd.crosstab(a, b).values
    return bool((ct > 0).sum(axis=1).max() == 1 and (ct > 0).sum(axis=0).max() == 1)


def encode(X):
    """One-hot descriptif pour la matrice de corrélation."""
    num, cat = core.split_types(X)
    parts = [X[num].apply(pd.to_numeric, errors="coerce")]
    if cat:
        parts.append(pd.get_dummies(X[cat].astype(str), prefix_sep=" = ").astype(float))
    E = pd.concat(parts, axis=1)
    const = [c for c in E.columns if E[c].nunique(dropna=True) <= 1]
    return E.drop(columns=const), const


def trouver(X, motif, libelle):
    hits = [c for c in X.columns if motif.lower() in c.lower()]
    assert len(hits) >= 1, f"colonne introuvable pour {libelle!r} (motif {motif!r})"
    return hits[0]


def main():
    df, X, y, roles = core.load_data("matures")
    print(f"[12] Redondance — n={len(y)} dossiers, {X.shape[1]} variables\n")

    # ------------------------------------------- 1. variables = même colonne ?
    C = {c: codes(X[c]) for c in X.columns}
    n_mod = {c: len(np.unique(C[c])) for c in X.columns}
    paires = [(a, b) for a, b in combinations(X.columns, 2)
              if n_mod[a] > 1 and n_mod[a] == n_mod[b] and meme_partition(C[a], C[b])]

    # regroupement transitif : si a==b et b==c, les trois sont un seul groupe
    groupes = []
    for a, b in paires:
        cible = next((g for g in groupes if a in g or b in g), None)
        if cible is None:
            groupes.append({a, b})
        else:
            cible |= {a, b}

    print(f"  Variables induisant la MÊME partition des 89 dossiers : "
          f"{len(groupes)} groupe(s)")
    lignes = []
    for i, g in enumerate(sorted(groupes, key=len, reverse=True), 1):
        membres = sorted(g)
        na = {c: int(X[c].isna().sum()) for c in membres}
        print(f"\n    Groupe {i} — {len(membres)} variables, "
              f"{n_mod[membres[0]]} modalités :")
        for c in membres:
            print(f"       · {c}   (manquants : {na[c]})")
        for c in membres:
            lignes.append({"groupe": i, "variable": c, "n_modalites": n_mod[c],
                           "n_manquants": na[c], "taille_groupe": len(membres)})
    pd.DataFrame(lignes).to_csv(
        core.config.OUTPUTS_DIR / "redondance_variables.csv", index=False, encoding="utf-8")

    n_redondantes = sum(len(g) - 1 for g in groupes)
    print(f"\n  -> {X.shape[1]} variables déclarées, "
          f"{X.shape[1] - n_redondantes} partitions distinctes "
          f"({n_redondantes} variable(s) sans information propre)")

    # ------------------------------------------- 2. matrice entre prédicteurs
    E, const = encode(X)
    M = E.corr()
    M.to_csv(core.config.OUTPUTS_DIR / "redondance_matrice.csv", encoding="utf-8")
    iu = np.triu_indices_from(M.values, k=1)
    hors_diag = np.abs(M.values[iu])
    n_nan = int(np.isnan(hors_diag).sum())
    print(f"\n  Matrice entre prédicteurs : {E.shape[1]} colonnes encodées "
          f"({len(const)} constantes écartées)")
    print(f"    paires |r| >= 0.90 : {int(np.nansum(hors_diag >= 0.90))} "
          f"sur {len(hors_diag) - n_nan} calculables ({n_nan} indéfinies)")
    print(f"    |r| médian : {np.nanmedian(hors_diag):.3f}")

    # -------------------- 2b. identités au niveau MODALITÉ, entre variables
    # C'est le niveau où « le revenu n'est pas renseigné » se révèle être
    # « la culture est du maïs » : les deux variables n'ont pas le même nombre
    # de modalités, donc le test de partition du §1 ne peut pas les apparier.
    source = {}
    for c in E.columns:
        source[c] = c.split(" = ")[0] if " = " in c else c
    binaires = [c for c in E.columns if E[c].dropna().nunique() == 2]
    jumelles = []
    for a, b in combinations(binaires, 2):
        if source[a] == source[b]:
            continue                      # même variable : artefact d'encodage
        va, vb = E[a], E[b]
        ok = va.notna() & vb.notna()
        if ok.sum() == 0:
            continue
        if bool((va[ok] == vb[ok]).all()):
            jumelles.append({"colonne_A": a, "colonne_B": b, "sens": "identique",
                             "n_compare": int(ok.sum()), "n_incomplet": int((~ok).sum())})
        elif bool((va[ok] != vb[ok]).all()):
            jumelles.append({"colonne_A": a, "colonne_B": b, "sens": "complémentaire",
                             "n_compare": int(ok.sum()), "n_incomplet": int((~ok).sum())})
    J = pd.DataFrame(jumelles)
    J.to_csv(core.config.OUTPUTS_DIR / "redondance_modalites.csv", index=False, encoding="utf-8")
    print(f"\n  Modalités identiques ou complémentaires ENTRE variables distinctes : {len(J)}")
    for _, r in J.iterrows():
        signe = "==" if r.sens == "identique" else "=/="
        print(f"    {r.colonne_A}  {signe}  {r.colonne_B}"
              + (f"   ({r.n_incomplet} ligne(s) incomplète(s))" if r.n_incomplet else ""))

    # ------------------------------------------------------- 3. à la cible
    cible = E.corrwith(y.astype(float)).sort_values(key=np.abs, ascending=False)
    cible.rename("r_avec_cible").to_frame().to_csv(
        core.config.OUTPUTS_DIR / "redondance_cible.csv", encoding="utf-8")
    print(f"\n  Corrélation à la cible — 8 premières colonnes encodées :")
    for nom, v in cible.head(8).items():
        print(f"    {v:+.4f}   {nom}")

    # ---------------------------------------------------- 4. la contradiction
    anc = trouver(X, "nciennet", "ancienneté dans la profession")
    act = trouver(X, "Actif dans culture", "actif 3 saisons")
    an = pd.to_numeric(X[anc], errors="coerce")

    # Disclosure control: a year-by-year cross-tabulation has cells of one
    # file. Only the dichotomised table (no small cell) and summary statistics
    # are written.
    seuil = an >= 3
    ct = pd.crosstab(seuil.map({True: "ancienneté >= 3 ans", False: "ancienneté < 3 ans"}),
                     X[act].astype(str))
    ct.to_csv(core.config.OUTPUTS_DIR / "contradiction_anciennete.csv", encoding="utf-8")
    petites = int(((ct.values > 0) & (ct.values < 5)).sum())
    assert petites == 0, f"{petites} cellule(s) < 5 dans un tableau versionné — revoir l'agrégation"

    non = X[act].astype(str).str.upper().str.startswith("NON")
    n_non, n_non_3ans = int(non.sum()), int((non & (an >= 3)).sum())
    print(f"\n  Contradiction logique :")
    print(f"    {anc!r}  x  {act!r}")
    print(f"    ancienneté : min={an.min():.0f}  médiane={an.median():.0f}  max={an.max():.0f}")
    print(f"    dossiers « NON actif 3 saisons » : {n_non}  "
          f"dont ancienneté >= 3 ans : {n_non_3ans}")
    assert an.min() >= 3, "l'argument suppose une ancienneté minimale de 3 ans"
    assert n_non == n_non_3ans, "la contradiction n'est pas totale — revoir l'énoncé"
    print(f"    -> {n_non}/{n_non_3ans} : le champ n'enregistre pas un fait agricole")

    print("\n  Écrit : redondance_variables.csv · redondance_matrice.csv · "
          "redondance_modalites.csv · redondance_cible.csv · contradiction_anciennete.csv")


if __name__ == "__main__":
    main()
