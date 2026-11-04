# -*- coding: utf-8 -*-
"""
Generate the synthetic dataset released with the code.

Each column is drawn independently from its own distribution, which destroys
the joint structure: no synthetic row corresponds to a borrower. Numeric
values are drawn from the 5th to 95th percentiles only; categories seen fewer
than five times are pooled into "other". The three pairs of identical fields
are restored after drawing so that the redundancy check has something to find.
The script checks that no real row is reproduced. It needs the confidential
records and is included to document how the synthetic file was made.

Output: analyse/reanalyse/synthetic/analysis_matures.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config

SEED = config.SEED
MIN_MODALITE = 5          # en deçà, la modalité est fondue dans « other »
DEST = config.PROJECT_ROOT / "analyse" / "reanalyse" / "synthetic"

# Groupes établis identiques par 12_redondance.py : à recopier après tirage.
JUMELLES = [
    ("Actif dans culture pendant les 03 dernières saisons ?", "Justificatifs des activités ?"),
    ("miss_Revenu_glo", "miss_Taux_d_end"),
    ("Engagement Chez Confrère", "Traite chez confrère"),
]


def marginale_cat(s, rng, n):
    """Tire n valeurs de la distribution des modalités, rares fondues."""
    vc = s.astype(object).where(s.notna(), "@@NA@@").astype(str).value_counts()
    rares = vc[vc < MIN_MODALITE].index.tolist()
    vc = vc.drop(rares)
    if rares:
        vc["other"] = sum(s.astype(str).isin(rares))
    p = (vc / vc.sum()).values
    tirage = rng.choice(vc.index.to_numpy(), size=n, p=p)
    return pd.Series(tirage).replace("@@NA@@", np.nan), len(rares)


def marginale_num(s, rng, n):
    """Tire n valeurs par échantillonnage des quantiles : la forme est
    conservée, aucune valeur individuelle extrême n'est reproduite."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return pd.Series([np.nan] * n), 0
    q = np.quantile(v, np.linspace(0.05, 0.95, 19))     # bornes exclues
    tir = rng.choice(q, size=n) + rng.normal(0, (q.max() - q.min()) / 60 or 1, n)
    if (v % 1 == 0).all():
        tir = np.round(tir).astype(float)
    tir = np.clip(tir, q.min(), q.max())
    manquants = float(s.isna().mean())
    masque = rng.random(n) < manquants
    out = pd.Series(tir)
    out[masque] = np.nan
    return out, int(masque.sum())


def main():
    if config.USING_SYNTHETIC:
        sys.exit("make_synthetic.py needs the confidential records, which are not "
                 "part of this release; the synthetic file it produced is included.")
    rng = np.random.default_rng(SEED)
    src = pd.read_csv(config.DERIVED_DATA)
    n = len(src)
    print(f"[synthetic] source : {n} lignes, {src.shape[1]} colonnes "
          f"(lue localement, jamais copiée)\n")

    out, n_rares, n_na = {}, 0, 0
    for col in src.columns:
        s = src[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 6:
            out[col], k = marginale_num(s, rng, n)
            n_na += k
        else:
            out[col], k = marginale_cat(s, rng, n)
            n_rares += k
    syn = pd.DataFrame(out)

    # recopier les colonnes établies identiques
    recopiees = 0
    for a, b in JUMELLES:
        if a in syn.columns and b in syn.columns:
            syn[b] = syn[a].values
            recopiees += 1

    DEST.mkdir(parents=True, exist_ok=True)
    chemin = DEST / "analysis_matures.csv"
    syn.to_csv(chemin, index=False, encoding="utf-8")

    print(f"  modalités rares fondues dans « other » : {n_rares}")
    print(f"  paires identiques restaurées après tirage : {recopiees}/{len(JUMELLES)}")
    print(f"  taux de défaut synthétique : {syn['target'].astype(float).mean():.3f} "
          f"(réel : {src['target'].astype(float).mean():.3f})")

    # --- contrôle de non-divulgation : aucune ligne réelle ne doit survivre
    cles = [c for c in src.columns if src[c].nunique() > 3][:8]
    a = set(map(tuple, src[cles].astype(str).values))
    b = set(map(tuple, syn[cles].astype(str).values))
    communes = a & b
    print(f"\n  Contrôle : lignes réelles reproduites à l'identique "
          f"(sur {len(cles)} colonnes discriminantes) : {len(communes)}")
    assert not communes, "une ligne réelle a été reproduite — générateur à revoir"
    print(f"  Écrit : {chemin.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
