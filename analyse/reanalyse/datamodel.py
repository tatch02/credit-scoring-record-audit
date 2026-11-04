# -*- coding: utf-8 -*-
"""
Loading and preparation of the raw records, shared by 00_eda.py and
01_socle_donnees.py so that both see exactly the same cleaning, encoding and
outcome definition.
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import pandas as pd
import numpy as np


def load_raw():
    if not config.RAW_DATA.exists():
        raise FileNotFoundError(
            f"Données brutes introuvables : {config.RAW_DATA}\n"
            "-> placez Dataset_Utilisable.xlsx dans sources/ (données réelles, hors dépôt)."
        )
    df = pd.read_excel(config.RAW_DATA)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def find(df, sub):
    sub = sub.lower()
    return next((c for c in df.columns if sub in c.lower()), None)


def target_matures(v):
    """Défaut/sain sur prêts MATURES ; exclut les 'en cours' (censure) -> NaN."""
    s = str(v).strip().lower()
    if s.startswith("en cours"):
        return np.nan
    if "souffrance" in s:
        return 1
    if "rembours" in s:
        return 0
    return np.nan


def target_all(v):
    """Comme ci-dessus mais classe les 'en cours' (sensibilité) : en souffrance=1, sain=0."""
    s = str(v).strip().lower()
    if "souffrance" in s:
        return 1
    if "rembours" in s or "sains" in s:
        return 0
    return np.nan


def clean_and_encode(df):
    """Nettoyage complet des modalités + ré-encodages (arbitrages validés)."""
    act = find(df, "autre activité")
    df[act] = df[act].replace({"Salarié": "salarié", "RAS": "aucune", "commercante": "Commerce"})
    cult = find(df, "nature de la culture")
    df[cult] = df[cult].replace({"maîs": "mais", "Maïs": "mais", "maïs": "mais"})
    mor = find(df, "moralité")
    df[mor] = df[mor].replace({"Mavaise": "Mauvaise"})
    df["superficie_ha"] = df[find(df, "superficie")].astype(str).str.extract(r"(\d+)").astype(float)
    df["contexte_ord"] = df[find(df, "contexte du marché")].map({"Défavorable": 0, "Moyen": 1, "Favorable": 2})
    apc = find(df, "appréciation du précédent")
    df["apprec_credit_ord"] = df[apc].map(
        {"Impayés": 0, "Remboursé avec difficultés": 1, "Remboursé sans difficultés": 2})
    df["sans_historique_credit"] = (df[apc] == "RAS").astype(int)  # RAS = client sans crédit antérieur
    return df


def make_table(df, target_series):
    """Table d'analyse (colonnes = modèle + manque + cible + hold-out + sensibilité)."""
    d = df.copy()
    d["target"] = target_series
    d = d[d["target"].notna()].copy()
    d["target"] = d["target"].astype(int)
    for c in [find(df, "revenu global"), find(df, "taux d'endettement")]:
        d["miss_" + re.sub(r"\W+", "_", c)[:10]] = d[c].isna().astype(int)

    constants = [find(df, "propriétaire de la surface"), find(df, "justificatif de propriété"),
                 find(df, "proximité"), find(df, "appréciation comité agence"),
                 find(df, "appréciation comité direction"), find(df, "décision comité crédit")]
    drop = set(filter(None, [
        find(df, "n°"), find(df, "montant du prêt accordé"), find(df, "sexe"),
        find(df, "situation matrimoniale"), find(df, "revenu global"), find(df, "taux d'endettement"),
        find(df, "situation de credit"), "target", find(df, "superficie"),
        find(df, "contexte du marché"), find(df, "appréciation du précédent")] + constants))
    model_vars = [c for c in d.columns if c not in drop and not c.startswith("miss_")]
    miss_vars = [c for c in d.columns if c.startswith("miss_")]
    holdout = [c for c in [find(df, "sexe"), find(df, "situation matrimoniale")] if c]
    sensitivity = [c for c in [find(df, "montant du prêt accordé")] if c]
    keep = model_vars + miss_vars + ["target"] + holdout + sensitivity
    roles = {"target": "target", "model_vars": model_vars, "miss_indicators": miss_vars,
             "holdout_equity": holdout, "sensitivity": sensitivity}
    return d[keep], roles
