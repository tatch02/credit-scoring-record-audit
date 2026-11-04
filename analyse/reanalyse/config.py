# -*- coding: utf-8 -*-
"""
Central configuration: paths and random seed.

No real data is ever versioned. The confidential tables live outside the
repository; when they are absent, the scripts use the synthetic dataset.
"""
from pathlib import Path

# Racine du dépôt : analyse/reanalyse/config.py -> remonte de 2 niveaux
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Données confidentielles (hors versionnement) ---
RAW_DATA = PROJECT_ROOT / "sources" / "Dataset_Utilisable.xlsx"
DERIVED_DIR = PROJECT_ROOT / "data"
DERIVED_DATA = DERIVED_DIR / "analysis_matures.csv"   # 89 prêts matures (analyse principale)
DERIVED_ALL = DERIVED_DIR / "analysis_all.csv"        # 100 prêts, en cours classés (sensibilité)

# --- Public release -----------------------------------------------------------
# The confidential tables never leave the authors' machines. When they are
# absent, the scripts read the synthetic dataset instead, which reproduces the
# schema but no borrower, so results fall to chance (see README.md).
SYNTHETIC_DATA = PROJECT_ROOT / "analyse" / "reanalyse" / "synthetic" / "analysis_matures.csv"
USING_SYNTHETIC = not DERIVED_DATA.exists()
if USING_SYNTHETIC:
    DERIVED_DATA = SYNTHETIC_DATA

# --- Sorties non sensibles (versionnées) ---
OUTPUTS_DIR = PROJECT_ROOT / "analyse" / "reanalyse" / "outputs"

# --- Reproductibilité ---
SEED = 42

# Création des dossiers si absents
DERIVED_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
