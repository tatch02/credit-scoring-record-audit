# -*- coding: utf-8 -*-
"""
Shared building blocks: data loading, a leakage-free preprocessor (imputation,
scaling and one-hot encoding fitted inside each fold), repeated stratified
cross-validation with out-of-fold probabilities, metrics and bootstrap
intervals.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             balanced_accuracy_score, recall_score, precision_score,
                             accuracy_score, confusion_matrix)

SEED = config.SEED
N_REPEATS = 10
B_BOOT = 1000


def load_data(which="matures"):
    """Retourne (df complet, X modèle, y, roles). which='matures' (89) ou 'all' (100)."""
    path = config.DERIVED_DATA if which == "matures" else config.DERIVED_ALL
    df = pd.read_csv(path)
    roles = json.loads((config.OUTPUTS_DIR / "columns_roles.json").read_text(encoding="utf-8"))
    feats = roles["model_vars"] + roles["miss_indicators"]
    return df, df[feats].copy(), df["target"].astype(int), roles


def split_types(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def make_preprocessor(X):
    """Impute (médiane/mode) + scale + one-hot — tout fitté intra-fold (aucune fuite)."""
    num, cat = split_types(X)
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(min_frequency=5, handle_unknown="infrequent_if_exist",
                                              sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def logreg_l2(X):
    """Modèle de référence : régression logistique L2, C validé intra-fold."""
    Cs = np.logspace(-2, 2, 10)
    return Pipeline([("pre", make_preprocessor(X)),
                     ("clf", LogisticRegressionCV(Cs=Cs, cv=5, penalty="l2", solver="liblinear",
                                                  scoring="neg_log_loss", max_iter=5000, n_jobs=-1, random_state=SEED))])


def oof_proba(model, X, y, n_repeats=N_REPEATS):
    """Probabilité out-of-fold moyennée sur les répétitions (1 valeur par observation)."""
    n = len(y); s, c = np.zeros(n), np.zeros(n)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    for tr, te in rskf.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        s[te] += m.predict_proba(X.iloc[te])[:, 1]; c[te] += 1
    return s / c


def metrics_from(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan
    return {"AUC-ROC": auc, "PR-AUC": average_precision_score(y, proba),
            "Brier": brier_score_loss(y, proba), "Bal.acc": balanced_accuracy_score(y, pred),
            "Recall(défaut)": recall_score(y, pred, pos_label=1, zero_division=0),
            "Spécificité": recall_score(y, pred, pos_label=0, zero_division=0),
            "Précision": precision_score(y, pred, pos_label=1, zero_division=0),
            "Accuracy": accuracy_score(y, pred)}


def bootstrap_ci(y, proba, B=B_BOOT):
    rng = np.random.default_rng(SEED); yv = np.asarray(y); n = len(yv); acc = defaultdict(list)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(yv[idx])) < 2:
            continue
        for k, v in metrics_from(yv[idx], proba[idx]).items():
            if not np.isnan(v):
                acc[k].append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}


def evaluate(model, X, y):
    """Retourne (métriques ponctuelles, IC 95 %)."""
    proba = oof_proba(model, X, y)
    return metrics_from(y, proba), bootstrap_ci(y, proba)


def fmt(point, ci, keys=None):
    keys = keys or list(point)
    out = {}
    for k in keys:
        lo, hi = ci.get(k, (np.nan, np.nan))
        out[k] = f"{point[k]:.2f} [{lo:.2f}–{hi:.2f}]"
    return out
