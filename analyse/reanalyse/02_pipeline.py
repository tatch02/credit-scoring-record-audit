# -*- coding: utf-8 -*-
"""
Leakage-free evaluation of the six model specifications (Table 2).

All preprocessing is fitted inside the training folds. Repeated stratified
5-fold cross-validation (10 repetitions), inner 5-fold tuning for the ridge
and lasso models, out-of-fold probabilities averaged across repetitions,
percentile bootstrap intervals. Also records lasso selection frequencies.

Outputs: outputs/results_pipeline.csv, outputs/selection_frequency.csv
Usage: python 02_pipeline.py
"""
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

KEYS = ["AUC-ROC", "PR-AUC", "Brier", "Bal.acc", "Recall(défaut)", "Spécificité"]


def build_models(X):
    pre = core.make_preprocessor(X)
    Cs = np.logspace(-2, 2, 10)
    return {
        "Baseline majoritaire": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "Scorecard (LogReg simple)": Pipeline([("pre", clone(pre)), ("clf", LogisticRegression(max_iter=5000, random_state=core.SEED))]),
        "LogReg L2 (C validé)": Pipeline([("pre", clone(pre)),
            ("clf", LogisticRegressionCV(Cs=Cs, cv=5, penalty="l2", solver="liblinear",
                                         scoring="neg_log_loss", max_iter=5000, n_jobs=-1, random_state=core.SEED))]),
        "LogReg L1 / LASSO": Pipeline([("pre", clone(pre)),
            ("clf", LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear",
                                         scoring="neg_log_loss", max_iter=5000, n_jobs=-1, random_state=core.SEED))]),
        "Random Forest": Pipeline([("pre", clone(pre)),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=core.SEED, n_jobs=-1))]),
        "SVM RBF": Pipeline([("pre", clone(pre)), ("clf", SVC(kernel="rbf", probability=True, random_state=core.SEED))]),
    }


def selection_frequency(X, y, n_repeats=5):
    """Fréquence de sélection (coef LASSO ≠ 0) par variable transformée, sur les plis."""
    counts, total = Counter(), 0
    Cs = np.logspace(-2, 2, 10)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=core.SEED)
    for tr, _ in rskf.split(X, y):
        pipe = Pipeline([("pre", core.make_preprocessor(X)),
                         ("lasso", LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear",
                                                        scoring="neg_log_loss", max_iter=5000, n_jobs=-1, random_state=core.SEED))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        names = pipe.named_steps["pre"].get_feature_names_out()
        coef = pipe.named_steps["lasso"].coef_.ravel()
        for nm in names[coef != 0]:
            counts[nm] += 1
        total += 1
    return pd.Series({k: v / total for k, v in counts.items()}).sort_values(ascending=False)


def main():
    _, X, y, _ = core.load_data()
    num, cat = core.split_types(X)
    print(f"[02] n={len(y)} | défaut/sain={y.value_counts().to_dict()} | "
          f"{len(num)} num + {len(cat)} cat | EPV={min(y.value_counts())/(len(num)+len(cat)):.1f}\n")

    rows = []
    for name, model in build_models(X).items():
        point, ci = core.evaluate(model, X, y)
        rows.append({"Modèle": name, **core.fmt(point, ci)})
        print(f"  OK {name}")
    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "results_pipeline.csv", index=False, encoding="utf-8")

    freq = selection_frequency(X, y)
    freq.to_csv(core.config.OUTPUTS_DIR / "selection_frequency.csv", encoding="utf-8", header=["freq"])

    pd.set_option("display.max_columns", None, "display.width", 200)
    print("\n=== Résultats (point [IC 95 %]) ===")
    print(res[["Modèle"] + KEYS].to_string(index=False))
    print("\n=== Fréquence de sélection LASSO (top 10) ===")
    print(freq.head(10).to_string())


if __name__ == "__main__":
    main()
