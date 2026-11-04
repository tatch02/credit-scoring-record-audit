# -*- coding: utf-8 -*-
"""
Lasso selection frequencies and coefficients (Table S8).

Stratified 5-fold cross-validation repeated 5 times (25 folds). In each fold,
an L1-penalised logistic regression with the penalty tuned by inner
cross-validation; records which encoded columns are retained and the mean
standardised coefficient (zero when not retained).

Output: outputs/lasso_report.csv
Usage: python 07_lasso_coefficients.py
"""
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
sys.stdout.reconfigure(encoding="utf-8")


def main():
    _, X, y, _ = core.load_data("matures")
    Cs = np.logspace(-2, 2, 10)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=core.SEED)
    coef_sum, nonzero = defaultdict(float), defaultdict(int)
    lambdas, total = [], 0
    for tr, _ in rskf.split(X, y):
        pipe = Pipeline([("pre", core.make_preprocessor(X)),
                         ("lasso", LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear",
                                                        scoring="neg_log_loss", max_iter=5000, n_jobs=-1, random_state=core.SEED))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        names = pipe.named_steps["pre"].get_feature_names_out()
        coef = pipe.named_steps["lasso"].coef_.ravel()
        lambdas.append(1.0 / pipe.named_steps["lasso"].C_[0])
        for nm, cf in zip(names, coef):
            coef_sum[nm] += cf
            nonzero[nm] += int(cf != 0)
        total += 1

    df = pd.DataFrame([{"feature": nm, "selection_freq": round(nonzero[nm] / total, 3),
                        "mean_coef_std_scale": round(coef_sum[nm] / total, 3)} for nm in coef_sum])
    df = df.sort_values("selection_freq", ascending=False)
    df.to_csv(core.config.OUTPUTS_DIR / "lasso_report.csv", index=False, encoding="utf-8")

    print(f"[07] LASSO report ({total} folds) -> lasso_report.csv")
    print(f"     mean λ = 1/C : {np.mean(lambdas):.3f} (median C: {np.median([1/l for l in lambdas]):.2f})")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
