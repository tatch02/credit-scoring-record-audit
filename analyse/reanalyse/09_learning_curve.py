# -*- coding: utf-8 -*-
"""
Learning curve, first protocol.

Out-of-fold AUC of the reference model on growing stratified subsamples, to
check whether performance is limited by sample size, together with a
two-variable model (credit history only). Superseded by
15_courbe_apprentissage.py, which holds the test set fixed.

Output: outputs/learning_curve.csv
Usage: python 09_learning_curve.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
sys.stdout.reconfigure(encoding="utf-8")

FRACTIONS = [0.4, 0.55, 0.7, 0.85, 1.0]
N_DRAWS = 8          # tirages par fraction (moyenne pour stabiliser)
SEED = core.config.SEED


def eval_subset(X, y, idx, n_repeats):
    """AUC out-of-fold sur un sous-échantillon."""
    Xs, ys = X.iloc[idx], y.iloc[idx]
    try:
        proba = core.oof_proba(core.logreg_l2(Xs), Xs, ys, n_repeats=n_repeats)
        m = core.metrics_from(ys, proba)
        return m.get("AUC-ROC", np.nan)
    except Exception:
        return np.nan


def main():
    _, X, y, _ = core.load_data("matures")
    # MÊME définition que 03_ablation.py (« Historique de crédit SEUL ») :
    # appréciation du PRÉCÉDENT crédit (ordinale) + drapeau « sans crédit antérieur ».
    # Toutes deux connues AVANT la décision -> prédicteurs légitimes, non circulaires.
    hist = [c for c in X.columns if "apprec_credit_ord" in c or "sans_historique" in c]

    print(f"[09] n total = {len(y)} | fractions = {FRACTIONS} | {N_DRAWS} tirages/fraction\n")

    rows = []
    for frac in FRACTIONS:
        n_sub = int(round(frac * len(y)))
        aucs_full, aucs_hist = [], []
        if frac == 1.0:
            draws = [np.arange(len(y))]
        else:
            sss = StratifiedShuffleSplit(n_splits=N_DRAWS, train_size=n_sub, random_state=SEED)
            draws = [tr for tr, _ in sss.split(X, y)]
        for idx in draws:
            aucs_full.append(eval_subset(X, y, idx, n_repeats=5))
            if hist:
                aucs_hist.append(eval_subset(X[hist], y, idx, n_repeats=5))
        rows.append({
            "fraction": frac,
            "n": n_sub,
            "AUC_full_mean": np.nanmean(aucs_full),
            "AUC_full_sd": np.nanstd(aucs_full),
            "AUC_hist2var_mean": np.nanmean(aucs_hist) if aucs_hist else np.nan,
        })
        print(f"  OK n={n_sub:3d} ({frac:.0%})  AUC complet = {np.nanmean(aucs_full):.3f}"
              f"  | historique seul = {np.nanmean(aucs_hist) if aucs_hist else float('nan'):.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "learning_curve.csv", index=False)
    print("\n=== Courbe d'apprentissage (AUC out-of-fold) ===")
    print(res.to_string(index=False))

    d = res["AUC_full_mean"].iloc[-1] - res["AUC_full_mean"].iloc[0]
    print(f"\nDelta AUC entre {FRACTIONS[0]:.0%} et 100 % des données : {d:+.3f}")
    print("Lecture : un delta faible => la performance plafonne, elle n'est PAS limitée par n.")


if __name__ == "__main__":
    main()
