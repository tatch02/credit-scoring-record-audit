# -*- coding: utf-8 -*-
"""
Internal consistency checks of the reported figures.

Assertions that counts sum correctly, that recall recomputes from the
confusion matrix, that probabilities lie in the unit interval, and that the
class counts match the analysed sample. Stops with a clear message on the first
failure rather than produce a plausible but wrong number.
Usage: python check_integrity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, accuracy_score
sys.stdout.reconfigure(encoding="utf-8")


def main():
    n = 0

    # --- 1. Intégrité des données ---
    _, Xm, ym, roles = core.load_data("matures")
    _, Xa, ya, _ = core.load_data("all")
    assert len(ym) == 89, f"matured n={len(ym)} ≠ 89"; n += 1
    assert len(ya) == 100, f"all n={len(ya)} ≠ 100"; n += 1
    assert len(ya) - len(ym) == 11, "excluded in-progress ≠ 11"; n += 1
    assert set(ym.unique()) <= {0, 1}, "target not binary"; n += 1
    assert ym.isna().sum() == 0, "NaN in target"; n += 1
    assert (ym == 1).sum() + (ym == 0).sum() == len(ym), "default+sound ≠ n"; n += 1
    assert (ym == 1).sum() == 49 and (ym == 0).sum() == 40, "matured class counts off"; n += 1
    assert Xm.shape[0] == len(ym), "X/y length mismatch"; n += 1
    epv = min((ym == 1).sum(), (ym == 0).sum()) / len(roles["model_vars"])
    assert 1.5 < epv < 2.5, f"EPV {epv:.2f} out of expected range"; n += 1

    # --- 2. Sorties du modèle (référence) ---
    proba = core.oof_proba(core.logreg_l2(Xm), Xm, ym)
    assert not np.isnan(proba).any(), "NaN in OOF probabilities"; n += 1
    assert proba.min() >= 0 and proba.max() <= 1, "probability out of [0,1]"; n += 1
    assert len(proba) == len(ym), "one OOF probability per observation expected"; n += 1

    # --- 3. Auto-cohérence des métriques (matrice de confusion ↔ recall/spécificité) ---
    yv, pred = ym.values, (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    assert tn + fp + fn + tp == len(yv), "confusion total ≠ n"; n += 1
    assert abs(recall_score(yv, pred, pos_label=1) - tp / (tp + fn)) < 1e-9, "recall ≠ TP/(TP+FN)"; n += 1
    assert abs(recall_score(yv, pred, pos_label=0) - tn / (tn + fp)) < 1e-9, "specificity ≠ TN/(TN+FP)"; n += 1
    assert abs(accuracy_score(yv, pred) - (tp + tn) / len(yv)) < 1e-9, "accuracy ≠ (TP+TN)/n"; n += 1
    auc = roc_auc_score(yv, proba)
    assert 0.0 <= auc <= 1.0, "AUC out of [0,1]"; n += 1

    print(f"[checks] OK all {n} consistency checks passed "
          f"(confusion: TN={tn} FP={fp} FN={fn} TP={tp}, AUC={auc:.2f})")


if __name__ == "__main__":
    main()
