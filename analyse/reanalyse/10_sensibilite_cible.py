# -*- coding: utf-8 -*-
"""
Sensitivity to the outcome definition (Table S3, panel A).

Seven loans are recorded as "fully repaid with difficulty". Three codings are
compared on the same 89 loans:
  A. reference     difficulty = repaid    (49 defaults / 40 repaid)
  B. strict        difficulty = default   (56 / 33)
  C. conservative  difficulty excluded    (49 / 33, n = 82)

Output: outputs/sensibilite_cible.csv
Usage: python 10_sensibilite_cible.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import core
import datamodel as dm
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

KEYS = ["AUC-ROC", "PR-AUC", "Bal.acc", "Recall(défaut)", "Spécificité"]

# Note: Le motif DOIT inclure « avec » : « remboursé SANS difficultés » contient
# aussi « difficult ». Un motif trop large attrape les 40 remboursés au lieu
# des 7 ambigus (bug attrapé par l'assertion ci-dessous — ne pas relâcher).
DIFFICULTY = "avec difficult"   # « Totalement remboursé avec difficultés »


def target_strict(v):
    """« Remboursé avec difficultés » compte comme DÉFAUT."""
    s = str(v).strip().lower()
    if s.startswith("en cours"):
        return np.nan
    if "souffrance" in s:
        return 1
    if "rembours" in s:
        return 1 if DIFFICULTY in s else 0
    return np.nan


def target_drop_ambiguous(v):
    """« Remboursé avec difficultés » exclu de l'analyse."""
    s = str(v).strip().lower()
    if s.startswith("en cours"):
        return np.nan
    if "souffrance" in s:
        return 1
    if "rembours" in s:
        return np.nan if DIFFICULTY in s else 0
    return np.nan


def main():
    df = dm.clean_and_encode(dm.load_raw())
    tgt_col = dm.find(df, "situation de credit")

    # contrôle : le motif attrape bien les 7 cas attendus
    counts = df[tgt_col].astype(str).str.strip().value_counts()
    print("[10] Modalités de la cible brute :")
    for k, v in counts.items():
        print(f"     {v:3d}  {k}")
    n_diff = int(df[tgt_col].astype(str).str.lower().str.contains(DIFFICULTY).sum())
    print(f"\n     « avec difficultés » détectés : {n_diff}")
    assert n_diff == 7, f"attendu 7 cas ambigus, trouvé {n_diff} — vérifier le motif"

    specs = {
        "A. Reference (difficulty = sound)":   dm.target_matures,
        "B. Strict (difficulty = default)":    target_strict,
        "C. Conservative (difficulty dropped)": target_drop_ambiguous,
    }

    rows = []
    for label, fn in specs.items():
        tbl, roles = dm.make_table(df, df[tgt_col].map(fn))
        y = tbl["target"].astype(int)
        X = tbl[roles["model_vars"] + roles["miss_indicators"]]
        point, ci = core.evaluate(core.logreg_l2(X), X, y)
        rows.append({
            "Target definition": label,
            "n": len(y),
            "defaults": int(y.sum()),
            "sound": int((1 - y).sum()),
            **core.fmt(point, ci, KEYS),
        })
        print(f"  OK {label}  (n={len(y)}, {int(y.sum())} défauts / {int((1-y).sum())} sains)")

    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUTS_DIR / "sensibilite_cible.csv", index=False, encoding="utf-8")
    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n=== Sensibilité à la définition de la cible (point [IC 95 %]) ===")
    print(res[["Target definition", "n", "defaults", "sound"] + KEYS].to_string(index=False))

    aucs = [float(str(a).split(" ")[0]) for a in res["AUC-ROC"]]
    print(f"\nÉcart max d'AUC entre définitions : {max(aucs) - min(aucs):.3f}")
    print("Lecture : un écart faible => le résultat est robuste au codage de la cible.")


if __name__ == "__main__":
    main()
