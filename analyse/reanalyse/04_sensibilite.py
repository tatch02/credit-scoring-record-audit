# -*- coding: utf-8 -*-
"""
Sensitivity of the reference model to population and variables (Table S3).

  - reference: 89 matured loans
  - adding the granted amount
  - reinstating the 11 in-progress loans (100 loans)

Output: outputs/sensibilite.csv
Usage: python 04_sensibilite.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

KEYS = ["AUC-ROC", "PR-AUC", "Bal.acc", "Recall(défaut)", "Spécificité"]


def main():
    rows = []

    df, X, y, roles = core.load_data("matures")
    p, ci = core.evaluate(core.logreg_l2(X), X, y)
    rows.append({"Analyse": "Référence (89 matures)", "n": len(y), **core.fmt(p, ci, KEYS)})

    Xa = X.copy()
    for c in roles["sensitivity"]:               # + montant accordé
        Xa[c] = df[c].values
    p, ci = core.evaluate(core.logreg_l2(Xa), Xa, y)
    rows.append({"Analyse": "+ montant accordé (89)", "n": len(y), **core.fmt(p, ci, KEYS)})

    _, X2, y2, _ = core.load_data("all")          # + 11 en cours
    p, ci = core.evaluate(core.logreg_l2(X2), X2, y2)
    rows.append({"Analyse": "+ 11 en cours (100)", "n": len(y2), **core.fmt(p, ci, KEYS)})

    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "sensibilite.csv", index=False, encoding="utf-8")
    pd.set_option("display.max_columns", None, "display.width", 220)
    print("=== Sensibilités (point [IC 95 %]) ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
