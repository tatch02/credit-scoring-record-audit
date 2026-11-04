# -*- coding: utf-8 -*-
"""
Does the model rely on the officers' judgements?

Compares the reference model on all variables, on facts only, and on the
officer's judgements only (market reputation and market context). If
judgements alone match the full model, the model copies the officer's view.

Output: outputs/circularity.csv
Usage: python 08_circularity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

KEYS = ["AUC-ROC", "PR-AUC", "Bal.acc", "Recall(défaut)"]


def main():
    _, X, y, _ = core.load_data("matures")
    judgments = [c for c in X.columns if "oralit" in c.lower() or c == "contexte_ord"]
    facts = [c for c in X.columns if c not in judgments]

    subsets = {
        "Full (reference)": X.columns.tolist(),
        "Facts only (no judgments)": facts,
        "Judgments only (reputation + market context)": judgments,
    }
    print(f"[08] Judgment variables: {judgments}")
    print(f"     Fact variables: {len(facts)}\n")

    rows = []
    for label, cols in subsets.items():
        Xs = X[cols]
        point, ci = core.evaluate(core.logreg_l2(Xs), Xs, y)
        rows.append({"Model": label, "n_var": len(cols), **core.fmt(point, ci, KEYS)})
        print(f"  OK {label} ({len(cols)} var)")

    res = pd.DataFrame(rows)
    res.to_csv(core.config.OUTPUTS_DIR / "circularity.csv", index=False)
    pd.set_option("display.max_columns", None, "display.width", 200)
    print("\n=== Circularity test (point [IC 95 %]) ===")
    print(res[["Model", "n_var"] + KEYS].to_string(index=False))


if __name__ == "__main__":
    main()
