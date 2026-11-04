# -*- coding: utf-8 -*-
"""
Partition of the 89 files by the two conditions and by crop (Section 5.3).

The 19 files with a declared income are exactly the 19 files not concerning
maize.

Output: outputs/partition.csv
Usage: python 20_partition.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import core


def trouver(X, motif):
    h = [c for c in X.columns if motif.lower() in c.lower()]
    assert len(h) >= 1, motif
    return h[0]


def main():
    df, X, y, roles = core.load_data("matures")
    yv = y.astype(int).values
    apprec = pd.to_numeric(X["apprec_credit_ord"], errors="coerce").values
    miss = pd.to_numeric(X[trouver(X, "miss_Revenu")], errors="coerce").values
    cult = X[trouver(X, "Nature de la culture")].astype(str).str.lower()
    mais = cult.str.contains("mais|maïs", regex=True).values

    A = apprec == 0            # crédit précédent en impayés
    B = miss == 0              # revenu déclaré présent
    C = ~A & ~B
    print(f"[20] Partition de {len(yv)} dossiers\n")

    rows = []
    for lib, m in [("Prior loan appraised as in arrears", A),
                   ("A declared income is recorded", B),
                   ("Neither condition holds", C)]:
        n, d = int(m.sum()), int(yv[m].sum())
        rows.append({"panel": "conditions", "group": lib, "n": n,
                     "defaults": d, "rate": round(d / n, 3) if n else np.nan})
        print(f"  {lib:36s} n={n:3d}  défauts={d:3d}  taux={d/n:.3f}")
    assert int(A.sum()) + int(B.sum()) + int(C.sum()) == len(yv), "partition incomplète"
    assert not (A & B).any(), "les deux conditions se chevauchent"

    print()
    for lib, m in [("Maize", mais), ("Not maize", ~mais)]:
        n, d = int(m.sum()), int(yv[m].sum())
        rows.append({"panel": "crop", "group": lib, "n": n,
                     "defaults": d, "rate": round(d / n, 3) if n else np.nan})
        print(f"  {lib:36s} n={n:3d}  défauts={d:3d}  taux={d/n:.3f}")

    # les deux découpages du groupe de 19 coïncident-ils ?
    coincide = bool((B == ~mais).all())
    print(f"\n  « revenu déclaré » et « hors maïs » désignent-ils les mêmes dossiers ? "
          f"{'OUI' if coincide else 'NON'}")
    assert coincide, "les deux groupes de 19 diffèrent — the two groups differ"

    pd.DataFrame(rows).to_csv(core.config.OUTPUTS_DIR / "partition.csv",
                              index=False, encoding="utf-8")
    print("  Écrit : partition.csv")


if __name__ == "__main__":
    main()
