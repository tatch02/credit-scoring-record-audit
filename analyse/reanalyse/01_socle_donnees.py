# -*- coding: utf-8 -*-
"""
Build the analysis tables from the raw loan records.

Uses datamodel.py for loading, cleaning, encoding and the outcome definition.
Writes two tables with identical columns:
  data/analysis_matures.csv  89 matured loans (main analysis)
  data/analysis_all.csv      100 loans, in-progress loans classified (sensitivity)
and outputs/columns_roles.json, the role of each column (no data).
Usage: python 01_socle_donnees.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import datamodel as dm
sys.stdout.reconfigure(encoding="utf-8")


def main():
    df = dm.clean_and_encode(dm.load_raw())
    tgt_col = dm.find(df, "situation de credit")

    mat, roles = dm.make_table(df, df[tgt_col].map(dm.target_matures))
    allr, _ = dm.make_table(df, df[tgt_col].map(dm.target_all))
    mat.to_csv(config.DERIVED_DATA, index=False, encoding="utf-8")
    allr.to_csv(config.DERIVED_ALL, index=False, encoding="utf-8")
    (config.OUTPUTS_DIR / "columns_roles.json").write_text(
        json.dumps(roles, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[01] matures={len(mat)} {mat['target'].value_counts().to_dict()} -> {config.DERIVED_DATA.name}")
    print(f"     tous   ={len(allr)} {allr['target'].value_counts().to_dict()} -> {config.DERIVED_ALL.name}")
    print(f"     variables modèle : {len(roles['model_vars'])} | EPV(matures) = "
          f"{min(mat['target'].value_counts()) / len(roles['model_vars']):.1f}")


if __name__ == "__main__":
    main()
