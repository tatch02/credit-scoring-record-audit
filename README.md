# Code and synthetic data for "Sound methods, unusable results"

Analysis code and a synthetic dataset accompanying the article *Sound methods,
unusable results: a data-collection framework for machine-learning credit
scoring in Cameroonian agricultural microfinance* (Tatchou Nkouindja, Fendji,
Thron, Murena and Atemkeng), submitted to *Data & Policy*.

## The study in brief

On 89 matured agricultural loans, five classifiers evaluated by repeated
cross-validation (all preprocessing learned inside training folds) reach an
AUC of 0.94 to 0.98. Two conditions do as well: chosen within each training
fold, they reach an out-of-fold AUC of 0.97. The strongest fields overlap, and
the sample omits refused applicants, every date and the internal scale that
sets loan ceilings. The high figure therefore describes how the record was
produced, not a borrower risk a lender could act on. The article proposes a
nine-dimension data-collection framework stating which data each lending task
requires.

## Repository layout

```
analyse/reanalyse/
  config.py, core.py, datamodel.py   shared paths, helpers and variable roles
  00_eda.py ... 24_*.py              one script per table, figure or check
  check_integrity.py                 consistency checks of reported figures
  make_synthetic.py                  generator of the synthetic dataset
  synthetic/analysis_matures.csv     synthetic data (89 rows)
  outputs/                           results written by the scripts
requirements.txt                     pinned dependencies
CITATION.cff                         citation metadata
```

## What is and is not here

The study analyses 100 agricultural loan files from a Cameroonian microfinance
institution. **Those records contain personal data of identifiable borrowers,
are subject to banking confidentiality, and are not part of this repository.**
Neither are any tables computed from them.

This repository contains:

- the complete analysis code, exactly as used for the article (comments are in
  French; the article and this file are in English);
- a **synthetic dataset** (`analyse/reanalyse/synthetic/analysis_matures.csv`,
  89 rows) that reproduces the schema, variable types, category frequencies and
  missingness of the analysed table, but no borrower. Each column was drawn
  independently from its distribution, numeric values from the 5th to 95th
  percentiles only, and categories seen fewer than five times were pooled
  into `other`. The three pairs of identical fields reported in the article
  were restored after drawing, so that the redundancy check has something to
  find. No synthetic row matches a real one. One column label was neutralised
  ("Ancienneté à l'institution") because the original could hint at the
  institution.

Because the joint structure is destroyed, models trained on the synthetic data
perform at chance (AUC = 0.43 for the reference model), as intended. The
synthetic data let a reader run and inspect the code; they cannot reproduce
the article's results.

## Running the code

Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # or .venv/bin/pip
cd analyse/reanalyse
python 02_pipeline.py
```

When the confidential tables are absent, `config.py` points every script to
the synthetic dataset. Random seeds are fixed (`SEED = 42`). Each script writes
its results to `analyse/reanalyse/outputs/`.

## Scripts

Status = what happens when the script is run on the synthetic data in this
repository (tested on 2026-11-04). Several scripts contain assertions that check a
figure of the real data (for instance that 45 files with prior arrears all
defaulted); on the synthetic data they stop there, which is the intended
behaviour of those safeguards.

| Script | Purpose | Status on synthetic data |
|---|---|---|
| `00_eda.py` | Descriptive statistics (Table 1) | needs the confidential records |
| `01_socle_donnees.py` | Builds the analysis tables from the confidential records | needs the confidential records |
| `02_pipeline.py` | Repeated (nested where tuned) cross-validation of the six specifications (Table 2) | runs |
| `03_ablation.py` | Single-field ablation (Table 5) | runs |
| `04_sensibilite.py` | Sensitivity analyses (Section 5.7, Supplement S3) | needs the confidential records |
| `05_equite.py` | Subgroup audit by sex and marital status (Section 5.7) | runs |
| `06_figures.py` | Figures of the long version of the manuscript | runs |
| `07_lasso_coefficients.py` | Lasso coefficients and selection (Supplement S6) | runs |
| `08_circularity.py` | Circularity test: reliance on officer judgements | runs |
| `09_learning_curve.py` | Learning curve, first protocol | runs |
| `09b_fig_learning_curve.py` | Figure for the first learning curve | runs |
| `10_sensibilite_cible.py` | Sensitivity to the outcome definition (Supplement S3) | needs the confidential records |
| `11_role_duree.py` | Role of the loan term | runs |
| `12_redondance.py` | Identical fields and partitions (Table 3, Section 5.2) | runs |
| `12b_fig_redondance.py` | Correlation among predictors (Figure 2) | stops: two fields drawn in the figure have their categories pooled into `other` in the synthetic data |
| `13_regle_triviale.py` | One- and two-condition rules (Table 4) | stops at an assertion that checks a property of the real data (the two conditions do not overlap) |
| `14_ablation_bloc.py` | Block ablation and stratified analysis (Table 5) | runs |
| `15_courbe_apprentissage.py` | Learning curve, final protocol (Section 5.6) | runs |
| `15b_fig_courbe.py` | Learning-curve figure (Supplementary Figure S2) | runs |
| `16_fig_checklist.py` | Data-collection checklist figure (long version) | runs |
| `17_fig_roc.py` | ROC, precision-recall and calibration (Supplement S2) | runs |
| `18_fig_process.py` | Lending-process diagram (Figure 1; needs no data) | runs |
| `19_compte_parametres.py` | Count of estimated parameters (Section 4.2) | runs |
| `20_partition.py` | Partition of the files by the two conditions and by crop (Section 5.3) | stops at an assertion that checks a property of the real data (the two conditions and the rest partition the files) |
| `21_heterogeneite_produit.py` | Product heterogeneity by loan term (Section 5.5) | runs |
| `23_regle_hors_pli.py` | Two-condition rule chosen within folds, scored out of fold (Table 4) | runs |
| `24_faux_positifs_groupes.py` | False-positive and false-negative rates by group (Supplement S4) | stops at an assertion that checks a figure of the real data (2 false positives, 3 false negatives) |
| `check_integrity.py` | Internal consistency checks of the reported figures | needs the confidential records |
| `make_synthetic.py` | Generator of the synthetic dataset (needs the confidential records) | needs the confidential records |

## Licence

Code: MIT licence (`LICENSE`). Synthetic dataset: Creative Commons Attribution
4.0 International (CC BY 4.0).

## Citation

Please cite the article; citation metadata are in
`CITATION.cff`.
