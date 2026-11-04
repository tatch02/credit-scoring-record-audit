# -*- coding: utf-8 -*-
"""
Figure: correlation among predictors (Figure 2).

Shows the fields most correlated with the outcome and the groups found
identical by 12_redondance.py, boxed. Produced at the exact insertion width
(455.2 pt), with no font smaller than 7 pt.

Outputs: outputs/figures/fig8_predictor_matrix.png,
         outputs/figures/dp/fig_correlation.pdf
Usage: python 12b_fig_redondance.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # polices TrueType incorporées
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import config

# Short English labels, as used in the article.
ETIQUETTES = {
    "Moralité par rapport au marché = Mauvaise":                    "Market reputation: poor",
    "Actif dans culture pendant les 03 dernières saisons ? = NON":  "Active in cultivation: no",
    "Justificatifs des activités ? = NON":                          "Documentary evidence: no",
    "apprec_credit_ord":                                            "Prior-loan appraisal",
    "contexte_ord":                                                 "Market context",
    "miss_Revenu_glo":                                              "Income not recorded",
    "miss_Taux_d_end":                                              "Debt ratio not recorded",
    "Nature de la culture = mais":                                  "Crop: maize",
    "Engagement Chez Confrère = OUI":                               "Commitment at competitor",
    "Traite chez confrère":                                         "Dealings with competitor",
}

# Groupes établis identiques par le script 12 (encadrés sur la figure).
GROUPES = [
    ["Active in cultivation: no", "Documentary evidence: no"],
    ["Income not recorded", "Debt ratio not recorded", "Crop: maize"],
    ["Commitment at competitor", "Dealings with competitor"],
]

LARGEUR_PT = 455.2
POUCES = LARGEUR_PT / 72.27


def main():
    M = pd.read_csv(config.OUTPUTS_DIR / "redondance_matrice.csv", index_col=0)
    manquantes = [c for c in ETIQUETTES if c not in M.columns]
    assert not manquantes, f"colonnes absentes de la matrice : {manquantes}"

    ordre = list(ETIQUETTES)
    S = M.loc[ordre, ordre]
    S.index = S.columns = [ETIQUETTES[c] for c in ordre]

    fig, ax = plt.subplots(figsize=(POUCES, POUCES * 0.80), dpi=400)
    im = ax.imshow(S.values, cmap="RdBu_r", vmin=-1, vmax=1)

    k = len(S)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(S.columns, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(S.index, fontsize=8)
    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)

    for a in range(k):
        for bb in range(k):
            v = S.values[a, bb]
            if np.isnan(v):
                continue
            ax.text(bb, a, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                    fontsize=7.2, color="white" if abs(v) > 0.55 else "#16181D")

    # encadre les groupes identiques
    for g in GROUPES:
        idx = [list(S.columns).index(x) for x in g]
        i0, i1 = min(idx), max(idx)
        ax.add_patch(Rectangle((i0 - .5, i0 - .5), i1 - i0 + 1, i1 - i0 + 1,
                               fill=False, edgecolor="#16181D", linewidth=2.2,
                               zorder=10, clip_on=False))   # au-dessus de la grille

    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("Pearson correlation", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    # Pas de titre interne : la légende LaTeX porte le propos. Une figure qui
    # s'auto-intitule produit deux légendes contradictoires sur la même page.
    fig.tight_layout()

    # tight_layout garde la taille de la figure : on vérifie que rien n'en sort
    fig.canvas.draw()
    W, H = fig.bbox.width, fig.bbox.height
    hors = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
            if t.get_visible() and t.get_text()
            and (lambda b: b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1)(
                t.get_window_extent())]
    assert not hors, f"texte hors de la figure : {hors}"

    for dest in (config.OUTPUTS_DIR / "figures" / "fig8_predictor_matrix.png",
                 config.PROJECT_ROOT / "article" / "figures" / "fig8_predictor_matrix.png",
                 config.OUTPUTS_DIR / "figures" / "dp" / "fig_correlation.pdf"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, facecolor="white")
        print(f"  écrit : {dest.relative_to(config.PROJECT_ROOT)}")
    print(f"  taille : {LARGEUR_PT:.1f} x {LARGEUR_PT * 0.80:.1f} pt (insertion à 100 %)")
    plt.close(fig)

    print("\n  Blocs encadrés (corrélation exactement 1) :")
    for g in GROUPES:
        sub = S.loc[g, g].values
        hors = sub[~np.eye(len(g), dtype=bool)]
        print(f"    {' == '.join(g)}   min |r| = {np.abs(hors).min():.6f}")


if __name__ == "__main__":
    main()
