# -*- coding: utf-8 -*-
"""
Figure: the data-collection framework, dimensions against tasks.

Marks the requirements without which each task is unattainable.

Output: outputs/figures/fig9_collection_checklist.png
Usage: python 16_fig_checklist.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import config

LOCK, REQ, NOT = "lock", "required", "not required"

DIMENSIONS = [
    ("1", "Outcome",              "default definition, dates, horizon",        REQ,  LOCK, REQ),
    ("2", "Selection cascade",    "refusals, stage, stated reason",            LOCK, NOT,  REQ),
    ("3", "Loan terms",           "requested, scale, granted, guarantors",     REQ,  REQ,  LOCK),
    ("4", "Financial proxies",    "crop, area, yield, price, missingness",     REQ,  REQ,  REQ),
    ("5", "Facts vs judgements",  "a tag per variable, composites split",      REQ,  REQ,  REQ),
    ("6", "Temporality",          "when each variable became known",           REQ,  LOCK, REQ),
    ("7", "Sensitive attributes", "retained outside the model, for audit",     REQ,  REQ,  REQ),
    ("8", "Governance",           "legal basis, consent, extraction protocol", REQ,  REQ,  REQ),
    ("9", "Structural integrity", "provenance per field; redundancy check",    REQ,  REQ,  REQ),
]

TACHES = [("Classification", "grant or refuse"),
          ("Monitoring", "an approved borrower"),
          ("Rationing", "how much to lend")]

COUL = {LOCK: "#8C2F39", REQ: "#C9CDD6", NOT: "#F3F4F7"}
TXT = {LOCK: "#FFFFFF", REQ: "#3D434E", NOT: "#949BA8"}
SIGNE = {LOCK: "LOCK", REQ: "required", NOT: "—"}

LARGEUR_PT = 455.2
POUCES = LARGEUR_PT / 72.27


def main():
    nb = len(DIMENSIONS)
    fig, ax = plt.subplots(figsize=(POUCES, POUCES * 0.62), dpi=200)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-0.9, nb + 1.9)
    ax.axis("off")

    x_dim, x0, larg, ecart = 0.15, 5.6, 1.72, 0.14

    # en-têtes de colonnes : nom et glose sur deux lignes distinctes, pour que
    # la glose puisse être plus petite et tenir dans la largeur de colonne
    for j, (nom, glose) in enumerate(TACHES):
        cx = x0 + j * (larg + ecart) + larg / 2
        ax.text(cx, nb + 1.22, nom, ha="center", va="center",
                fontsize=8.4, color="#16181D")
        ax.text(cx, nb + 0.88, glose, ha="center", va="center",
                fontsize=6.6, color="#5D6470", style="italic")
    ax.plot([x_dim, 11.05], [nb + 0.55, nb + 0.55], color="#16181D", lw=1.1)

    for i, (num, nom, desc, *etats) in enumerate(DIMENSIONS):
        y = nb - i - 0.5
        ax.text(x_dim, y, num, ha="left", va="center", fontsize=8.5,
                color="#8C2F39", weight="bold")
        ax.text(x_dim + 0.42, y + 0.16, nom, ha="left", va="center",
                fontsize=8.2, color="#16181D")
        ax.text(x_dim + 0.42, y - 0.19, desc, ha="left", va="center",
                fontsize=6.8, color="#5D6470", style="italic")

        for j, e in enumerate(etats):
            cx = x0 + j * (larg + ecart)
            ax.add_patch(FancyBboxPatch((cx, y - 0.32), larg, 0.64,
                                        boxstyle="round,pad=0,rounding_size=0.06",
                                        facecolor=COUL[e], edgecolor="none"))
            ax.text(cx + larg / 2, y, SIGNE[e], ha="center", va="center",
                    fontsize=7 if e != LOCK else 7.4, color=TXT[e],
                    weight="bold" if e == LOCK else "normal")

        if i < nb - 1:
            ax.plot([x_dim, 11.05], [y - 0.5, y - 0.5], color="#EAECF1", lw=0.6)

    # légende interne minimale : le sens du mot « lock »
    ax.add_patch(Rectangle((x_dim, -0.62), 0.3, 0.3, facecolor=COUL[LOCK], edgecolor="none"))
    ax.text(x_dim + 0.44, -0.47, "lock: without it the task is unattainable, not merely harder",
            ha="left", va="center", fontsize=7, color="#5D6470")

    fig.tight_layout(pad=0.3)
    for dest in (config.OUTPUTS_DIR / "figures" / "fig9_collection_checklist.png",
                 config.PROJECT_ROOT / "article" / "figures" / "fig9_collection_checklist.png"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  écrit : {dest.relative_to(config.PROJECT_ROOT)}")
    plt.close(fig)

    verrous = [(d[0], d[1], TACHES[j][0])
               for d in DIMENSIONS
               for j, etat in enumerate(d[3:]) if etat == LOCK]
    print(f"\n  {len(verrous)} verrou(s) :")
    for num, nom, tache in verrous:
        print(f"    dimension {num} ({nom}) -> {tache}")


if __name__ == "__main__":
    main()
