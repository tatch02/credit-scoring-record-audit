# -*- coding: utf-8 -*-
"""
Figure: where each entry in the dataset is produced along the lending process (Figure 1).

Needs no data. Chip widths are measured on the rendered text, so no label can
overflow; the script stops if one does. Produced at the exact insertion width,
with no font smaller than 7 pt.

Outputs: article/figures/fig10_process.png, outputs/figures/dp/fig_cascade.pdf
Usage: python 18_fig_process.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # polices TrueType incorporées
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import config

REC, JUG, DER, OUT = "record", "judgement", "derived", "outcome"
COUL = {REC: "#DCE1E9", JUG: "#EFD3D6", DER: "#D6E6DD", OUT: "#E2DAEB"}
BORD = {REC: "#6B7789", JUG: "#8C2F39", DER: "#2F6B4F", OUT: "#5B4A72"}
ENCRE, GRIS, ROUGE = "#16181D", "#5D6470", "#8C2F39"
SAUT = ("", None)                              # force un nouveau rang de puces

ETAPES = [
    ("", "Technical sheet", "standing scale, same for all files", [
        ("credit ceiling", REC), ("loan term", REC)]),
    ("", "Application", "the client files the dossier", [
        ("requested amount", REC), ("identity papers", REC),
        ("location map", REC), ("collateral or guarantee", REC)]),
    ("1", "Reception", "appraisal fee paid", [
        ("reception date", REC), ("first-time or repeat", REC)]),
    ("2", "Appraisal", "loan officer, field visit", [
        ("crop", REC), ("cultivated area", REC), ("expected quantities", REC),
        ("years in the occupation", REC), ("estimated revenue", DER),
        ("market reputation", JUG), ("behaviour", JUG), SAUT,
        ("active in cultivation", JUG), ("documentary evidence", JUG)]),
    ("3", "Branch credit committee", "fills gaps, adjusts the file", [
        ("file adjustments", REC)]),
    ("4", "Head of commitments", "re-checks, returns gaps", []),
    ("5", "Management committee", "first approval", []),
    ("6", "Commitments committee", "final decision, last resort", [
        ("granted amount", REC), ("final terms", REC)]),
    ("7", "Disbursement", "by tranches, by crop stage", [
        ("disbursement dates", REC), ("repayment schedule", REC)]),
    ("", "Monitoring", "scheduled and surprise visits", [
        ("incident dates", REC), ("final status", OUT)]),
]
PREMIERE_REFUS, DERNIERE_REFUS = "Appraisal", "Commitments committee"
DUPLIQUEES = ("active in cultivation", "documentary evidence")

LARGEUR_PT = 455.2
X_MAX = 12.0
PT_PAR_UNITE = LARGEUR_PT / X_MAX      # exact : l'axe occupe toute la figure

X_ETAPE, L_ETAPE = 0.10, 3.80
X_VAR, X_FIN = 4.12, 11.92
H_PUCE, H_LIGNE, PAD_PUCE, MARGE_PUCE = 0.30, 0.40, 0.10, 0.22
INTERLIGNE = 0.29                      # entre deux lignes de note

FS_ETAPE, FS_GLOSE, FS_PUCE, FS_NOTE = 8.6, 7.2, 7.2, 7.4
PLANCHER = 7.0

DPI = 400          # un seul dpi partout : mesurer et sauver au meme
_SCRATCH = None


def largeur(txt, fs, **kw):
    """Largeur du texte en unités de données, MESURÉE SUR LE RENDU.

    Le mapping x ne dépend que de la largeur de la figure, jamais de sa hauteur,
    donc une figure d'appoint de la bonne largeur suffit à mesurer exactement.
    """
    global _SCRATCH
    if _SCRATCH is None:
        f = plt.figure(figsize=(LARGEUR_PT / 72.27, 4.0), dpi=DPI)
        a = f.add_axes([0, 0, 1, 1])
        a.set_xlim(0, X_MAX); a.set_ylim(0, 10); a.axis("off")
        f.canvas.draw()
        _SCRATCH = (f, a)
    f, a = _SCRATCH
    t = a.text(0, 5, txt, fontsize=fs, **kw)
    f.canvas.draw()
    bb = t.get_window_extent()
    inv = a.transData.inverted()
    w = inv.transform((bb.x1, bb.y1))[0] - inv.transform((bb.x0, bb.y0))[0]
    t.remove()
    return w


def plier(txt, fs, dispo=None, **kw):
    """Découpe un texte en lignes qui tiennent dans la largeur disponible."""
    dispo = dispo if dispo is not None else (X_FIN - X_ETAPE)
    lignes, courante = [], ""
    for mot in txt.split():
        essai = f"{courante} {mot}".strip()
        if courante and largeur(essai, fs, **kw) > dispo:
            lignes.append(courante); courante = mot
        else:
            courante = essai
    if courante:
        lignes.append(courante)
    return lignes


def main():
    assert min(FS_ETAPE, FS_GLOSE, FS_PUCE, FS_NOTE) >= PLANCHER

    # ---- pass 1 : répartition des puces en rangs, d'où la hauteur totale
    plan = []
    for num, nom, glose, variables in ETAPES:
        rangs, courante, x = [], [], X_VAR
        for v, st in variables:
            if st is None:                          # saut de rang imposé
                if courante:
                    rangs.append(courante)
                courante, x = [], X_VAR
                continue
            w = largeur(v, FS_PUCE) + MARGE_PUCE
            if x + w > X_FIN and courante:
                rangs.append(courante); courante, x = [], X_VAR
            courante.append((v, st, x, w)); x += w + PAD_PUCE
        if courante:
            rangs.append(courante)
        h = max(1.00, len(rangs) * H_LIGNE + 0.42)
        if any(v in DUPLIQUEES for rg in rangs for v, *_ in rg):
            h += 0.34                               # place de l'accolade
        plan.append((num, nom, glose, rangs, h))

    note_haut = plier("Colour marks whether an entry is a fact or a judgement, "
                      "a distinction this dataset does not keep.", FS_NOTE,
                      style="italic")
    pied_1 = plier("The analysed dataset holds the survivors of this sequence, "
                   "described by the entries their own appraisal produced.", FS_NOTE)
    pied_2 = plier("The refused files, the dates and the technical sheet exist "
                   "elsewhere in the institution: non-collections, not absences.",
                   FS_NOTE, style="italic")
    tete = 0.20 + len(note_haut) * INTERLIGNE + 0.62
    pied = 0.30 + (len(pied_1) + len(pied_2)) * INTERLIGNE + 0.14
    haut = tete + sum(p[4] for p in plan) + pied

    pouces_l = LARGEUR_PT / 72.27
    fig = plt.figure(figsize=(pouces_l, pouces_l * haut / X_MAX), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])          # l'axe EST la figure : mapping exact
    ax.set_xlim(0, X_MAX); ax.set_ylim(0, haut); ax.axis("off")
    ax.set_facecolor("white")
    textes_etape = []                        # textes qui doivent tenir dans un cadre

    # ---- bandeau de légende
    y = haut - 0.24
    for ligne in note_haut:
        ax.text(X_ETAPE, y, ligne, fontsize=FS_NOTE, color=GRIS, va="center",
                style="italic")
        y -= INTERLIGNE
    y -= 0.12
    x = X_ETAPE
    for st, lib in [(REC, "a record"), (JUG, "an officer's judgement"),
                    (DER, "derived"), (OUT, "the outcome")]:
        ax.add_patch(FancyBboxPatch((x, y - 0.11), 0.22, 0.22,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=COUL[st], edgecolor=BORD[st], lw=0.7))
        ax.text(x + 0.31, y, lib, fontsize=FS_NOTE, color=ENCRE, va="center")
        x += 0.31 + largeur(lib, FS_NOTE) + 0.45
    y -= 0.30
    ax.plot([X_ETAPE, X_FIN], [y, y], color=ENCRE, lw=0.9)
    y -= 0.12

    # ---- les étapes
    pos, pos_puce = {}, {}
    for num, nom, glose, rangs, h in plan:
        y_haut, y_bas = y, y - h
        cy_cadre = (y_haut + y_bas) / 2
        pos[nom] = (y_haut, y_bas)

        ax.add_patch(FancyBboxPatch((X_ETAPE, y_bas + 0.06), L_ETAPE, h - 0.12,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="#F4F5F8", edgecolor="#B4BCC9", lw=0.8))
        etiq = f"{num} · {nom}" if num else nom
        textes_etape.append(ax.text(X_ETAPE + 0.15, cy_cadre + 0.15, etiq,
                                    fontsize=FS_ETAPE, color=ENCRE, va="center"))
        textes_etape.append(ax.text(X_ETAPE + 0.15, cy_cadre - 0.19, glose,
                                    fontsize=FS_GLOSE, color=GRIS, va="center",
                                    style="italic"))

        for i, rang in enumerate(rangs):
            yy = y_haut - 0.34 - i * H_LIGNE
            for v, st, x0, w in rang:
                ax.add_patch(FancyBboxPatch((x0, yy - H_PUCE / 2), w, H_PUCE,
                                            boxstyle="round,pad=0,rounding_size=0.05",
                                            facecolor=COUL[st], edgecolor=BORD[st], lw=0.6))
                ax.text(x0 + w / 2, yy, v, fontsize=FS_PUCE, color=ENCRE,
                        ha="center", va="center")
                if v in DUPLIQUEES:
                    pos_puce[v] = (x0, x0 + w, yy, y_bas)
        y = y_bas

    # ---- le constat central : deux champs, un seul acte
    a, b = pos_puce[DUPLIQUEES[0]], pos_puce[DUPLIQUEES[1]]
    assert a[2] == b[2], "les deux champs identiques doivent être sur le même rang"
    x0, x1 = min(a[0], b[0]), max(a[1], b[1])
    yb = a[2] - H_PUCE / 2 - 0.04
    ax.plot([x0, x0, x1, x1], [yb, yb - 0.09, yb - 0.09, yb], color=ROUGE, lw=1.0)
    y_lib = yb - 0.27
    assert y_lib - 0.12 > a[3] + 0.06, "l'étiquette de l'accolade sort du rang"
    # aligné à gauche sur l'accolade : centré, il chevauchait le trait des refus
    ax.text(x0, y_lib, "identical on all 89 files: two headings, one content",
            fontsize=FS_NOTE, color=ROUGE, ha="left", va="center")

    # ---- les refusés : cinq points de décision, étapes 2 à 6
    y_top, y_bot = pos[PREMIERE_REFUS][0], pos[DERNIERE_REFUS][1]
    # la flèche passe entre les étapes 4 et 5, où aucun rang de puces n'est tracé
    ymid = pos["Head of commitments"][1]
    xa = X_ETAPE + L_ETAPE + 0.08
    ax.plot([xa, xa], [y_top - 0.06, y_bot + 0.06], color=ROUGE, lw=1.0,
            linestyle=(0, (2, 2)))
    lignes_boite = [("Refused at any of these stages", FS_NOTE, ROUGE),
                    ("archived on paper in the branch,", FS_NOTE, GRIS),
                    ("never digitised", FS_NOTE, GRIS)]
    l_boite = max(largeur(t, f) for t, f, _ in lignes_boite) + 0.32
    h_boite = len(lignes_boite) * INTERLIGNE + 0.26
    x_boite = X_FIN - l_boite
    ax.annotate("", xy=(x_boite - 0.04, ymid), xytext=(xa, ymid),
                arrowprops=dict(arrowstyle="-|>", color=ROUGE, lw=1.0,
                                linestyle=(0, (2, 2)), mutation_scale=8))
    ax.add_patch(FancyBboxPatch((x_boite, ymid - h_boite / 2), l_boite, h_boite,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="#FBF0F1", edgecolor=ROUGE, lw=0.9))
    yl = ymid + (len(lignes_boite) - 1) * INTERLIGNE / 2
    for t, f, c in lignes_boite:
        ax.text(x_boite + 0.16, yl, t, fontsize=f, color=c, va="center")
        yl -= INTERLIGNE

    # ---- pied
    y -= 0.16
    ax.plot([X_ETAPE, X_FIN], [y, y], color=ENCRE, lw=0.9)
    y -= 0.30
    for ligne in pied_1:
        ax.text(X_ETAPE, y, ligne, fontsize=FS_NOTE, color=ENCRE, va="center")
        y -= INTERLIGNE
    for ligne in pied_2:
        ax.text(X_ETAPE, y, ligne, fontsize=FS_NOTE, color=GRIS, va="center",
                style="italic")
        y -= INTERLIGNE
    assert y > -0.05, f"le pied sort de la figure ({y:.2f})"

    # ---- contrôles : rien ne déborde, ni de la figure ni d'un cadre d'étape
    fig.canvas.draw()
    inv = ax.transData.inverted()
    hors = []
    for txt in ax.texts:
        bb = txt.get_window_extent()
        (gx, by), (dx, hy) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        if gx < -0.01 or dx > X_FIN + 0.02 or by < -0.01 or hy > haut + 0.01:
            hors.append((txt.get_text()[:44], round(dx, 2)))
    assert not hors, f"texte hors de la figure : {hors}"
    assert len(note_haut) == 1 and len(pied_1) == 1 and len(pied_2) == 1,         "une note est coupée sur deux lignes"
    trop_large = []
    for txt in textes_etape:
        dx = inv.transform((txt.get_window_extent().x1, 0))[0]
        if dx > X_ETAPE + L_ETAPE - 0.10:
            trop_large.append((txt.get_text(), round(dx, 2)))
    assert not trop_large, f"texte hors de son cadre d'étape : {trop_large}"
    debord = [(v, x0 + w) for _, _, _, rangs, _ in plan for rg in rangs
              for v, st, x0, w in rg if x0 + w > X_FIN + 1e-9]
    assert not debord, debord

    sorties = [config.PROJECT_ROOT / "article" / "figures" / "fig10_process.png",
               config.OUTPUTS_DIR / "figures" / "dp" / "fig_cascade.pdf"]
    for dest in sorties:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, facecolor="white")   # dpi de la figure, inchangé
        print(f"  écrit : {dest.relative_to(config.PROJECT_ROOT)}")
    plt.close(fig)

    print(f"  taille : {LARGEUR_PT:.1f} x {haut * PT_PAR_UNITE:.1f} pt "
          f"(insertion à 100 %)")
    print(f"  plus petite police : {min(FS_GLOSE, FS_PUCE, FS_NOTE)} pt à l'échelle 1")


if __name__ == "__main__":
    main()
