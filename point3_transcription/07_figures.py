"""07 - Figures de synthese pour la presentation.

Produit trois graphiques dans resultats/figures/ :
  1. la part d'extraits contenant des dialogues, par emotion ;
  2. la repartition du taux de parole dans les extraits ;
  3. la longueur des transcriptions (nombre de mots).

Usage :
    python 07_figures.py
    python 07_figures.py --csv resultats/transcriptions_wav.csv
"""

import argparse
import collections
import csv

import matplotlib
matplotlib.use("Agg")          # pas de fenetre, on ecrit des fichiers
import matplotlib.pyplot as plt

import config


def lire(chemin):
    with chemin.open(encoding="utf-8") as flux:
        return list(csv.DictReader(flux))


def main():
    analyseur = argparse.ArgumentParser(description="Figures du point 3")
    analyseur.add_argument("--csv", default=str(config.RESULTATS / "transcriptions_voix.csv"))
    options = analyseur.parse_args()

    lignes = lire(chemin := __import__("pathlib").Path(options.csv))
    figures = config.RESULTATS / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # 1. Part d'extraits avec parole, par emotion
    par_emotion = collections.defaultdict(lambda: [0, 0])
    for ligne in lignes:
        cle = ligne["emotion"]
        par_emotion[cle][1] += 1
        par_emotion[cle][0] += int(ligne["a_parole"])

    emotions = sorted(par_emotion)
    parts = [par_emotion[e][0] / par_emotion[e][1] for e in emotions]

    figure, axe = plt.subplots(figsize=(8, 4))
    axe.bar(emotions, parts, color="#4a7ebb")
    axe.set_ylabel("Part d'extraits avec des dialogues")
    axe.set_title("Detection de parole par emotion")
    axe.yaxis.set_major_formatter(lambda valeur, _: f"{valeur:.0%}")
    axe.tick_params(axis="x", rotation=40)
    figure.tight_layout()
    figure.savefig(figures / "parole_par_emotion.png", dpi=150)
    plt.close(figure)

    # 2. Repartition du taux de parole
    ratios = [float(ligne["ratio_parole"]) for ligne in lignes if ligne["ratio_parole"]]
    figure, axe = plt.subplots(figsize=(8, 4))
    axe.hist(ratios, bins=20, color="#7ea86b", edgecolor="white")
    axe.set_xlabel("Part de l'extrait couverte par de la parole")
    axe.set_ylabel("Nombre d'extraits")
    axe.set_title(f"Distribution du taux de parole ({len(ratios)} extraits)")
    figure.tight_layout()
    figure.savefig(figures / "distribution_parole.png", dpi=150)
    plt.close(figure)

    # 3. Longueur des transcriptions
    mots = [len(ligne["texte"].split()) for ligne in lignes if ligne["texte"]]
    figure, axe = plt.subplots(figsize=(8, 4))
    axe.hist(mots, bins=20, color="#bb7a4a", edgecolor="white")
    axe.set_xlabel("Nombre de mots dans la transcription")
    axe.set_ylabel("Nombre d'extraits")
    axe.set_title(f"Longueur des transcriptions ({len(mots)} extraits)")
    figure.tight_layout()
    figure.savefig(figures / "longueur_transcriptions.png", dpi=150)
    plt.close(figure)

    print(f"{len(lignes)} extraits | figures -> {figures}")


if __name__ == "__main__":
    main()
