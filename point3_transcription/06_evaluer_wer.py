"""05 - Mesurer la qualite des transcriptions sur un echantillon transcrit a la main.

Le fichier reference/transcriptions_manuelles.csv contient trois colonnes :
    video_id,texte,a_parole
Laisser a_parole vide tant que l'extrait n'a pas ete ecoute : il est ignore.
Mettre texte vide et a_parole a 0 pour un extrait sans dialogue.

Deux mesures sont calculees :
  - le WER (taux d'erreur sur les mots) ;
  - l'exactitude de la detection de parole (y a-t-il des dialogues ou non).

Usage :
    python 05_evaluer_wer.py
    python 05_evaluer_wer.py --csv resultats/config_C.csv   # comparer une autre config
"""

import argparse
import csv
from pathlib import Path

import jiwer

import config

REFERENCE = Path(__file__).resolve().parent / "reference" / "transcriptions_manuelles.csv"


def normaliser(texte):
    """Minuscules, ponctuation simplifiee, espaces reduits."""
    texte = texte.lower()
    for signe in ",.;:!?\"()[]{}« »":
        texte = texte.replace(signe, " ")
    return " ".join(texte.split())


def lire_reference():
    reference = {}
    with REFERENCE.open(encoding="utf-8") as flux:
        for ligne in csv.DictReader(flux):
            if not (ligne.get("a_parole") or "").strip():
                continue                       # ligne pas encore annotee
            reference[ligne["video_id"]] = {
                "texte": normaliser(ligne.get("texte") or ""),
                "a_parole": int(ligne["a_parole"]),
            }
    return reference


def lire_sortie(chemin):
    produit = {}
    with chemin.open(encoding="utf-8") as flux:
        for ligne in csv.DictReader(flux):
            produit[ligne["video_id"]] = {
                "texte": normaliser(ligne["texte"]),
                "a_parole": int(ligne["a_parole"]),
            }
    return produit


def main():
    analyseur = argparse.ArgumentParser(description="Evaluation des transcriptions")
    analyseur.add_argument("--csv", default=str(config.RESULTATS / "transcriptions_voix.csv"))
    options = analyseur.parse_args()

    reference = lire_reference()
    produit = lire_sortie(Path(options.csv))
    communs = [identifiant for identifiant in reference if identifiant in produit]

    if not communs:
        print("Aucun extrait annote en commun. Remplir reference/transcriptions_manuelles.csv")
        return

    # Exactitude de la detection de parole
    justes = sum(reference[i]["a_parole"] == produit[i]["a_parole"] for i in communs)

    # WER calculé sur les extraits qui contiennent vraiment des paroles
    parles = [i for i in communs if reference[i]["texte"]]
    if parles:
        mesures = jiwer.process_words(
            [reference[i]["texte"] for i in parles],
            [produit[i]["texte"] for i in parles],
        )
        wer = mesures.wer
    else:
        wer = float("nan")

    # Textes produits alors que l'extrait est silencieux (fausses alarmes)
    silencieux = [i for i in communs if not reference[i]["texte"]]
    fausses = sum(1 for i in silencieux if produit[i]["texte"])

    print(f"fichier evalue : {options.csv}")
    print(f"extraits annotees        : {len(communs)}")
    print(f"WER                      : {wer:.1%}")
    print(f"detection de parole      : {justes / len(communs):.1%}")
    print(f"fausses alarmes          : {fausses}/{len(silencieux)} extraits silencieux")


if __name__ == "__main__":
    main()
