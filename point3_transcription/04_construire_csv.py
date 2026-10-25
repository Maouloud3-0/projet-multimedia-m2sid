"""04 - Rassembler les transcriptions dans un tableau unique (CSV).

Le CSV est le livrable du point 3 : il sera utilise par le modele texte
du point 4. Chaque ligne = un extrait.

Usage :
    python 04_construire_csv.py
"""

import argparse
import csv

import config
import utils

COLONNES = [
    "video_id", "film_id", "partition", "emotion", "duree", "langue",
    "a_parole", "ratio_parole", "no_speech_prob", "logprob_moyen", "texte",
]


def main():
    analyseur = argparse.ArgumentParser(description="Assemblage des transcriptions")
    analyseur.add_argument("--source", default="voix",
                           help="dossier de transcriptions : voix, wav, voix_sans_vad...")
    options = analyseur.parse_args()

    annotations = utils.annotations()
    partition = utils.partition()
    ligne_dossier = config.TRANSCRIPTIONS / options.source
    lignes = []

    for fichier in sorted(ligne_dossier.glob("*.json")):
        resultat = utils.lire_json(fichier)
        video_id = resultat["video_id"]
        film_id = resultat["film_id"]
        annotation = annotations.get(video_id, {})

        lignes.append({
            "video_id": video_id,
            "film_id": film_id,
            "partition": partition.get(film_id, "?"),
            "emotion": annotation.get("emotion", ""),
            "duree": annotation.get("duration", ""),
            "langue": resultat.get("langue", ""),
            "a_parole": int(bool(resultat.get("a_parole"))),
            "ratio_parole": "" if resultat.get("ratio_parole") is None else resultat["ratio_parole"],
            "no_speech_prob": resultat.get("no_speech_prob", ""),
            "logprob_moyen": "" if resultat.get("logprob_moyen") is None else resultat["logprob_moyen"],
            "texte": resultat.get("texte", ""),
        })

    if not lignes:
        print(f"Aucune transcription dans {ligne_dossier} : lancer 03_transcrire.py d'abord.")
        return

    config.RESULTATS.mkdir(parents=True, exist_ok=True)
    sortie = config.RESULTATS / f"transcriptions_{options.source}.csv"
    with sortie.open("w", newline="", encoding="utf-8") as flux:
        ecrivain = csv.DictWriter(flux, fieldnames=COLONNES)
        ecrivain.writeheader()
        ecrivain.writerows(lignes)

    total = len(lignes)
    avec_parole = sum(ligne["a_parole"] for ligne in lignes)
    print(f"{total} extraits, {avec_parole} avec de la parole ({avec_parole / total:.1%})")
    print("Termine ->", sortie)


if __name__ == "__main__":
    main()
