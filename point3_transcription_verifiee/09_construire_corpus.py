"""Assemble le corpus final : un extrait par ligne, avec sa partition, son émotion,
la description de la scène (annotation d'origine) et le dialogue transcrit.

Entrées : movie/annotation/, splits/*.txt, resultats/transcriptions_verifiees.csv
Sortie  : resultats/corpus.csv, lu par les modèles texte, audio, vidéo et la fusion.
"""
import csv
from collections import Counter
from pathlib import Path

from lib.commun import CODE, RESULTATS, arguments, lire_json

SPLITS = CODE.parent / "splits"
COLONNES = ["video_id", "film_id", "partition", "emotion", "caption", "dialogue",
            "a_dialogue", "statut_texte", "langue", "duree"]


def lire_partition():
    partition = {}
    for nom in ("train", "dev", "test"):
        for film_id in (SPLITS / f"{nom}.txt").read_text(encoding="utf-8").split():
            partition[film_id] = nom
    return partition


def lire_transcriptions(chemin):
    with open(chemin, encoding="utf-8", newline="") as flux:
        return {ligne["video_id"]: ligne for ligne in csv.DictReader(flux)}


def main():
    parseur = arguments(__doc__)
    parseur.add_argument("--transcriptions", type=Path, default=RESULTATS / "transcriptions_verifiees.csv")
    parseur.add_argument("--sortie", type=Path, default=RESULTATS / "corpus.csv")
    args = parseur.parse_args()
    partition = lire_partition()
    transcriptions = lire_transcriptions(args.transcriptions)

    lignes = []
    for fichier in sorted((args.data / "annotation").glob("*/*.json")):
        if fichier.name.startswith("._"):
            continue
        annotation = lire_json(fichier)
        film_id, video_id = fichier.parent.name, annotation["video_id"]
        t = transcriptions[video_id]
        lignes.append(dict(video_id=video_id, film_id=film_id, partition=partition[film_id],
                           emotion=annotation["emotion"], caption=annotation.get("caption") or "",
                           dialogue=t["texte"], a_dialogue=t["a_dialogue"], statut_texte=t["statut"],
                           langue=t["langue"], duree=annotation["duration"]))

    with open(args.sortie, "w", encoding="utf-8", newline="") as flux:
        ecrivain = csv.DictWriter(flux, fieldnames=COLONNES)
        ecrivain.writeheader()
        ecrivain.writerows(lignes)

    print(f"{len(lignes)} extraits dans {args.sortie}")
    for nom in ("train", "dev", "test"):
        extraits = [l for l in lignes if l["partition"] == nom]
        films = {l["film_id"] for l in extraits}
        avec_dialogue = sum(l["a_dialogue"] == "1" for l in extraits)
        print(f"  {nom:5s} : {len(films)} films, {len(extraits)} extraits, {avec_dialogue} avec dialogue anglais")
    print("  émotions :", dict(sorted(Counter(l["emotion"] for l in lignes).items())))


if __name__ == "__main__":
    main()
