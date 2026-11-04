"""Écrit la transcription finale : resultats/transcriptions_verifiees.csv (un extrait par ligne).

Avec --json, écrit aussi un fichier <film>/<extrait>.json par extrait
(format de movie/transcription/).
"""
import csv
from pathlib import Path

from lib.commun import RESULTATS, arguments, chemin_extrait, ecrire_json, lire_jsonl

COLONNES = ["video_id", "film_id", "texte", "a_dialogue", "statut", "langue", "raison_exclusion"]


def fiche(ligne):
    statut = ligne["text_status"]
    langue = "en" if ligne["asr_text_available"] else None
    if statut == "non_english":
        langue = (ligne.get("language_evidence") or {}).get("language")
    return dict(film_id=ligne["film_id"], video_id=ligne["video_id"],
                asr_text=ligne["asr_text"], asr_text_available=ligne["asr_text_available"],
                text_status=statut, language=langue, exclusion_reason=ligne.get("exclusion_reason"))


def main():
    parseur = arguments(__doc__)
    parseur.add_argument("--sortie", type=Path, default=RESULTATS / "transcriptions_verifiees.csv")
    parseur.add_argument("--json", type=Path, help="dossier pour un fichier JSON par extrait")
    args = parseur.parse_args()
    fiches = [fiche(ligne) for ligne in lire_jsonl(args.work / "cleaned.jsonl")]

    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sortie, "w", encoding="utf-8", newline="") as flux:
        ecrivain = csv.writer(flux)
        ecrivain.writerow(COLONNES)
        for f in sorted(fiches, key=lambda f: f["video_id"]):
            ecrivain.writerow([f["video_id"], f["film_id"], f["asr_text"], int(f["asr_text_available"]),
                               f["text_status"], f["language"] or "", f["exclusion_reason"] or ""])
    print(f"{len(fiches)} extraits dans {args.sortie}")

    if args.json:
        for f in fiches:
            ecrire_json(chemin_extrait(args.json, f), f)
        print(f"Fichiers JSON écrits dans {args.json}")


if __name__ == "__main__":
    main()
