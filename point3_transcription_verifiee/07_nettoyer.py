"""Applique les décisions de la vérification, nettoie le texte et détecte la langue.

Seul le dialogue en anglais est gardé comme entrée du modèle texte ; les autres
extraits restent dans le corpus avec un texte vide et la raison de l'exclusion.
"""
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from lib.commun import RACINE, RESULTATS, arguments, ecrire_jsonl, lire_json, lire_jsonl
from lib.nettoyage import DetecteurLangue, nettoyer, normaliser


def nettoyer_tout(routage, decisions, donnees, regles, detecteur):
    par_extrait = {d["video_id"]: d for d in decisions}
    sortie = []
    for ligne in tqdm(routage, desc="Nettoyage"):
        film, clip = ligne["film_id"], ligne["video_id"]
        verif = par_extrait.get(clip)
        verifie = verif is not None and verif.get("review_status") in {"reviewed", "automatic_resolved"}
        # Une décision manuelle remplace toujours la proposition automatique.
        texte = verif["final_text"] if verifie else ligne["proposed_text"]
        source = verif.get("selected_source") if verifie else ligne["proposed_source"]
        resultat = nettoyer(texte, source, ligne["raw_transcripts"], detecteur, regles)

        annotation = lire_json(donnees / "annotation" / film / (clip + ".json"))
        description, _ = normaliser(annotation.get("caption"), regles, retirer_musique=False)
        description_ok = bool(description) and detecteur.detect(description)["language"] == "en"
        resultat.update(film_id=film, video_id=clip, emotion=annotation["emotion"], selected_source=source,
                        selection_origin=verif.get("decision_origin", "human") if verifie else "automatic_routing",
                        human_reviewed=bool(verifie and verif.get("human_reviewed")),
                        human_confirmed=bool(verifie and verif.get("human_confirmed")),
                        caption_text=description if description_ok else "", caption_text_available=description_ok)
        sortie.append(resultat)
    return sortie


def main():
    parseur = arguments(__doc__)
    parseur.add_argument("--routing", type=Path, default=RESULTATS / "aiguillage.jsonl")
    parseur.add_argument("--decisions", type=Path, default=RESULTATS / "decisions_verification.jsonl")
    args = parseur.parse_args()
    decisions = lire_jsonl(args.decisions) if args.decisions.exists() else []
    sortie = nettoyer_tout(lire_jsonl(args.routing), decisions, args.data,
                           lire_json(args.config)["cleaning"], DetecteurLangue(RACINE))
    ecrire_jsonl(args.work / "cleaned.jsonl", sortie)
    print(dict(Counter(r["text_status"] for r in sortie)))


if __name__ == "__main__":
    main()
