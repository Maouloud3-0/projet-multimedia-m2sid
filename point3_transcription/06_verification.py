"""Vérification manuelle des extraits douteux dans Label Studio.

  python 06_verification.py preparer   -> tâches + interface dans work/review/
  python 06_verification.py lancer     -> démarre Label Studio (http://127.0.0.1:8081)
  python 06_verification.py exporter --input export.json -> resultats/decisions_verification.jsonl

Pour chaque extrait, on voit la vidéo, on écoute l'audio (et la voix séparée si elle
existe) et on choisit : une des transcriptions, un texte corrigé, ou « pas de dialogue ».
"""
import argparse
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

from lib.commun import (CODE, DONNEES, RACINE, RESULTATS, TRAVAIL, ecrire_json, ecrire_jsonl,
                        lire_json, lire_jsonl)
from lib.verification import interface_xml, lire_decision


def lien_media(chemin):
    """Label Studio lit les fichiers locaux à partir de RACINE (chemin absolu, ou relatif à RACINE)."""
    return "/data/local-files/?d=" + quote((RACINE / chemin).resolve().relative_to(RACINE).as_posix())


def preparer(routage, donnees, dossier):
    import cv2   # seulement dans .venv-review, pour lire la fréquence d'images des vidéos
    taches = {"original": [], "separated": []}
    for ligne in routage:
        if ligne["route"] != "uncertain":
            continue
        film, clip = ligne["film_id"], ligne["video_id"]
        video = donnees / "video" / film / (clip + ".mp4")
        lecteur = cv2.VideoCapture(str(video))
        images_par_seconde = lecteur.get(cv2.CAP_PROP_FPS)
        lecteur.release()
        brutes = ligne["raw_transcripts"]
        tache = dict(film_id=film, video_id=clip,
                     clip_heading=f"{clip} | {ligne['duration_seconds']:.2f} seconds",
                     case_info="Reasons: " + "; ".join(ligne["reasons"]),
                     raw_transcripts=brutes,
                     original_audio=lien_media(donnees / "audio" / film / (clip + ".mp3")),
                     video=lien_media(video), video_fps=images_par_seconde)
        for modele, prefixe in [("whisper", "whisper"), ("original_qwen", "qwen"), ("separated_qwen", "separated")]:
            r = brutes.get(modele)
            tache[prefixe + "_display"] = r["text"] if r and r["text"].strip() else "[No model text]"
            tache[prefixe + "_meta"] = "Language: " + str(r.get("language") or "unknown") if r else "Unavailable"
        separe = ligne.get("separated_audio")
        tache["separated_audio"] = lien_media(separe) if separe else None
        taches["separated" if separe else "original"].append(dict(data=tache))

    # Deux projets Label Studio : avec ou sans piste de voix séparée.
    gabarit = (CODE / "config" / "review.xml").read_text("utf-8")
    for groupe, liste in taches.items():
        if liste:
            ecrire_json(dossier / (groupe + "_tasks.json"), liste)
            (dossier / (groupe + ".xml")).write_text(interface_xml(gabarit, separe=groupe == "separated"), "utf-8")
    print(f"{sum(map(len, taches.values()))} extraits à vérifier dans {dossier}")
    print("Créer un projet Label Studio par groupe, coller le XML dans la configuration et importer les tâches.")


def exporter(taches, automatiques=()):
    """automatiques : identifiants d'annotations importées automatiquement (pas relues à la main)."""
    automatiques = set(automatiques)
    decisions = []
    for tache in taches:
        donnees = tache["data"]
        annotations = [a for a in tache.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        humaines = [a for a in annotations if a["id"] not in automatiques] or annotations
        derniere = max(humaines, key=lambda a: (a.get("updated_at") or a.get("created_at") or "", a["id"]))
        decision = lire_decision(donnees, derniere)
        auto = derniere["id"] in automatiques
        decision.update(film_id=donnees["film_id"], video_id=donnees["video_id"],
                        review_status="automatic_resolved" if auto else "reviewed",
                        decision_origin="automatic" if auto else "human",
                        human_reviewed=not auto, human_confirmed=not auto and decision["human_confirmed"])
        decisions.append(decision)
    return sorted(decisions, key=lambda d: d["video_id"])


def main():
    parseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commandes = parseur.add_subparsers(dest="commande", required=True)
    p = commandes.add_parser("preparer")
    p.add_argument("--routing", type=Path, default=RESULTATS / "aiguillage.jsonl")
    p.add_argument("--data", type=Path, default=DONNEES)
    p.add_argument("--output", type=Path, default=TRAVAIL / "review")
    p = commandes.add_parser("lancer")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--data-dir", type=Path, default=TRAVAIL / "label_studio", help="base de données de Label Studio")
    p = commandes.add_parser("exporter")
    p.add_argument("--input", type=Path, required=True, help="export JSON standard de Label Studio")
    p.add_argument("--automatiques", type=Path, help="liste JSON d'identifiants d'annotations importées")
    p.add_argument("--output", type=Path, default=RESULTATS / "decisions_verification.jsonl")
    args = parseur.parse_args()

    if args.commande == "preparer":
        preparer(lire_jsonl(args.routing), args.data, args.output)
    elif args.commande == "exporter":
        automatiques = lire_json(args.automatiques) if args.automatiques else []
        decisions = exporter(lire_json(args.input), automatiques)
        ecrire_jsonl(args.output, decisions)
        print(f"{len(decisions)} décisions enregistrées dans {args.output}")
    else:
        env = os.environ.copy()
        env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
        env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(RACINE)
        subprocess.run([str(RACINE / ".venv-review" / "Scripts" / "label-studio.exe"), "start",
                        "--port", str(args.port), "--internal-host", "127.0.0.1", "--no-browser",
                        "--data-dir", str(args.data_dir)], cwd=RACINE, env=env, check=True)


if __name__ == "__main__":
    main()
