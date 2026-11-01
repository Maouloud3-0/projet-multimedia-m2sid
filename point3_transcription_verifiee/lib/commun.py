"""Chemins du projet et petites fonctions de lecture / écriture partagées."""
import argparse
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]

# Dossier du corpus : movie/, work/, models/ et les environnements Python.
RACINE = Path(os.environ.get(
    "PROJET_SID_DIR",
    r"C:\Users\zuowa\Downloads\PROJET_SID_2026\PROJET_SID_2026",
))
TRAVAIL = RACINE / "work"            # fichiers intermédiaires (non versionnés)
RESULTATS = CODE / "resultats"       # sorties versionnées dans le dépôt
DONNEES = RACINE / "movie"


def arguments(description):
    parseur = argparse.ArgumentParser(description=description)
    parseur.add_argument("--config", type=Path, default=CODE / "config" / "pipeline.json")
    parseur.add_argument("--work", type=Path, default=TRAVAIL)
    parseur.add_argument("--data", type=Path, default=DONNEES)
    parseur.add_argument("--limite", type=int, help="ne traiter que les N premiers extraits (essai)")
    return parseur


def lire_json(chemin):
    return json.loads(Path(chemin).read_text(encoding="utf-8-sig"))


def ecrire_json(chemin, valeur):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(valeur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lire_jsonl(chemin):
    lignes = Path(chemin).read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(ligne) for ligne in lignes if ligne.strip()]


def ecrire_jsonl(chemin, lignes):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as flux:
        for ligne in lignes:
            flux.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def chemin_extrait(dossier, ligne, suffixe=".json"):
    """Un fichier par extrait, rangé par film : dossier/<film_id>/<video_id><suffixe>."""
    return Path(dossier) / ligne["film_id"] / (ligne["video_id"] + suffixe)


def extraits(args):
    """Liste des extraits produite par 01_preparer_audio.py."""
    lignes = lire_jsonl(args.work / "clips.jsonl")
    if args.limite:
        lignes = lignes[:args.limite]
    return lignes


def deja_fait(chemin):
    """Permet de relancer un script sans refaire les extraits déjà traités."""
    chemin = Path(chemin)
    return chemin.exists() and lire_json(chemin).get("status") == "completed"


def modeles(groupe):
    """Chemins des modèles locaux (relatifs au dossier du corpus)."""
    entrees = lire_json(CODE / "config" / "modeles.json")[groupe]
    return {nom: (RACINE / chemin).resolve() for nom, chemin in entrees.items()}
