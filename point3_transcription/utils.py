"""Petites fonctions partagees par les scripts du point 3."""

import json
import random
from pathlib import Path

import config


def extraits():
    """Liste des extraits du corpus : [{"video_id":..., "film_id":..., "mp3":...}]."""
    liste = []
    for mp3 in sorted(config.AUDIO_MP3.glob("*/*.mp3")):
        liste.append({
            "video_id": mp3.stem,
            "film_id": mp3.parent.name,
            "mp3": mp3,
        })
    return liste


def echantillon(liste, nombre, graine=42):
    """Sous-ensemble aleatoire reproductible (meme tirage dans tous les scripts)."""
    if not nombre or nombre >= len(liste):
        return liste
    generateur = random.Random(graine)
    return sorted(generateur.sample(liste, nombre), key=lambda e: e["video_id"])


def partition():
    """Dictionnaire film_id -> "train" / "dev" / "test".

    Les listes de films sont communes a tout le projet (dossier splits/).
    """
    dossier = Path(__file__).resolve().parents[1] / "splits"
    affectation = {}
    for nom in ("train", "dev", "test"):
        contenu = (dossier / f"{nom}.txt").read_text(encoding="utf-8")
        for film_id in contenu.split():
            affectation[film_id.strip()] = nom
    return affectation


def annotations():
    """Dictionnaire video_id -> contenu du fichier JSON d'annotation."""
    dico = {}
    for fichier in sorted(config.ANNOTATIONS.glob("*/*.json")):
        if fichier.name.startswith("._"):
            continue
        contenu = json.loads(fichier.read_text(encoding="utf-8"))
        dico[contenu["video_id"]] = contenu
    return dico


def preparer_dossiers():
    for dossier in (config.DATA, config.WAV, config.VOIX,
                    config.TRANSCRIPTIONS, config.RESULTATS):
        dossier.mkdir(parents=True, exist_ok=True)


def ecrire_json(chemin, contenu):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, ensure_ascii=False, indent=2),
                      encoding="utf-8")


def lire_json(chemin):
    return json.loads(chemin.read_text(encoding="utf-8"))
