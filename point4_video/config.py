"""Configuration du point 4 (vidéo)."""

import os
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent

CORPUS = RACINE / "point3_transcription_verifiee" / "resultats" / "corpus.csv"
VIDEOS = Path(os.environ.get("EMOVID_DIR", r"C:\Users\zuowa\Downloads\PROJET_SID_2026\PROJET_SID_2026\movie")) / "video"
DATA = DOSSIER / "data"            # images et vecteurs (pas dans git)
IMAGES = DATA / "images"
RESULTATS = DOSSIER / "resultats"

EMOTIONS = ["amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"]
GRAINE = 42
NB_IMAGES = 16        # images par extrait
LARGEUR_MAX = 640     # images gardées à cette largeur au plus (assez pour les visages)
