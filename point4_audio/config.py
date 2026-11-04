"""Configuration du point 4 (audio)."""

import os
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent

CORPUS = RACINE / "point3_transcription_verifiee" / "resultats" / "corpus.csv"
FIN_PUB = RACINE / "point4_video" / "resultats" / "fin_pub.csv"   # écrans de fin Movieclips (point4_video/00)
AUDIOS = Path(os.environ.get(
    "EMOVID_DIR",
    r"C:\Users\konta\Desktop\Master SID\M2 SID\Projet Multi\PROJET_SID_2026\PROJET_SID_2026\movie")) / "audio"
DATA = DOSSIER / "data"            # descripteurs et vecteurs (pas dans git)
MODELES_HF = RACINE / ".cache" / "modeles"
RESULTATS = DOSSIER / "resultats"

EMOTIONS = ["amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"]
GRAINE = 42
DUREE_MAX = 30.0      # secondes gardées au plus (les extraits font de 4 à 30 s, 6,4 s en médiane)
NB_SEGMENTS = 16      # tranches de même durée sur la partie utile, comme les 16 images de la vidéo
