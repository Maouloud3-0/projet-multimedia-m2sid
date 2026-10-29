"""Configuration commune du point 4 (audio et video)."""

import os
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent

DONNEES = Path(os.environ.get(
    "EMOVID_DIR",
    r"C:\Users\konta\Desktop\Master SID\M2 SID\Projet Multi\PROJET_SID_2026\PROJET_SID_2026\movie",
))
ANNOTATIONS = DONNEES / "annotation"
VIDEOS = DONNEES / "video"
AUDIOS = DONNEES / "audio"

POINT3 = RACINE / "point3_transcription"
WAV = POINT3 / "data" / "wav"
VOIX = POINT3 / "data" / "voix"

DATA = DOSSIER / "data"
RESULTATS = DOSSIER / "resultats"

EMOTIONS = ["amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"]
FREQUENCE_AUDIO = 16000
GRAINE = 42
