"""Configuration du point 4 (texte)."""

from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent

CORPUS = RACINE / "point3_transcription_verifiee" / "resultats" / "corpus.csv"
DATA = DOSSIER / "data"            # cache des embeddings (pas dans git)
RESULTATS = DOSSIER / "resultats"

EMOTIONS = ["amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"]
GRAINE = 42