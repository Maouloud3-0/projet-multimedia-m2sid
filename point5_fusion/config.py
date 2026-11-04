"""Chemins et constantes du point 5 (fusion des trois modalités)."""

from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent

CORPUS = RACINE / "point3_transcription_verifiee" / "resultats" / "corpus.csv"
DATA = DOSSIER / "data"                  # vecteurs du texte (pas dans git)
RESULTATS = DOSSIER / "resultats"

# sorties du point 4 : probabilités (dans git) et vecteurs gelés (data/, régénérables)
PROBAS = {
    "video": RACINE / "point4_video" / "resultats",
    "texte": RACINE / "point4_texte" / "resultats",
    "audio": RACINE / "point4_audio" / "resultats",
}
DATA_VIDEO = RACINE / "point4_video" / "data"
DATA_AUDIO = RACINE / "point4_audio" / "data"

EMOTIONS = ["amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"]
GRAINE = 42
NB_TRANCHES = 16                         # images (vidéo) et tranches de temps (audio)
