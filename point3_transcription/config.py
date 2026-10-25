"""Parametres du point 3 (transcription audio).

Tout se regle ici : un seul endroit a modifier.
Les dossiers de sortie sont crees automatiquement par les scripts.
"""

import os
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent

# --- Donnees fournies par l'enseignant -----------------------------------
# Le corpus n'est pas dans le depot (579 Mo). On indique son emplacement
# avec la variable d'environnement EMOVID_DIR, ou avec le chemin par defaut
# ci-dessous.
DONNEES = Path(os.environ.get(
    "EMOVID_DIR",
    r"C:\Users\konta\Desktop\Master SID\M2 SID\Projet Multi\PROJET_SID_2026\PROJET_SID_2026\movie",
))
AUDIO_MP3 = DONNEES / "audio"
ANNOTATIONS = DONNEES / "annotation"

# --- Dossiers produits ----------------------------------------------------
DATA = DOSSIER / "data"
WAV = DATA / "wav"                      # 01 : MP3 convertis en 16 kHz mono
VOIX = DATA / "voix"                    # 02 : piste voix isolee par Demucs
TRANSCRIPTIONS = DATA / "transcriptions"  # 03 : un JSON par extrait
RESULTATS = DOSSIER / "resultats"       # 04 : le tableau final

# --- Audio ----------------------------------------------------------------
FREQUENCE = 16000                       # Whisper travaille en 16 kHz mono

# --- Whisper --------------------------------------------------------------
MODELE = "large-v3-turbo"               # "small" ou "large-v3" pour comparer
DISPOSITIF = "cuda"                     # "cpu" si pas de carte graphique
# Dossier local du modele (facultatif). Utile quand les poids sont deja
# telecharges : WHISPER_MODELE=chemin/vers/le/dossier
CHEMIN_MODELE = os.environ.get("WHISPER_MODELE", MODELE)
LANGUE = None                           # None = detection automatique

# --- Filtrage des transcriptions -----------------------------------------
SEUIL_SANS_PAROLE = 0.6   # no_speech_prob au-dessus : segment douteux
SEUIL_LOGPROB = -1.0      # logprob moyen en dessous : segment douteux
SEUIL_COMPRESSION = 2.4   # au-dessus : le modele se repete en boucle
SEUIL_RATIO_PAROLE = 0.10 # en dessous : extrait considere comme sans parole
