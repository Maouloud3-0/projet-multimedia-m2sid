"""Telecharger un modele Whisper au format CTranslate2, sans huggingface_hub.

Le cache de huggingface_hub cree des dossiers temporaires et des liens
symboliques qui echouent sur certaines machines Windows. Ce script
telecharge donc simplement les fichiers avec urllib dans un dossier local,
que 03_transcrire.py sait utiliser via la variable WHISPER_MODELE.

Usage :
    python telecharger_modele.py small
    python telecharger_modele.py large-v3-turbo
"""

import pathlib
import sys
import urllib.request

# Nom court -> depot Hugging Face
DEPOTS = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

FICHIERS = ["config.json", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]

RACINE = pathlib.Path(__file__).resolve().parents[1] / ".cache"


def telecharger(nom):
    if nom not in DEPOTS:
        sys.exit(f"Modele inconnu : {nom}. Choix : {', '.join(DEPOTS)}")

    depot = DEPOTS[nom]
    destination = RACINE / f"whisper-{nom}"
    destination.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{depot}/resolve/main/"

    for fichier in ["model.bin"] + FICHIERS:
        cible = destination / fichier
        if cible.exists():
            print(f"  deja present : {fichier}")
            continue
        print(f"  telechargement : {fichier}")
        with urllib.request.urlopen(base + fichier) as flux:
            cible.write_bytes(flux.read())
        print(f"    {cible.stat().st_size / 1e6:.1f} Mo")

    print("\nModele pret. Pour l'utiliser :")
    print(f'  set WHISPER_MODELE={destination}')


if __name__ == "__main__":
    telecharger(sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo")
