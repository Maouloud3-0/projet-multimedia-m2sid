"""Telechargement simple d'un modele Hugging Face dans .cache/.

Le cache standard de huggingface_hub pose probleme sur certaines machines
Windows. Ici on utilise seulement l'API HTTP et des fichiers ordinaires.
"""

import argparse
import json
from pathlib import Path
import urllib.request

RACINE = Path(__file__).resolve().parents[1] / ".cache" / "modeles"


def telecharger(depot):
    destination = RACINE / depot.replace("/", "__")
    destination.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"https://huggingface.co/api/models/{depot}") as flux:
        informations = json.load(flux)
    noms = [f["rfilename"] for f in informations["siblings"]]

    # Les poids safetensors suffisent : inutile de doubler avec le .bin.
    if any(nom.endswith(".safetensors") for nom in noms):
        noms = [nom for nom in noms if not nom.endswith("pytorch_model.bin")]
    noms = [nom for nom in noms
            if nom.endswith((".json", ".safetensors", ".bin", ".txt", ".model"))
            and "/" not in nom]

    for nom in noms:
        cible = destination / nom
        if cible.exists():
            continue
        print(f"  {nom}")
        with urllib.request.urlopen(
                f"https://huggingface.co/{depot}/resolve/main/{nom}") as flux:
            cible.write_bytes(flux.read())

    print("Modele pret :", destination)
    return destination


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser()
    analyseur.add_argument("depot")
    arguments = analyseur.parse_args()
    telecharger(arguments.depot)
