"""02 - Isoler la voix de la musique et des effets sonores (Demucs).

Dans un film, la musique et les bruits couvrent souvent les dialogues.
Demucs separe chaque extrait en deux pistes : "vocals" (les voix) et le reste.
On ne garde que la piste voix, qui sert ensuite a la transcription.

Le modele est charge une seule fois, puis on boucle sur les extraits.
Les extraits deja traites sont sautes : on peut relancer le script.

Usage :
    python 02_separer_voix.py
    python 02_separer_voix.py --limite 10      # essai sur 10 extraits
"""

import argparse
import time

import torch
import torchaudio
from demucs.api import Separator, save_audio

import config
import utils


def en_mono_16k(piste, frequence_source):
    """Passe la piste en mono 16 kHz (format attendu par Whisper)."""
    if piste.shape[0] > 1:
        piste = piste.mean(dim=0, keepdim=True)
    return torchaudio.functional.resample(piste, frequence_source, config.FREQUENCE)


def main():
    analyseur = argparse.ArgumentParser(description="Separation voix / reste")
    analyseur.add_argument("--limite", type=int, default=0,
                           help="ne traiter que les N premiers extraits")
    options = analyseur.parse_args()

    utils.preparer_dossiers()
    if not any(config.WAV.glob("*.wav")):
        print("Aucun WAV : lancer d'abord 01_convertir_audio.py")
        return

    print(f"Chargement de htdemucs sur {config.DISPOSITIF}...")
    separateur = Separator(model="htdemucs", device=config.DISPOSITIF, progress=False)

    liste = utils.extraits()
    if options.limite:
        liste = liste[:options.limite]

    depart = time.time()
    faits = 0
    for extrait in liste:
        entree = config.WAV / f"{extrait['video_id']}.wav"
        sortie = config.VOIX / f"{extrait['video_id']}.wav"
        if sortie.exists() or not entree.exists():
            continue

        _, pistes = separateur.separate_audio_file(entree)
        voix = en_mono_16k(pistes["vocals"], separateur.samplerate)
        save_audio(voix, sortie, samplerate=config.FREQUENCE)

        faits += 1
        if faits % 50 == 0:
            ecoule = time.time() - depart
            print(f"  {faits} extraits | {ecoule / 60:.1f} min "
                  f"| {ecoule / faits:.2f} s/extrait")

    print(f"{faits} pistes voix ->", config.VOIX)


if __name__ == "__main__":
    main()
