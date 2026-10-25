"""Separation voix / musique avec Demucs.

Demucs n'est pas un filtre : c'est un reseau de neurones entraine sur des
morceaux dont les pistes sont separees (MUSDB18). Il predit un masque
temps-frequence qui garde la voix et rejette le reste.

Ce module est partage par 02_separer_voix.py et 08_experience_filtres.py.
"""

import torchaudio
from demucs.api import Separator, save_audio

import config


def charger_separateur():
    """Charge htdemucs une seule fois (environ 10 s)."""
    return Separator(model="htdemucs", device=config.DISPOSITIF, progress=False)


def separer(separateur, entree, sortie):
    """Ecrit la piste voix de entree dans sortie, en mono 16 kHz."""
    _, pistes = separateur.separate_audio_file(entree)
    voix = pistes["vocals"]

    if voix.shape[0] > 1:
        voix = voix.mean(dim=0, keepdim=True)
    voix = torchaudio.functional.resample(voix, separateur.samplerate, config.FREQUENCE)

    sortie.parent.mkdir(parents=True, exist_ok=True)
    save_audio(voix, sortie, samplerate=config.FREQUENCE)
