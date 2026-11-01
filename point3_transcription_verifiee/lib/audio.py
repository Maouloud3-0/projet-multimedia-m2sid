"""Conversion des MP3 d'origine en WAV 16 kHz mono pour la transcription."""
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def chemin_mp3(ligne, donnees):
    return Path(donnees) / "audio" / ligne["film_id"] / (ligne["video_id"] + ".mp3")


def en_mono(audio):
    """Moyenne des canaux. Si les canaux s'annulent (phase inversée), on garde le plus fort."""
    niveaux = np.sqrt(np.mean(audio.astype(np.float64) ** 2, axis=0))
    mono = audio.mean(axis=1)
    if audio.shape[1] > 1 and np.sqrt(np.mean(mono.astype(np.float64) ** 2)) < 0.2 * niveaux.max():
        mono = audio[:, int(niveaux.argmax())]
    return mono


def preparer_audio(source, cible):
    """Extrait complet, sans découpe : on laisse la détection de parole aux étapes suivantes."""
    audio, frequence = sf.read(source, dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError(f"Audio vide : {source}")
    diviseur = math.gcd(frequence, 16000)
    signal = resample_poly(en_mono(audio), 16000 // diviseur, frequence // diviseur).astype(np.float32)
    cible = Path(cible)
    cible.parent.mkdir(parents=True, exist_ok=True)
    sf.write(cible, signal, 16000, subtype="PCM_16")
    return len(signal) / 16000
