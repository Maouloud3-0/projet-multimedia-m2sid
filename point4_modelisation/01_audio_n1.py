"""N1 audio : caracteristiques acoustiques simples + classifieurs classiques."""

from concurrent.futures import ThreadPoolExecutor
import time

import librosa
import numpy as np

import config
import utils


def chemin_audio(video_id):
    wav = config.WAV / f"{video_id}.wav"
    if wav.exists():
        return wav
    return config.AUDIOS / video_id[:5] / f"{video_id}.mp3"


def moyenne_ecart_type(tableau):
    return np.concatenate([tableau.mean(axis=1), tableau.std(axis=1)])


def caracteristiques(video_id):
    """Resume spectral et energetique d'un extrait, sans modele profond."""
    y, sr = librosa.load(chemin_audio(video_id), sr=config.FREQUENCE_AUDIO, mono=True)
    if len(y) < sr // 4:
        y = np.pad(y, (0, sr // 4 - len(y)))

    blocs = [
        moyenne_ecart_type(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)),
        moyenne_ecart_type(librosa.feature.chroma_stft(y=y, sr=sr)),
        moyenne_ecart_type(librosa.feature.spectral_centroid(y=y, sr=sr)),
        moyenne_ecart_type(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        moyenne_ecart_type(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        moyenne_ecart_type(librosa.feature.spectral_contrast(y=y, sr=sr)),
        moyenne_ecart_type(librosa.feature.zero_crossing_rate(y)),
        moyenne_ecart_type(librosa.feature.rms(y=y)),
        np.array([librosa.beat.tempo(y=y, sr=sr)[0]], dtype=np.float32),
    ]
    return np.nan_to_num(np.concatenate(blocs)).astype(np.float32)


def main():
    utils.preparer_dossiers()
    donnees = utils.charger_extraits()
    cache = config.DATA / "audio_n1.npz"

    if cache.exists():
        ancien = np.load(cache, allow_pickle=True)
        X = ancien["X"]
        assert list(ancien["video_id"]) == list(donnees["video_id"])
    else:
        print(f"Extraction de {len(donnees)} extraits audio...")
        depart = time.time()
        with ThreadPoolExecutor(max_workers=4) as executeur:
            valeurs = list(executeur.map(caracteristiques, donnees["video_id"]))
        X = np.stack(valeurs)
        np.savez_compressed(cache, X=X, video_id=donnees["video_id"].to_numpy(dtype=str))
        print(f"{X.shape[1]} caracteristiques | {(time.time() - depart) / 60:.1f} min")

    utils.evaluer(X, donnees, "audio_n1")


if __name__ == "__main__":
    main()
