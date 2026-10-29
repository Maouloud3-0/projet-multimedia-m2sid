"""N2 audio : encodeur wav2vec2 deja adapte a l'emotion (SUPERB ER)."""

import time

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

import config
import telecharger_hf
import utils

DEPOT = "superb/wav2vec2-base-superb-er"


def main():
    utils.preparer_dossiers()
    donnees = utils.charger_extraits()
    cache = config.DATA / "audio_wav2vec2_emotion.npz"

    if cache.exists():
        X = np.load(cache, allow_pickle=True)["X"]
    else:
        dossier = telecharger_hf.telecharger(DEPOT)
        extracteur = Wav2Vec2FeatureExtractor.from_pretrained(dossier)
        modele = Wav2Vec2ForSequenceClassification.from_pretrained(
            dossier, output_hidden_states=True).eval()
        appareil = "cuda" if torch.cuda.is_available() else "cpu"
        modele.to(appareil)

        valeurs = []
        depart = time.time()
        for debut in range(0, len(donnees), 4):
            ids = donnees["video_id"].iloc[debut:debut + 4]
            audios = []
            for video_id in ids:
                wav = config.WAV / f"{video_id}.wav"
                chemin = wav if wav.exists() else config.AUDIOS / video_id[:5] / f"{video_id}.mp3"
                audios.append(librosa.load(chemin, sr=config.FREQUENCE_AUDIO, mono=True)[0])
            entrees = extracteur(audios, sampling_rate=config.FREQUENCE_AUDIO,
                                 return_tensors="pt", padding=True)
            entrees = {cle: valeur.to(appareil) for cle, valeur in entrees.items()}
            with torch.inference_mode(), torch.autocast(
                    device_type=appareil, dtype=torch.float16,
                    enabled=appareil == "cuda"):
                sortie = modele(**entrees)
            moyenne = sortie.hidden_states[-1].mean(dim=1)
            # On conserve aussi les scores du classifieur emotion du modele.
            vecteurs = torch.cat([moyenne, sortie.logits], dim=1)
            valeurs.append(vecteurs.float().cpu().numpy())
            if len(valeurs) % 100 == 0:
                print(f"  {debut + len(ids)}/{len(donnees)} | "
                      f"{(time.time() - depart) / 60:.1f} min")

        X = np.concatenate(valeurs).astype(np.float32)
        np.savez_compressed(cache, X=X, video_id=donnees["video_id"].to_numpy(dtype=str))

    utils.evaluer(X, donnees, "audio_wav2vec2_emotion")


if __name__ == "__main__":
    main()
