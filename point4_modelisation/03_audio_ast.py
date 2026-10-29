"""N2 audio : embeddings AST AudioSet, puis classifieurs simples."""

import argparse
import time

import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTModel

import config
import telecharger_hf
import utils

DEPOT = "MIT/ast-finetuned-audioset-10-10-0.4593"


def chemin_audio(video_id, source):
    if source == "voix":
        return config.VOIX / f"{video_id}.wav"
    wav = config.WAV / f"{video_id}.wav"
    return wav if wav.exists() else config.AUDIOS / video_id[:5] / f"{video_id}.mp3"


def main():
    analyseur = argparse.ArgumentParser()
    analyseur.add_argument("--source", choices=["brut", "voix"], default="brut")
    options = analyseur.parse_args()
    nom = f"audio_ast_{options.source}"

    utils.preparer_dossiers()
    donnees = utils.charger_extraits()
    cache = config.DATA / f"{nom}.npz"

    if cache.exists():
        ancien = np.load(cache, allow_pickle=True)
        X = ancien["X"]
    else:
        dossier = telecharger_hf.telecharger(DEPOT)
        extracteur = ASTFeatureExtractor.from_pretrained(dossier)
        modele = ASTModel.from_pretrained(dossier).eval()
        appareil = "cuda" if torch.cuda.is_available() else "cpu"
        modele.to(appareil)

        valeurs = []
        depart = time.time()
        for debut in range(0, len(donnees), 4):
            ids = donnees["video_id"].iloc[debut:debut + 4]
            audios = [librosa.load(chemin_audio(i, options.source),
                                   sr=config.FREQUENCE_AUDIO, mono=True)[0] for i in ids]
            entrees = extracteur(audios, sampling_rate=config.FREQUENCE_AUDIO,
                                 return_tensors="pt", padding=True)
            entrees = {cle: valeur.to(appareil) for cle, valeur in entrees.items()}
            with torch.inference_mode(), torch.autocast(
                    device_type=appareil, dtype=torch.float16,
                    enabled=appareil == "cuda"):
                cache_sequence = modele(**entrees).last_hidden_state
            valeurs.append(cache_sequence.mean(dim=1).float().cpu().numpy())
            if len(valeurs) % 100 == 0:
                print(f"  {debut + len(ids)}/{len(donnees)} | "
                      f"{(time.time() - depart) / 60:.1f} min")

        X = np.concatenate(valeurs).astype(np.float32)
        np.savez_compressed(cache, X=X, video_id=donnees["video_id"].to_numpy(dtype=str))

    utils.evaluer(X, donnees, nom)


if __name__ == "__main__":
    main()
