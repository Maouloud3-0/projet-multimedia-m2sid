"""Vecteurs audio par des modèles pré-entraînés gelés (rien n'est appris ici).

  ast       AST AudioSet (MIT, 2021)                     sons, musique et voix (527 catégories AudioSet)
  clap      CLAP musique et parole (LAION, 2023)         son-texte, un vecteur par fenêtre de 10 s
  whisper   encodeur de Whisper large-v3 (OpenAI, 2023)  parole multilingue, robuste au bruit
  wavlm     WavLM large (Microsoft, 2022)                parole, auto-supervisé
  w2v2_emo  wav2vec2 large robust, affiné sur l'émotion (valence, éveil, dominance ; audEERING, 2022)
Sorties, en float16 :
  data/couches_<modele>.npy   (extraits, couches, 2 x dimension) : moyenne et écart-type dans le temps,
                              pour chaque couche (le choix de la couche se fait en CV dans 03)
  data/seg_<modele>.npy       (extraits, 16, dimension) : moyenne des couches sur 16 tranches de même durée
                              (entrée de la tête d'attention de 04) ; pas pour clap
  data/audioset.npy           (extraits, 2 x 527) : probabilités AudioSet d'AST, moyenne et maximum

python 02_encodeurs.py                      tous les modèles
python 02_encodeurs.py --modeles wavlm      un seul
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm
from transformers import (ASTFeatureExtractor, ASTForAudioClassification, AutoFeatureExtractor, AutoModel,
                          ClapModel, ClapProcessor, WhisperFeatureExtractor, WhisperModel)

import config
import utils

MODELES = {
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "clap": "laion/larger_clap_music_and_speech",
    "whisper": "openai/whisper-large-v3",
    "wavlm": "microsoft/wavlm-large",
    "w2v2_emo": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
}
FREQUENCES = {"clap": 48000}
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"


def telecharger(depot):
    """Fichiers du modèle dans .cache/modeles/ (copie simple, sans liens symboliques) ; un seul format de poids."""
    noms = [f for f in HfApi().list_repo_files(depot) if "/" not in f]
    # format standard seulement (pas les variantes .fp32 / .fp16 que certains dépôts ajoutent)
    standard = [f for f in noms if f.startswith(("model", "pytorch_model")) and ".fp" not in f]
    poids = [f for f in standard if ".safetensors" in f] or [f for f in standard if ".bin" in f]
    autres = [f for f in noms if f.endswith((".json", ".txt", ".model")) and f not in standard
              and ".fp" not in f]   # configuration et vocabulaire ; aucun code téléchargé
    dossier = config.MODELES_HF / depot.replace("/", "__")
    snapshot_download(depot, local_dir=dossier, allow_patterns=poids + autres)
    return dossier


def resumer(couches):
    """couches : (L, T, D) -> moyenne et écart-type dans le temps par couche, et 16 tranches de la moyenne des couches."""
    couches = couches.float()
    stats = torch.cat([couches.mean(1), couches.std(1, unbiased=False)], dim=-1)
    melange = couches.mean(0)
    seg = torch.stack([melange[a:b].mean(0) for a, b in utils.tranches(melange.shape[0])])
    return stats.cpu().numpy(), seg.cpu().numpy()


class Encodeur:
    def __init__(self, nom):
        self.nom = nom
        dossier = telecharger(MODELES[nom])
        if nom == "ast":
            self.extracteur = ASTFeatureExtractor.from_pretrained(dossier)
            self.modele = ASTForAudioClassification.from_pretrained(dossier)
        elif nom == "clap":
            self.extracteur = ClapProcessor.from_pretrained(dossier)
            self.modele = ClapModel.from_pretrained(dossier)
        elif nom == "whisper":
            self.extracteur = WhisperFeatureExtractor.from_pretrained(dossier)
            self.modele = WhisperModel.from_pretrained(dossier, torch_dtype=torch.float16).encoder
        else:
            self.extracteur = AutoFeatureExtractor.from_pretrained(dossier)
            self.modele = AutoModel.from_pretrained(dossier)          # WavLM et wav2vec2, sans leur tête
        self.modele = self.modele.to(APPAREIL).eval()
        self.frequence = FREQUENCES.get(nom, 16000)

    def fenetres(self, signal, secondes):
        pas = int(secondes * self.frequence)
        return [signal[d:d + pas] for d in range(0, len(signal), pas) if len(signal[d:d + pas]) >= self.frequence // 2] \
            or [signal[:pas]]

    @torch.no_grad()
    def __call__(self, signal):
        """Renvoie (statistiques par couche, tranches ou None, probabilités AudioSet ou None)."""
        if self.nom == "ast":
            couches, probas = [], []
            for morceau in self.fenetres(signal, 10.0):           # AST lit 10,24 s au plus
                entree = self.extracteur(morceau, sampling_rate=16000, return_tensors="pt").to(APPAREIL)
                sortie = self.modele(**entree, output_hidden_states=True)
                trames = len(morceau) // 160
                colonnes = max(1, min(101, (trames - 16) // 10 + 1))     # colonnes de patchs qui contiennent du son
                h = torch.stack(sortie.hidden_states)[:, 0, 2:]          # sans les deux jetons de résumé
                h = h.reshape(h.shape[0], 12, 101, -1).mean(1)[:, :colonnes]   # moyenne sur les fréquences
                couches.append(h)
                probas.append(sortie.logits[0].sigmoid())
            probas = torch.stack(probas)
            stats, seg = resumer(torch.cat(couches, dim=1))
            return stats, seg, torch.cat([probas.mean(0), probas.max(0).values]).cpu().numpy()
        if self.nom == "clap":
            vecteurs = []
            for morceau in self.fenetres(signal, 10.0):
                entree = self.extracteur(audio=morceau, sampling_rate=48000, return_tensors="pt").to(APPAREIL)
                vecteurs.append(self.modele.get_audio_features(**entree)[0])
            v = torch.stack(vecteurs)
            stats = torch.cat([v.mean(0), v.std(0, unbiased=False)])[None]
            return stats.cpu().numpy(), None, None
        if self.nom == "whisper":
            entree = self.extracteur(signal, sampling_rate=16000, return_tensors="pt").input_features
            sortie = self.modele(entree.to(APPAREIL, torch.float16), output_hidden_states=True)
            utiles = max(1, min(1500, int(np.ceil(len(signal) / 320))))      # une position toutes les 20 ms
            return (*resumer(torch.stack(sortie.hidden_states)[:, 0, :utiles]), None)
        entree = self.extracteur(signal, sampling_rate=self.frequence, return_tensors="pt").input_values
        with torch.autocast(APPAREIL, dtype=torch.float16, enabled=APPAREIL == "cuda"):
            sortie = self.modele(entree.to(APPAREIL), output_hidden_states=True)
        return (*resumer(torch.stack(sortie.hidden_states)[:, 0]), None)


def extraire(nom, corpus):
    sortie = config.DATA / f"couches_{nom}.npy"
    if sortie.exists():
        print(f"{nom} déjà fait")
        return
    debut = time.time()
    encodeur = Encodeur(nom)
    stats, segs, audioset = [], [], []
    ids = list(corpus["video_id"])
    with ThreadPoolExecutor(max_workers=4) as lecteurs:
        for d in tqdm(range(0, len(ids), 32), desc=nom):
            signaux = lecteurs.map(lambda v: utils.charger_audio(v, encodeur.frequence), ids[d:d + 32])
            for signal in signaux:
                s, g, a = encodeur(signal)
                stats.append(s.astype(np.float16))
                segs.append(g)
                audioset.append(a)
    np.save(sortie, np.stack(stats))
    if segs[0] is not None:
        np.save(config.DATA / f"seg_{nom}.npy", np.stack(segs).astype(np.float16))
    if audioset[0] is not None:
        np.save(config.DATA / "audioset.npy", np.stack(audioset).astype(np.float32))
    print(f"{nom} : {np.stack(stats).shape} en {(time.time() - debut) / 60:.1f} min")
    del encodeur
    torch.cuda.empty_cache()


def main():
    analyseur = argparse.ArgumentParser()
    analyseur.add_argument("--modeles", nargs="+", default=list(MODELES), choices=list(MODELES))
    options = analyseur.parse_args()
    corpus = utils.charger_corpus()
    print(f"appareil : {APPAREIL}")
    for nom in options.modeles:
        try:
            extraire(nom, corpus)
        except Exception as erreur:            # un modèle qui ne se charge pas ne bloque pas les autres
            print(f"{nom} : échec ({type(erreur).__name__}: {erreur})")


if __name__ == "__main__":
    main()
