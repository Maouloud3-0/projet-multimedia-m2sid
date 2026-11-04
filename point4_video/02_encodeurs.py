"""Vecteurs d'images par des modèles pré-entraînés gelés (rien n'est appris ici).

Chaque modèle donne un vecteur par image (16 par extrait), gardé tel quel :
le regroupement des images (moyenne, écart-type, attention) se fait plus tard.
  clip_b    CLIP ViT-B/16 (OpenAI, 2021)          image-texte, petit modèle
  clip_l    CLIP ViT-L/14 (OpenAI, 2021)          même famille, plus grand
  clip_h    OpenCLIP ViT-H/14 (LAION-2B, 2022)    même famille, encore plus grand
  siglip2   SigLIP 2 so400m/14, 384 px (2025)     image-texte, génération récente
  pe        Perception Encoder Core L/14, 336 px (Meta, 2025)   image-texte, génération récente
  dinov2    DINOv2 base (2023)                    images seules, sans texte
  videomae  VideoMAE base, Kinetics-400 (2022)    les 16 images ensemble : mouvement (un vecteur par extrait)
Sortie : data/emb_<modele>.npy, forme (extraits, images, dimension), en float16.

python 02_encodeurs.py                      tous les modèles
python 02_encodeurs.py --modeles clip_l     un seul
"""

import argparse
import time

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor,
                          VideoMAEForVideoClassification)

import config
import utils

MODELES = {
    "clip_b": "openai/clip-vit-base-patch16",
    "clip_l": "openai/clip-vit-large-patch14",
    "clip_h": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "siglip2": "google/siglip2-so400m-patch14-384",
    "pe": "hf-hub:timm/PE-Core-L-14-336",          # chargé par OpenCLIP
    "dinov2": "facebook/dinov2-base",
    "videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
}
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"
TYPE = torch.float16 if APPAREIL == "cuda" else torch.float32


def images(video_id):
    return [Image.open(config.IMAGES / video_id / f"{k:02d}.jpg").convert("RGB") for k in range(config.NB_IMAGES)]


def charger(nom):
    chemin = MODELES[nom]
    if nom.startswith("clip"):
        return CLIPModel.from_pretrained(chemin, dtype=TYPE), CLIPProcessor.from_pretrained(chemin)
    if nom == "pe":
        modele, _, pretraitement = open_clip.create_model_and_transforms(chemin)
        return modele, pretraitement
    if nom == "siglip2":
        return AutoModel.from_pretrained(chemin, dtype=TYPE), AutoProcessor.from_pretrained(chemin)
    if nom == "videomae":
        return VideoMAEForVideoClassification.from_pretrained(chemin, dtype=TYPE), AutoImageProcessor.from_pretrained(chemin)
    return AutoModel.from_pretrained(chemin, dtype=TYPE), AutoImageProcessor.from_pretrained(chemin)


@torch.no_grad()
def encoder(nom, modele, processeur, liste):
    if nom == "pe":
        entree = torch.stack([processeur(image) for image in liste]).to(APPAREIL)
        with torch.autocast(device_type=APPAREIL, dtype=torch.float16, enabled=APPAREIL == "cuda"):
            return modele.encode_image(entree, normalize=True).float()
    if nom == "videomae":
        entree = processeur(liste, return_tensors="pt")["pixel_values"].to(APPAREIL, TYPE)
        cachee = modele(pixel_values=entree, output_hidden_states=True).hidden_states[-1]
        return cachee.mean(dim=1).float()                       # (1, d) : un vecteur pour l'extrait
    entree = processeur(images=liste, return_tensors="pt")["pixel_values"].to(APPAREIL, TYPE)
    if nom == "dinov2":
        return modele(pixel_values=entree).pooler_output.float()
    vecteurs = modele.get_image_features(pixel_values=entree).float()
    return torch.nn.functional.normalize(vecteurs, dim=-1)       # CLIP, SigLIP : vecteurs normés


def main():
    analyseur = argparse.ArgumentParser()
    analyseur.add_argument("--modeles", nargs="+", default=list(MODELES))
    options = analyseur.parse_args()
    ids = utils.charger_corpus()["video_id"].tolist()

    for nom in options.modeles:
        sortie = config.DATA / f"emb_{nom}.npy"
        if sortie.exists():
            print(f"{nom} déjà calculé")
            continue
        debut = time.time()
        modele, processeur = charger(nom)
        modele.to(APPAREIL).eval()
        vecteurs = [encoder(nom, modele, processeur, images(video_id)).cpu().numpy().astype(np.float16)
                    for video_id in tqdm(ids, desc=nom)]
        vecteurs = np.stack(vecteurs)
        if not np.isfinite(vecteurs).all():     # float16 qui déborde : on ne garde pas un fichier faux
            print(f"{nom} : valeurs non finies, fichier non enregistré")
            continue
        np.save(sortie, vecteurs)
        print(f"{nom} : {vecteurs.shape} en {(time.time() - debut) / 60:.1f} min", flush=True)
        del modele
        if APPAREIL == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
