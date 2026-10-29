"""N2 video : 8 images par extrait, embeddings CLIP moyennes."""

import time

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

import config
import telecharger_hf
import utils

DEPOT = "openai/clip-vit-base-patch16"


def images_video(chemin, nombre_images=8):
    """Lit la video sequentiellement et conserve seulement les images choisies."""
    lecteur = cv2.VideoCapture(str(chemin))
    total = int(lecteur.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = set(np.linspace(0, max(total - 1, 0), nombre_images).astype(int))
    images = []
    position = 0
    while True:
        lu, image = lecteur.read()
        if not lu:
            break
        if position in positions:
            images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        position += 1
    lecteur.release()
    return images


def main():
    utils.preparer_dossiers()
    donnees = utils.charger_extraits()
    cache = config.DATA / "video_clip.npz"

    if cache.exists():
        X = np.load(cache, allow_pickle=True)["X"]
    else:
        dossier = telecharger_hf.telecharger(DEPOT)
        processeur = CLIPProcessor.from_pretrained(dossier)
        modele = CLIPModel.from_pretrained(dossier).eval()
        appareil = "cuda" if torch.cuda.is_available() else "cpu"
        modele.to(appareil)

        valeurs = []
        depart = time.time()
        for numero, ligne in enumerate(donnees.itertuples(index=False), start=1):
            chemin = config.VIDEOS / ligne.film_id / f"{ligne.video_id}.mp4"
            images = images_video(chemin)
            if not images:
                valeurs.append(np.zeros(512, dtype=np.float32))
                continue
            entrees = processeur(images=images, return_tensors="pt")
            entrees = {cle: valeur.to(appareil) for cle, valeur in entrees.items()}
            with torch.inference_mode(), torch.autocast(
                    device_type=appareil, dtype=torch.float16,
                    enabled=appareil == "cuda"):
                vecteurs = modele.get_image_features(**entrees)
            valeurs.append(vecteurs.mean(dim=0).float().cpu().numpy())
            if numero % 100 == 0:
                print(f"  {numero}/{len(donnees)} | {(time.time() - depart) / 60:.1f} min")

        X = np.stack(valeurs).astype(np.float32)
        np.savez_compressed(cache, X=X, video_id=donnees["video_id"].to_numpy(dtype=str))

    utils.evaluer(X, donnees, "video_clip")


if __name__ == "__main__":
    main()
