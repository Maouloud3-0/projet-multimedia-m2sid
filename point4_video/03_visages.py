"""Visages : détection sur les 16 images, puis deux modèles d'expression faciale (gelés).

Sur chaque image (data/images, 640 px de large, sans recadrage) :
  - YuNet (OpenCV) détecte les visages, score >= 0,6 ; on garde les 2 plus grands ;
  - chaque visage est découpé avec une marge de 20 % puis passé à :
      vit  ViT entraîné sur FER2013        7 expressions + vecteur de 768
      hse  HSEmotion EfficientNet-B0,      8 expressions + valence et éveil + vecteur de 1280
           entraîné sur AffectNet (photos réelles en couleur)
Tout est gardé image par image et visage par visage : le regroupement par extrait
(moyenne, maximum, plus grand visage, attention) se fait dans les scripts suivants.

Sortie : data/visages.npz
  masque        (extraits, 16, 2)          visage présent dans cet emplacement
  nb_visages    (extraits, 16)             nombre de visages détectés dans l'image
  boites        (extraits, 16, 2, 6)       x, y, largeur, hauteur (en part de l'image), aire, score
  vit_probas    (extraits, 16, 2, 7)       vit_vecteurs (extraits, 16, 2, 768)
  hse_probas    (extraits, 16, 2, 8)       hse_va (extraits, 16, 2, 2)     hse_vecteurs (extraits, 16, 2, 1280)
"""

import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import timm
import torch
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification

import config
import utils

YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
VIT = "trpakov/vit-face-expression"
HSE = "enet_b0_8_va_mtl"
VISAGES_MAX = 2
SCORE_MIN = 0.6
MARGE = 0.2
TAILLE_MIN = 16          # côté minimal du visage découpé, en pixels

APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"
local = threading.local()


def detecteur():
    # un détecteur par fil d'exécution (l'objet OpenCV n'est pas partagé)
    if not hasattr(local, "yunet"):
        local.yunet = cv2.FaceDetectorYN.create(str(config.DATA / "yunet.onnx"), "", (640, 360), SCORE_MIN, 0.3, 50)
    return local.yunet


def detecter(video_id):
    """Pour un extrait : visages découpés et leurs boîtes, image par image."""
    sortie = []
    for k in range(config.NB_IMAGES):
        image = np.array(Image.open(config.IMAGES / video_id / f"{k:02d}.jpg").convert("RGB"))
        h, w, _ = image.shape
        yunet = detecteur()
        yunet.setInputSize((w, h))
        _, visages = yunet.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        visages = [] if visages is None else sorted(visages, key=lambda v: -v[2] * v[3])
        gardes = []
        for v in visages[:VISAGES_MAX]:
            x, y, lv, hv, score = float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[-1])
            x0, y0 = int(max(0, x - MARGE * lv)), int(max(0, y - MARGE * hv))
            x1, y1 = int(min(w, x + lv * (1 + MARGE))), int(min(h, y + hv * (1 + MARGE)))
            if x1 - x0 >= TAILLE_MIN and y1 - y0 >= TAILLE_MIN:
                gardes.append((image[y0:y1, x0:x1], [x / w, y / h, lv / w, hv / h, lv * hv / (w * h), score]))
        sortie.append((len(visages), gardes))
    return video_id, sortie


def charger_modeles():
    if not (config.DATA / "yunet.onnx").exists():
        urllib.request.urlretrieve(YUNET, config.DATA / "yunet.onnx")
    vit_proc = AutoImageProcessor.from_pretrained(VIT)
    vit = ViTForImageClassification.from_pretrained(VIT).to(APPAREIL).eval()
    # Le fichier HSEmotion a été enregistré avec timm 0.9 : on recharge ses poids
    # dans une EfficientNet-B0 de la version de timm installée (mêmes poids, mêmes noms).
    ancien = EmotiEffLibRecognizer(engine="torch", model_name=HSE, device="cpu")
    hse = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    hse.load_state_dict(ancien.model.state_dict())
    hse = hse.to(APPAREIL).eval()
    poids = torch.tensor(ancien.classifier_weights, device=APPAREIL)
    biais = torch.tensor(ancien.classifier_bias, device=APPAREIL)
    hse_prep = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return vit_proc, vit, hse, poids, biais, hse_prep


def extraire(ids, sortie):
    n = len(ids)
    vit_proc, vit, hse, poids, biais, hse_prep = charger_modeles()
    forme = (n, config.NB_IMAGES, VISAGES_MAX)
    masque = np.zeros(forme, bool)
    nb_visages = np.zeros(forme[:2], np.uint8)
    boites = np.zeros(forme + (6,), np.float32)
    vit_probas = np.zeros(forme + (7,), np.float16)
    vit_vecteurs = np.zeros(forme + (768,), np.float16)
    hse_probas = np.zeros(forme + (8,), np.float16)
    hse_va = np.zeros(forme + (2,), np.float16)
    hse_vecteurs = np.zeros(forme + (1280,), np.float16)

    debut = time.time()
    barre = tqdm(total=n, desc="extraits")
    with ThreadPoolExecutor(max_workers=4) as executeur, torch.no_grad():
        for depart in range(0, n, 64):            # par paquets de 64 extraits, pour borner la mémoire
            for i, (video_id, images) in enumerate(executeur.map(detecter, ids[depart:depart + 64]), start=depart):
                barre.update()
                decoupes, places = [], []
                for k, (nombre, gardes) in enumerate(images):
                    nb_visages[i, k] = min(nombre, 255)
                    for j, (visage, boite) in enumerate(gardes):
                        masque[i, k, j] = True
                        boites[i, k, j] = boite
                        decoupes.append(visage)
                        places.append((k, j))
                if not decoupes:
                    continue
                k, j = np.array(places).T
                resultat = vit(**vit_proc(images=decoupes, return_tensors="pt").to(APPAREIL), output_hidden_states=True)
                vit_probas[i, k, j] = resultat.logits.softmax(-1).cpu().numpy()
                vit_vecteurs[i, k, j] = resultat.hidden_states[-1][:, 0].cpu().numpy()
                entree = torch.stack([hse_prep(Image.fromarray(v)) for v in decoupes]).to(APPAREIL)
                vecteurs = hse(entree)
                scores = vecteurs @ poids.T + biais                  # 8 expressions puis valence, éveil
                hse_probas[i, k, j] = scores[:, :8].softmax(-1).cpu().numpy()
                hse_va[i, k, j] = scores[:, 8:].cpu().numpy()
                hse_vecteurs[i, k, j] = vecteurs.cpu().numpy()
    barre.close()

    np.savez(sortie, video_id=np.array(ids), masque=masque, nb_visages=nb_visages, boites=boites,
             vit_probas=vit_probas, vit_vecteurs=vit_vecteurs,
             hse_probas=hse_probas, hse_va=hse_va, hse_vecteurs=hse_vecteurs)
    print(f"{(time.time() - debut) / 60:.1f} min | extraits avec au moins un visage : {masque.any(axis=(1, 2)).sum()} / {n} "
          f"| images avec visage : {masque[..., 0].mean():.0%}")
    for nom, tableau in [("vit_vecteurs", vit_vecteurs), ("hse_vecteurs", hse_vecteurs)]:
        if not np.isfinite(tableau).all():
            print(f"attention : valeurs non finies dans {nom}")


def main():
    extraire(utils.charger_corpus()["video_id"].tolist(), config.DATA / "visages.npz")


if __name__ == "__main__":
    main()
