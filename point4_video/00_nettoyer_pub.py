"""Nettoyage : repérer l'écran de fin publicitaire (Fandango Movieclips) à la fin des extraits.

Certains extraits se terminent par l'écran de fin de la chaîne YouTube d'origine
(vignettes d'autres vidéos, « CLICK ABOVE TO WATCH MORE CLIPS »). Ce n'est pas du
film. On le reconnaît à son bandeau du haut : fond gris foncé à gauche, bouton bleu
« SUBSCRIBE » à droite. Il dure jusqu'à la dernière image.

Règles, sur chaque image :
  1. le tiers droit du bandeau est bleu vif sur au moins un quart de sa surface ;
  2. la partie gauche du bandeau est gris foncé uniforme sur au moins 40 % ;
  3. 1 et 2 tiennent sans interruption jusqu'à la dernière image (au moins 3 images) ;
  4. le bandeau ressemble à celui d'un écran de fin connu (corrélation >= 0,6),
     pour écarter quelques scènes bleutées qui passent 1 à 3.
Les étiquettes d'émotion ne sont pas utilisées.

Sortie : resultats/fin_pub.csv, avec fin_utile = instant où arrêter la vidéo.
"""

import time
from concurrent.futures import ProcessPoolExecutor

import av
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
import utils

REFERENCES = ["00240_clip_054", "00180_clip_017"]   # un écran de fin par version (récente, ancienne)
BANDEAU = 1 / 7          # le bandeau occupe environ le septième supérieur de l'image
TIERS_DROIT = 2 / 3
PARTIE_GAUCHE = 0.6
PART_BLEU = 0.25
PART_GRIS = 0.40
IMAGES_MIN = 3
RESSEMBLANCE = 0.6
MARGE = 0.1              # on arrête la vidéo 0,1 s avant l'écran


def lire(video_id):
    with av.open(str(config.VIDEOS / video_id[:5] / f"{video_id}.mp4")) as video:
        flux = video.streams.video[0]
        flux.thread_type = "AUTO"
        images = list(video.decode(flux))
        return (np.array([float(i.pts * flux.time_base) for i in images]),
                [i.to_ndarray(format="rgb24") for i in images])


def regles_1_2(image):
    h, w, _ = image.shape
    haut = image[:int(BANDEAU * h)].astype(np.int16)
    droite = haut[:, int(TIERS_DROIT * w):]
    gauche = haut[:, :int(PARTIE_GAUCHE * w)]
    bleu_vif = (droite[..., 2] > 170) & (droite[..., 0] < 110) & (droite[..., 1] > 90)
    gris_fonce = (gauche.max(-1) < 90) & (gauche.max(-1) - gauche.min(-1) < 25)
    return bleu_vif.mean() >= PART_BLEU and gris_fonce.mean() >= PART_GRIS


def bandeau(image):
    return np.asarray(Image.fromarray(image[:int(BANDEAU * image.shape[0])]).resize((96, 12)), dtype=np.float32)


def correlation(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-6))


def analyser(video_id, modeles):
    instants, images = lire(video_id)
    fin = instants[-1] + np.median(np.diff(instants))
    marques = [regles_1_2(i) for i in images]
    debut = len(marques)
    while debut > 0 and marques[debut - 1]:        # règle 3
        debut -= 1
    ligne = {"video_id": video_id, "duree": round(fin, 2), "debut_pub": np.nan, "duree_pub": 0.0,
             "fin_utile": round(fin, 2)}
    if len(marques) - debut >= IMAGES_MIN:
        ressemblance = max(correlation(bandeau(i), m) for i in images[debut:] for m in modeles)
        if ressemblance >= RESSEMBLANCE:            # règle 4
            t = instants[debut]
            ligne.update(debut_pub=round(t, 2), duree_pub=round(fin - t, 2), fin_utile=round(max(0.0, t - MARGE), 2))
    return ligne


def main():
    corpus = utils.charger_corpus()
    modeles = [bandeau(lire(v)[1][-1]) for v in REFERENCES]
    debut = time.time()
    with ProcessPoolExecutor(max_workers=6) as executeur:
        taches = executeur.map(analyser, corpus["video_id"], [modeles] * len(corpus), chunksize=8)
        lignes = list(tqdm(taches, total=len(corpus), desc="extraits"))   # environ 10 min
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(config.RESULTATS / "fin_pub.csv", index=False)

    avec = tableau[tableau["debut_pub"].notna()].merge(corpus[["video_id", "partition"]])
    print(f"Écran de fin trouvé sur {len(avec)} extraits sur {len(tableau)} ({(time.time() - debut) / 60:.0f} min)")
    print(f"par partition : {avec['partition'].value_counts().to_dict()}")
    print(f"durée de l'écran (s) : min {avec['duree_pub'].min()}, médiane {avec['duree_pub'].median()}, "
          f"max {avec['duree_pub'].max()}")


if __name__ == "__main__":
    main()
