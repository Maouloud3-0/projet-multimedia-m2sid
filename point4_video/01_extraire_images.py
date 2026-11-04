"""Extraction des images : 16 images par extrait, réparties uniformément sur la partie utile.

Partie utile = du début à fin_utile (écran de fin publicitaire retiré, voir 00).
On prend le milieu de 16 intervalles égaux, ce qui évite la toute première et la
toute dernière image (souvent un fondu ou un changement de plan).
Images gardées en couleur, largeur 640 px au plus, sans recadrage : chaque encodeur
fait ensuite son propre prétraitement, et les visages sont cherchés sur l'image entière.
Sortie : data/images/<video_id>/00.jpg ... 15.jpg
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


def extraire(argument):
    video_id, fin_utile = argument
    dossier = config.IMAGES / video_id
    if len(list(dossier.glob("*.jpg"))) == config.NB_IMAGES:
        return video_id, False
    cibles = (np.arange(config.NB_IMAGES) + 0.5) / config.NB_IMAGES * fin_utile
    gardees, precedente = {}, None
    with av.open(str(config.VIDEOS / video_id[:5] / f"{video_id}.mp4")) as video:
        flux = video.streams.video[0]
        flux.thread_type = "AUTO"
        for image in video.decode(flux):
            t = float(image.pts * flux.time_base)
            # première image à partir de chaque instant visé
            for k in np.where((cibles <= t) & ~np.isin(np.arange(config.NB_IMAGES), list(gardees)))[0]:
                gardees[k] = image.to_image()
            precedente = image
            if len(gardees) == config.NB_IMAGES:
                break
    for k in range(config.NB_IMAGES):           # extrait plus court que prévu : dernière image lue
        gardees.setdefault(k, precedente.to_image())
    dossier.mkdir(parents=True, exist_ok=True)
    for k, image in gardees.items():
        if image.width > config.LARGEUR_MAX:
            image = image.resize((config.LARGEUR_MAX, round(image.height * config.LARGEUR_MAX / image.width)))
        image.save(dossier / f"{k:02d}.jpg", quality=90)
    return video_id, True


def main():
    corpus = utils.charger_corpus()
    fin = pd.read_csv(config.RESULTATS / "fin_pub.csv").set_index("video_id")["fin_utile"]
    arguments = [(v, float(fin[v])) for v in corpus["video_id"]]
    debut = time.time()
    with ProcessPoolExecutor(max_workers=6) as executeur:
        taches = executeur.map(extraire, arguments, chunksize=8)
        faits = sum(nouveau for _, nouveau in tqdm(taches, total=len(arguments), desc="extraits"))
    print(f"{faits} extraits traités, {len(arguments) - faits} déjà présents ({(time.time() - debut) / 60:.0f} min)")


if __name__ == "__main__":
    main()
