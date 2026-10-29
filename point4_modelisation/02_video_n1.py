"""N1 video : annotations visuelles + couleurs de quelques images."""

from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np

import config
import utils


def statistiques_image(image):
    """Couleurs moyennes et histogrammes d'une image reduite."""
    image = cv2.resize(image, (160, 90))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    valeurs = [*rgb.mean(axis=(0, 1)), *rgb.std(axis=(0, 1))]
    for canal in range(3):
        histogramme = cv2.calcHist([rgb], [canal], None, [8], [0, 1]).flatten()
        valeurs.extend(histogramme / histogramme.sum())

    teinte = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    valeurs.extend(teinte / teinte.sum())
    valeurs.extend([hsv[:, :, 1].mean() / 255, hsv[:, :, 2].mean() / 255])
    return np.array(valeurs, dtype=np.float32)


def caracteristiques(ligne):
    """Moyenne les statistiques de 4 images, puis ajoute les annotations JSON."""
    chemin = config.VIDEOS / ligne.film_id / f"{ligne.video_id}.mp4"
    lecteur = cv2.VideoCapture(str(chemin))
    nombre = int(lecteur.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    for position in np.linspace(0.1, 0.9, 4):
        lecteur.set(cv2.CAP_PROP_POS_FRAMES, int(position * max(nombre - 1, 0)))
        lu, image = lecteur.read()
        if lu:
            images.append(statistiques_image(image))
    lecteur.release()

    visuel = np.mean(images, axis=0) if images else np.zeros(44, dtype=np.float32)
    return np.concatenate([
        [ligne.brightness, ligne.colorfulness, ligne.hue / 360.0, len(images) / 4],
        visuel,
    ]).astype(np.float32)


def main():
    utils.preparer_dossiers()
    donnees = utils.charger_extraits()
    cache = config.DATA / "video_n1.npz"

    if cache.exists():
        ancien = np.load(cache, allow_pickle=True)
        X = ancien["X"]
        assert list(ancien["video_id"]) == list(donnees["video_id"])
    else:
        print(f"Lecture de 4 images dans {len(donnees)} videos...")
        depart = time.time()
        with ThreadPoolExecutor(max_workers=4) as executeur:
            valeurs = list(executeur.map(caracteristiques,
                                         donnees.itertuples(index=False)))
        X = np.stack(valeurs)
        np.savez_compressed(cache, X=X, video_id=donnees["video_id"].to_numpy(dtype=str))
        print(f"{X.shape[1]} caracteristiques | {(time.time() - depart) / 60:.1f} min")

    utils.evaluer(X, donnees, "video_n1")


if __name__ == "__main__":
    main()
