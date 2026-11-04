"""N1 : caractéristiques simples calculées sur les 16 images, sans modèle pré-entraîné.

Par image : moyenne et écart-type de la teinte, de la saturation et de la luminosité (HSV),
histogramme couleur (8 teintes x 4 saturations x 4 luminosités) et différence moyenne
avec l'image précédente (mouvement grossier). Par extrait : moyenne et écart-type sur les 16 images.
Puis régression logistique, C et poids des classes choisis en CV sur train (5 plis par film).
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
import utils

GRILLE = {"logisticregression__C": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
          "logisticregression__class_weight": ["balanced", None]}


def caracteristiques(video_id):
    par_image, precedente = [], None
    for k in range(config.NB_IMAGES):
        rgb = Image.open(config.IMAGES / video_id / f"{k:02d}.jpg").convert("RGB").resize((160, 90))
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255
        h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
        histogramme = np.histogramdd(np.stack([h, s, v], 1), bins=(8, 4, 4), range=((0, 1),) * 3)[0].ravel()
        mouvement = 0.0 if precedente is None else float(np.abs(hsv[..., 2] - precedente).mean())
        precedente = hsv[..., 2]
        par_image.append(np.concatenate([[h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std(), mouvement],
                                         histogramme / histogramme.sum()]))
    par_image = np.array(par_image)
    return np.concatenate([par_image.mean(0), par_image.std(0)])


def main():
    corpus = utils.charger_corpus()
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    films = corpus["film_id"].to_numpy()

    cache = config.DATA / "n1.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        with ThreadPoolExecutor(max_workers=6) as executeur:
            X = np.array(list(tqdm(executeur.map(caracteristiques, corpus["video_id"]), total=len(corpus),
                                   desc="images")))
        np.save(cache, X)

    debut = time.time()
    modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    resultat = utils.recherche(modele, GRILLE, X[train], y[train], films[train])
    meilleur = resultat.best_estimator_
    s = utils.scores(y[dev], meilleur.predict(X[dev]))
    utils.sauvegarder("n1", corpus, utils.probabilites_ordonnees(meilleur, X),
                      utils.probabilites_hors_pli(meilleur, X, y, films, train), y[dev], meilleur.predict(X[dev]))
    ligne = {"modele": "n1", "dimension": X.shape[1],
             "C": resultat.best_params_["logisticregression__C"],
             "class_weight": resultat.best_params_["logisticregression__class_weight"],
             "f1_cv": resultat.best_score_, "f1_dev": s["f1_macro"], "exactitude_dev": s["exactitude"]}
    pd.DataFrame([ligne]).to_csv(config.RESULTATS / "scores_n1.csv", index=False)
    print(f"N1 ({X.shape[1]} caractéristiques) : C={ligne['C']} {ligne['class_weight']}  F1 cv {ligne['f1_cv']:.3f}  "
          f"F1 dev {s['f1_macro']:.3f}  exactitude dev {s['exactitude']:.3f}  ({time.time() - debut:.0f} s)")


if __name__ == "__main__":
    main()
