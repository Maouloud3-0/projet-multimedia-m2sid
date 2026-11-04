"""N1 : descripteurs acoustiques classiques, sans modèle pré-entraîné.

eGeMAPS v02 (openSMILE, 88 valeurs) : jeu de référence pour l'émotion dans la voix
    (hauteur, intensité, qualité de voix, rythme), résumé sur tout l'extrait.
librosa : MFCC et leurs dérivées, chroma, contraste spectral, centroïde, largeur de bande,
    rolloff, platitude, énergie, passages par zéro, force des attaques ; moyenne et écart-type,
    plus le tempo.
Puis régression logistique, C et poids des classes choisis en CV sur train (5 plis par film).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import opensmile
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
import utils

GRILLE = {"logisticregression__C": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
          "logisticregression__class_weight": ["balanced", None]}
FREQUENCE = 16000
LOCAL = threading.local()      # un extracteur openSMILE par fil d'exécution


def egemaps(y):
    if not hasattr(LOCAL, "smile"):
        LOCAL.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                      feature_level=opensmile.FeatureLevel.Functionals)
    return LOCAL.smile.process_signal(y, FREQUENCE).to_numpy().ravel()


def caracteristiques(video_id):
    y = utils.charger_audio(video_id, FREQUENCE)
    mfcc = librosa.feature.mfcc(y=y, sr=FREQUENCE, n_mfcc=20)
    trames = [mfcc, librosa.feature.delta(mfcc),
              librosa.feature.chroma_stft(y=y, sr=FREQUENCE),
              librosa.feature.spectral_contrast(y=y, sr=FREQUENCE),
              librosa.feature.spectral_centroid(y=y, sr=FREQUENCE),
              librosa.feature.spectral_bandwidth(y=y, sr=FREQUENCE),
              librosa.feature.spectral_rolloff(y=y, sr=FREQUENCE),
              librosa.feature.spectral_flatness(y=y),
              librosa.feature.rms(y=y),
              librosa.feature.zero_crossing_rate(y),
              librosa.onset.onset_strength(y=y, sr=FREQUENCE)[None]]
    resume = [np.concatenate([t.mean(1), t.std(1)]) for t in trames]
    tempo = librosa.feature.tempo(y=y, sr=FREQUENCE)
    return np.nan_to_num(np.concatenate([egemaps(y), *resume, tempo])).astype(np.float32)


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
                                   desc="descripteurs")))
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
