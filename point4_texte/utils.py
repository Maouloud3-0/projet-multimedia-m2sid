"""Fonctions communes aux scripts du point 4."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

import config


def charger_corpus():
    corpus = pd.read_csv(config.CORPUS, dtype={"film_id": str}, keep_default_na=False)
    config.DATA.mkdir(exist_ok=True)
    config.RESULTATS.mkdir(exist_ok=True)
    return corpus


def textes(corpus, entree):
    if entree == "description":
        return corpus["caption"]
    if entree == "dialogue":
        return corpus["dialogue"]
    return corpus["caption"] + " " + corpus["dialogue"]


def scores(y_vrai, y_predit):
    return {"f1_macro": f1_score(y_vrai, y_predit, average="macro"),
            "exactitude": accuracy_score(y_vrai, y_predit)}


def recherche(modele, grille, X, y, films):
    # 5 plis par film : un film n'est jamais à la fois dans l'apprentissage et l'évaluation
    grille_cv = GridSearchCV(modele, grille, scoring="f1_macro",
                             cv=GroupKFold(n_splits=5), n_jobs=-1)
    return grille_cv.fit(X, y, groups=films)


def probabilites_hors_pli(modele, X, y, films, train):
    # chaque extrait de train est prédit par le modèle des 4 autres plis ; NaN hors train
    sorties = cross_val_predict(modele, X[train], y[train], groups=films[train],
                                cv=GroupKFold(n_splits=5), method="predict_proba", n_jobs=-1)
    probas = np.full((len(y), len(config.EMOTIONS)), np.nan)
    probas[train] = sorties   # classes triées par ordre alphabétique = config.EMOTIONS
    return probas


def probabilites_ordonnees(modele, X):
    # remet les colonnes dans l'ordre de config.EMOTIONS
    sorties = modele.predict_proba(X)
    probas = np.zeros((X.shape[0], len(config.EMOTIONS)))
    for colonne, classe in enumerate(modele.classes_):
        probas[:, config.EMOTIONS.index(classe)] = sorties[:, colonne]
    return probas


def sauvegarder(nom, corpus, probas, hors_pli, y_dev, y_predit_dev):
    np.savez_compressed(config.RESULTATS / f"probas_{nom}.npz",
                        video_id=corpus["video_id"].to_numpy(dtype=str),
                        partition=corpus["partition"].to_numpy(dtype=str),
                        probabilites=probas, probabilites_hors_pli=hors_pli,
                        emotions=np.array(config.EMOTIONS))
    matrice = confusion_matrix(y_dev, y_predit_dev, labels=config.EMOTIONS)
    figure, axe = plt.subplots(figsize=(7, 6))
    axe.imshow(matrice, cmap="Blues")
    axe.set_xticks(range(8), config.EMOTIONS, rotation=45, ha="right")
    axe.set_yticks(range(8), config.EMOTIONS)
    axe.set_xlabel("Prédiction")
    axe.set_ylabel("Émotion réelle")
    axe.set_title(f"{nom} (dev)")
    for i in range(8):
        for j in range(8):
            axe.text(j, i, matrice[i, j], ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(config.RESULTATS / f"confusion_{nom}.png", dpi=150)
    plt.close(figure)
