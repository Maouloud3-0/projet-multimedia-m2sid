"""Compare plusieurs tetes de classification sur les vecteurs deja calcules."""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import utils


def candidats():
    modeles = {}
    for C in (0.1, 1.0, 10.0):
        modeles[f"logistique_C{C}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                               random_state=config.GRAINE),
        )
    for C in (1.0, 10.0):
        modeles[f"svm_C{C}"] = make_pipeline(
            StandardScaler(),
            SVC(C=C, class_weight="balanced", probability=True,
                random_state=config.GRAINE),
        )
    return modeles


def main():
    analyseur = argparse.ArgumentParser()
    analyseur.add_argument("--seulement", default="",
                           help="filtrer les configurations par prefixe")
    options = analyseur.parse_args()
    utils.preparer_dossiers()
    donnees = utils.charger_extraits()
    y = donnees["emotion"].to_numpy()
    ensembles = donnees["partition"].to_numpy()
    train = ensembles == "train"
    dev = ensembles == "dev"
    resultats = []

    vecteurs = {}
    for cache in sorted(config.DATA.glob("*.npz")):
        if cache.name.startswith("probas_"):
            continue
        with np.load(cache, allow_pickle=True) as ancien:
            if "X" in ancien:
                vecteurs[cache.stem] = ancien["X"]

    # Les descripteurs classiques et les embeddings peuvent se completer.
    for nom in ("audio_ast_brut", "audio_wavlm_brut"):
        if "audio_n1" in vecteurs and nom in vecteurs:
            vecteurs[f"{nom}_plus_n1"] = np.concatenate(
                [vecteurs["audio_n1"], vecteurs[nom]], axis=1)

    if "video_n1" in vecteurs and "video_clip" in vecteurs:
        vecteurs["video_clip_plus_n1"] = np.concatenate(
            [vecteurs["video_n1"], vecteurs["video_clip"]], axis=1)

    if options.seulement:
        vecteurs = {nom: X for nom, X in vecteurs.items()
                    if nom.startswith(options.seulement)}

    for nom, X in vecteurs.items():
        print()
        print(f"{nom} ({X.shape[1]} dimensions)")

        meilleur = None
        for etiquette, modele in candidats().items():
            depart = time.time()
            modele.fit(X[train], y[train])
            prediction = modele.predict(X[dev])
            f1 = f1_score(y[dev], prediction, average="macro")
            exactitude = accuracy_score(y[dev], prediction)
            resultats.append({"configuration": nom, "modele": etiquette,
                              "f1_macro_dev": f1, "exactitude_dev": exactitude})
            print(f"  {etiquette:<18} F1 {f1:.3f} | acc {exactitude:.3f} | "
                  f"{time.time() - depart:.0f} s")
            if meilleur is None or f1 > meilleur[0]:
                meilleur = (f1, etiquette, modele)

        _, etiquette, modele = meilleur
        sorties = modele.predict_proba(X)
        probabilites = np.zeros((len(donnees), len(config.EMOTIONS)), dtype=np.float32)
        for colonne, classe in enumerate(modele.classes_):
            probabilites[:, config.EMOTIONS.index(classe)] = sorties[:, colonne]
        np.savez_compressed(config.DATA / f"probas_{nom}_optimise.npz",
                            video_id=donnees["video_id"].to_numpy(dtype=str),
                            partition=ensembles.astype(str), y=y.astype(str),
                            probabilites=probabilites,
                            emotions=np.array(config.EMOTIONS), modele=etiquette)
        joblib.dump(modele, config.DATA / f"modele_{nom}_optimise.joblib")

    tableau = pd.DataFrame(resultats)
    sortie = config.RESULTATS / "comparaison_classifieurs.csv"
    if options.seulement and sortie.exists():
        ancien = pd.read_csv(sortie)
        ancien = ancien[~ancien["configuration"].str.startswith(options.seulement)]
        tableau = pd.concat([ancien, tableau], ignore_index=True)
    tableau.to_csv(sortie, index=False)
    print()
    print("Meilleurs resultats :")
    print(tableau.loc[tableau.groupby("configuration")["f1_macro_dev"].idxmax()]
          .to_string(index=False))


if __name__ == "__main__":
    main()
