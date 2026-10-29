"""Lecture des donnees et evaluation partagee par toutes les modalites."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def preparer_dossiers():
    config.DATA.mkdir(parents=True, exist_ok=True)
    config.RESULTATS.mkdir(parents=True, exist_ok=True)


def charger_extraits():
    """Une ligne par extrait, avec sa classe et sa partition par film."""
    partition = {}
    for nom in ("train", "dev", "test"):
        contenu = (config.RACINE / "splits" / f"{nom}.txt").read_text(encoding="utf-8")
        for film_id in contenu.split():
            partition[film_id] = nom

    lignes = []
    for fichier in sorted(config.ANNOTATIONS.glob("*/*.json")):
        if fichier.name.startswith("._"):
            continue
        with fichier.open(encoding="utf-8") as flux:
            a = json.load(flux)
        film_id = fichier.parent.name
        lignes.append({
            "video_id": a["video_id"],
            "film_id": film_id,
            "partition": partition[film_id],
            "emotion": a["emotion"],
            "brightness": a["brightness"],
            "colorfulness": a["colorfulness"],
            "hue": a["hue"],
        })
    return pd.DataFrame(lignes)


def modeles():
    """Deux classifieurs simples, ponderes pour le desequilibre des classes."""
    return {
        "logistique": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000,
                               random_state=config.GRAINE),
        ),
        "foret": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            random_state=config.GRAINE, n_jobs=1,
        ),
    }


def evaluer(X, donnees, nom):
    """Entraine sur train, choisit sur dev, puis sauvegarde les probabilites."""
    preparer_dossiers()
    y = donnees["emotion"].to_numpy()
    ensembles = donnees["partition"].to_numpy()
    train = ensembles == "train"
    dev = ensembles == "dev"

    resultats = []
    entraines = {}
    for etiquette, modele in modeles().items():
        modele.fit(X[train], y[train])
        prediction = modele.predict(X[dev])
        score = f1_score(y[dev], prediction, average="macro")
        exactitude = accuracy_score(y[dev], prediction)
        resultats.append({"configuration": nom, "modele": etiquette,
                          "f1_macro_dev": score, "exactitude_dev": exactitude})
        entraines[etiquette] = modele
        print(f"{nom} | {etiquette:<10} | F1 macro {score:.3f} | exactitude {exactitude:.3f}")

    meilleur = max(resultats, key=lambda r: r["f1_macro_dev"])
    nom_modele = meilleur["modele"]
    modele = entraines[nom_modele]

    # Probabilites dans un ordre de classes fixe, pour la future fusion.
    probabilites = np.zeros((len(donnees), len(config.EMOTIONS)), dtype=np.float32)
    sorties = modele.predict_proba(X)
    for colonne, classe in enumerate(modele.classes_):
        probabilites[:, config.EMOTIONS.index(classe)] = sorties[:, colonne]

    np.savez_compressed(
        config.DATA / f"probas_{nom}.npz",
        video_id=donnees["video_id"].to_numpy(dtype=str),
        partition=ensembles.astype(str),
        y=y.astype(str),
        probabilites=probabilites,
        emotions=np.array(config.EMOTIONS),
        modele=nom_modele,
    )
    joblib.dump(modele, config.DATA / f"modele_{nom}.joblib")

    resume = pd.DataFrame(resultats).sort_values("f1_macro_dev", ascending=False)
    resume.to_csv(config.RESULTATS / f"scores_{nom}.csv", index=False)

    prediction = modele.predict(X[dev])
    detail = pd.DataFrame({
        "emotion": config.EMOTIONS,
        "rappel_dev": recall_score(y[dev], prediction, labels=config.EMOTIONS,
                                   average=None, zero_division=0),
        "effectif_dev": [(y[dev] == e).sum() for e in config.EMOTIONS],
    })
    detail.to_csv(config.RESULTATS / f"rappels_{nom}.csv", index=False)

    matrice = confusion_matrix(y[dev], prediction, labels=config.EMOTIONS)
    figure, axe = plt.subplots(figsize=(8, 7))
    image = axe.imshow(matrice, cmap="Blues")
    axe.set_xticks(range(8), config.EMOTIONS, rotation=45, ha="right")
    axe.set_yticks(range(8), config.EMOTIONS)
    axe.set_xlabel("Prediction")
    axe.set_ylabel("Emotion reelle")
    axe.set_title(f"{nom} - {nom_modele} (dev)")
    for i in range(8):
        for j in range(8):
            axe.text(j, i, matrice[i, j], ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axe)
    figure.tight_layout()
    figure.savefig(config.RESULTATS / f"confusion_{nom}.png", dpi=150)
    plt.close(figure)

    print(f"Meilleur modele : {nom_modele}")
    return resume
