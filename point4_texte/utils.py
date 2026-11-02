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


# ---------- fusion (04 et 06) ----------

def predire(probas):
    return np.array(config.EMOTIONS)[probas.argmax(axis=1)]


def charger_probas(nom, corpus):
    fichier = np.load(config.RESULTATS / f"probas_{nom}.npz")
    if not (fichier["video_id"] == corpus["video_id"].to_numpy(dtype=str)).all():
        raise ValueError(f"probas_{nom}.npz n'est pas dans l'ordre du corpus, relancer le script du modèle")
    return fichier["probabilites"], fichier["probabilites_hors_pli"]


def bootstrap_films(y, predit_a, predit_b, films, tirages=2000):
    # IC 95 % de F1(a) - F1(b), en tirant les films avec remise
    rng = np.random.default_rng(config.GRAINE)
    liste = np.unique(films)
    indices = {f: np.where(films == f)[0] for f in liste}
    ecarts = []
    for _ in range(tirages):
        tirage = np.concatenate([indices[f] for f in rng.choice(liste, len(liste))])
        ecarts.append(scores(y[tirage], predit_a[tirage])["f1_macro"]
                      - scores(y[tirage], predit_b[tirage])["f1_macro"])
    return np.percentile(ecarts, [2.5, 97.5])


def fusionner(etape, noms, sortie):
    """Moyenne des probabilités des k meilleurs modèles en CV ; k est lui aussi choisi en CV."""
    corpus = charger_corpus()
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    probas, hors_pli = {}, {}
    for nom in noms:
        probas[nom], hors_pli[nom] = charger_probas(nom, corpus)

    def f1(p, partie):
        return scores(y[partie], predire(p[partie]))["f1_macro"]

    seuls = {nom: f1(hors_pli[nom], train) for nom in noms}
    classement = sorted(noms, key=seuls.get, reverse=True)
    print("Modèles seuls (F1 macro) :")
    for nom in classement:
        print(f"  {nom:16s} cv {seuls[nom]:.3f}  dev {f1(probas[nom], dev):.3f}")

    lignes = []
    for k in range(1, len(noms) + 1):
        membres = classement[:k]
        moyenne_cv = np.mean([hors_pli[nom] for nom in membres], axis=0)
        moyenne = np.mean([probas[nom] for nom in membres], axis=0)
        lignes.append({"k": k, "f1_cv": f1(moyenne_cv, train), "f1_dev": f1(moyenne, dev),
                       "modeles": " + ".join(membres)})
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(config.RESULTATS / f"scores_fusion_{etape}.csv", index=False)
    print("\nMoyenne des k meilleurs :")
    print(tableau.round(3).to_string(index=False))

    k = int(tableau.loc[tableau["f1_cv"].idxmax(), "k"])
    membres = classement[:k]
    fusion = np.mean([probas[nom] for nom in membres], axis=0)
    fusion_cv = np.mean([hors_pli[nom] for nom in membres], axis=0)
    sauvegarder(sortie, corpus, fusion, fusion_cv, y[dev], predire(fusion[dev]))

    bas, haut = bootstrap_films(y[dev], predire(fusion[dev]), predire(probas[classement[0]][dev]),
                                corpus["film_id"].to_numpy()[dev])
    s = scores(y[dev], predire(fusion[dev]))
    print(f"\nRetenu (sur train) : k = {k} ({' + '.join(membres)})")
    print(f"F1 macro cv {f1(fusion_cv, train):.3f}, dev {s['f1_macro']:.3f}, exactitude dev {s['exactitude']:.3f}")
    print(f"Gain sur le meilleur modèle seul : IC 95 % par films [{bas:+.3f}, {haut:+.3f}]")
