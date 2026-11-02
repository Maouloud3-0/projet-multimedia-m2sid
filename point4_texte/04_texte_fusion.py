"""Fusion texte : moyenne des probabilités des k meilleurs modèles (k choisi en CV sur train)."""

import numpy as np
import pandas as pd

import config
import utils

NOMS = ["tfidf", "emb_mpnet", "emb_e5", "emb_bge", "emb_goemotions", "emb_ekman", "ft_ekman"]
TIRAGES = 2000


def charger(nom, corpus):
    fichier = np.load(config.RESULTATS / f"probas_{nom}.npz")
    if not (fichier["video_id"] == corpus["video_id"].to_numpy(dtype=str)).all():
        raise ValueError(f"probas_{nom}.npz n'est pas dans l'ordre du corpus, relancer le script du modèle")
    return fichier["probabilites"], fichier["probabilites_hors_pli"]


def predire(probas):
    return np.array(config.EMOTIONS)[probas.argmax(axis=1)]


def bootstrap_films(y, predit_a, predit_b, films):
    # IC 95 % de F1(a) - F1(b), en tirant les films de dev avec remise
    rng = np.random.default_rng(config.GRAINE)
    liste = np.unique(films)
    indices = {f: np.where(films == f)[0] for f in liste}
    ecarts = []
    for _ in range(TIRAGES):
        tirage = np.concatenate([indices[f] for f in rng.choice(liste, len(liste))])
        ecarts.append(utils.scores(y[tirage], predit_a[tirage])["f1_macro"]
                      - utils.scores(y[tirage], predit_b[tirage])["f1_macro"])
    return np.percentile(ecarts, [2.5, 97.5])


def main():
    corpus = utils.charger_corpus()
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    probas, hors_pli = {}, {}
    for nom in NOMS:
        probas[nom], hors_pli[nom] = charger(nom, corpus)

    def f1(p, partie):
        return utils.scores(y[partie], predire(p[partie]))["f1_macro"]

    seuls = {nom: f1(hors_pli[nom], train) for nom in NOMS}
    classement = sorted(NOMS, key=seuls.get, reverse=True)
    print("Modèles seuls (F1 macro) :")
    for nom in classement:
        print(f"  {nom:16s} cv {seuls[nom]:.3f}  dev {f1(probas[nom], dev):.3f}")

    lignes = []
    for k in range(1, len(NOMS) + 1):
        membres = classement[:k]
        moyenne_cv = np.mean([hors_pli[nom] for nom in membres], axis=0)
        moyenne = np.mean([probas[nom] for nom in membres], axis=0)
        lignes.append({"k": k, "f1_cv": f1(moyenne_cv, train), "f1_dev": f1(moyenne, dev),
                       "modeles": " + ".join(membres)})
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(config.RESULTATS / "scores_fusion.csv", index=False)
    print("\nMoyenne des k meilleurs :")
    print(tableau.round(3).to_string(index=False))

    k = int(tableau.loc[tableau["f1_cv"].idxmax(), "k"])
    membres = classement[:k]
    fusion = np.mean([probas[nom] for nom in membres], axis=0)
    fusion_cv = np.mean([hors_pli[nom] for nom in membres], axis=0)
    utils.sauvegarder("texte", corpus, fusion, fusion_cv, y[dev], predire(fusion[dev]))

    bas, haut = bootstrap_films(y[dev], predire(fusion[dev]), predire(probas[classement[0]][dev]),
                                corpus["film_id"].to_numpy()[dev])
    print(f"\nRetenu (sur train) : k = {k} ({' + '.join(membres)})")
    print(f"F1 macro dev {f1(fusion, dev):.3f}, exactitude dev "
          f"{utils.scores(y[dev], predire(fusion[dev]))['exactitude']:.3f}")
    print(f"Gain sur le meilleur modèle seul : IC 95 % par films [{bas:+.3f}, {haut:+.3f}]")


if __name__ == "__main__":
    main()
