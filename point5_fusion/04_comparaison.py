"""Comparaison des trois options de fusion et système final.

Candidats : les sorties de 01 (A, fusion tardive), 02 (B, fusion précoce) et 03 (C, attention multimodale).
Le système retenu est celui qui a le meilleur F1 hors pli sur train (CV), comme au point 4 ; dev mesure le gain
sur la vidéo seule (IC 95 % par films). Test n'est utilisé qu'ici, une seule fois, pour le système retenu et
pour les trois modalités seules (références) : aucun choix n'est fait sur test.

Sorties : resultats/scores_comparaison.csv, resultats/probas_finale.npz, resultats/confusion_finale(_test).png
"""

import numpy as np
import pandas as pd

import config
import utils

OPTIONS = {
    "A": ["tardive_moyenne", "tardive_ponderee", "tardive_stacking",
          "tardive_moyenne_gele", "tardive_ponderee_gele", "tardive_stacking_gele"],
    "B": ["precoce_brut", "precoce_equilibre"],
    "C": ["attention"],
}


def main():
    corpus = utils.charger_corpus()
    y = corpus["emotion"].to_numpy()
    films = corpus["film_id"].to_numpy()
    parties = {p: (corpus["partition"] == p).to_numpy() for p in ["train", "dev", "test"]}

    systemes = {f"seule_{m}": ("—", *utils.charger_probas(config.PROBAS[m], m, corpus)) for m in utils.MODALITES}
    for option, noms in OPTIONS.items():
        for nom in noms:
            if (config.RESULTATS / f"probas_{nom}.npz").exists():     # une option non lancée est ignorée
                systemes[nom] = (option, *utils.charger_probas(config.RESULTATS, nom, corpus))

    f1 = lambda p, partie: utils.scores(y[parties[partie]], utils.predire(p[parties[partie]]))["f1_macro"]
    lignes = [{"option": option, "systeme": nom, "f1_cv": f1(h, "train"), "f1_dev": f1(p, "dev")}
              for nom, (option, p, h) in systemes.items()]
    tableau = pd.DataFrame(lignes)

    candidats = tableau[tableau["option"] != "—"]
    retenu = candidats.loc[candidats["f1_cv"].idxmax(), "systeme"]
    _, probas, hors_pli = systemes[retenu]
    video = systemes["seule_video"][1]
    dev = parties["dev"]
    bas, haut = utils.bootstrap_films(y[dev], utils.predire(probas[dev]), utils.predire(video[dev]), films[dev])

    # test : une seule fois, pour le système retenu et les références
    tableau["f1_test"] = np.nan
    for nom in [retenu, "seule_video", "seule_texte", "seule_audio"]:
        tableau.loc[tableau["systeme"] == nom, "f1_test"] = f1(systemes[nom][1], "test")
    tableau.to_csv(config.RESULTATS / "scores_comparaison.csv", index=False)
    print(tableau.round(3).to_string(index=False))

    utils.sauvegarder("finale", corpus, probas, hors_pli, y[dev], utils.predire(probas[dev]))
    test = parties["test"]
    utils.matrice("finale", y[test], utils.predire(probas[test]), "test")
    s_dev = utils.scores(y[dev], utils.predire(probas[dev]))
    s_test = utils.scores(y[test], utils.predire(probas[test]))
    bas_t, haut_t = utils.bootstrap_films(y[test], utils.predire(probas[test]), utils.predire(video[test]),
                                          films[test])
    print(f"\nRetenu (meilleur F1 cv) : {retenu}")
    print(f"dev  : F1 {s_dev['f1_macro']:.3f}, exactitude {s_dev['exactitude']:.3f}, "
          f"gain sur la vidéo seule IC 95 % [{bas:+.3f}, {haut:+.3f}]")
    print(f"test : F1 {s_test['f1_macro']:.3f}, exactitude {s_test['exactitude']:.3f}, "
          f"gain sur la vidéo seule IC 95 % [{bas_t:+.3f}, {haut_t:+.3f}]")


if __name__ == "__main__":
    main()
