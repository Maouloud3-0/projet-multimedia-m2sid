"""Option A — fusion tardive : chaque modalité a déjà donné ses 8 probabilités (point 4), on les combine.

  moyenne     moyenne simple des probabilités des modalités
  ponderee    moyenne pondérée ; poids de 0 à 1 par pas de 0,1 pour chaque modalité, choisis en CV
  stacking    régression logistique sur les log-probabilités mises bout à bout (8 par modalité) ;
              C et poids des classes choisis en CV (5 plis groupés par film)

Tout est appris sur les probabilités hors pli de train : ce sont des prédictions de modèles qui n'ont pas
vu l'extrait, comme le seront celles de dev et de test. Dev ne sert qu'à mesurer.

Deux jeux d'entrées :
  final   les systèmes finaux du point 4 (texte avec fine-tuning, combinaisons de plusieurs modèles par modalité)
  gele    un seul encodeur gelé par modalité (PE + visages, e5, Whisper) : les mêmes entrées que B et C,
          pour les comparer à armes égales
Pour le jeu final, on mesure aussi l'apport de chaque modalité (moyenne pondérée sur chaque sous-ensemble).

Sorties : resultats/scores_tardive.csv, resultats/probas_tardive_<méthode>[_gele].npz
"""

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import utils

JEUX = {
    "final": {"video": "video", "texte": "texte", "audio": "audio"},
    "gele": {"video": "emb_pe", "texte": "emb_e5", "audio": "emb_whisper"},
}
POIDS = np.round(np.arange(0, 1.01, 0.1), 1)
GRILLE = {"logisticregression__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
          "logisticregression__class_weight": ["balanced", None]}


class Fusion:
    def __init__(self, jeu):
        self.corpus = utils.charger_corpus()
        self.train = (self.corpus["partition"] == "train").to_numpy()
        self.dev = (self.corpus["partition"] == "dev").to_numpy()
        self.y = self.corpus["emotion"].to_numpy()
        self.films = self.corpus["film_id"].to_numpy()
        self.probas, self.hors_pli = {}, {}
        for modalite, nom in JEUX[jeu].items():
            self.probas[modalite], self.hors_pli[modalite] = utils.charger_probas(config.PROBAS[modalite], nom,
                                                                                  self.corpus)

    def f1(self, p, partie):
        return utils.scores(self.y[partie], utils.predire(p[partie]))["f1_macro"]

    def ligne(self, methode, modalites, probas, hors_pli, parametres):
        s = utils.scores(self.y[self.dev], utils.predire(probas[self.dev]))
        return {"methode": methode, "modalites": "+".join(modalites), "parametres": json.dumps(parametres),
                "f1_cv": self.f1(hors_pli, self.train), "f1_dev": s["f1_macro"], "exactitude_dev": s["exactitude"]}

    def moyenne(self, modalites, poids=None):
        poids = poids or {m: 1.0 for m in modalites}
        total = sum(poids.values())
        combiner = lambda d: sum(poids[m] * d[m] for m in modalites) / total
        return combiner(self.probas), combiner(self.hors_pli)

    def ponderee(self, modalites):
        # les poids ne comptent qu'à une constante près : on garde le premier meilleur de la grille
        grille = [dict(zip(modalites, w)) for w in itertools.product(POIDS, repeat=len(modalites)) if sum(w) > 0]
        meilleur = max(grille, key=lambda w: self.f1(self.moyenne(modalites, w)[1], self.train))
        return (*self.moyenne(modalites, meilleur), meilleur)

    def stacking(self, modalites):
        entree = lambda d: np.log(np.hstack([d[m] for m in modalites]) + 1e-6)
        X, X_cv = entree(self.probas), entree(self.hors_pli)
        modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        recherche = GridSearchCV(modele, GRILLE, scoring="f1_macro", cv=GroupKFold(5), n_jobs=-1)
        recherche.fit(X_cv[self.train], self.y[self.train], groups=self.films[self.train])
        hors_pli = np.full_like(X_cv[:, :8], np.nan)
        hors_pli[self.train] = cross_val_predict(recherche.best_estimator_, X_cv[self.train], self.y[self.train],
                                                 groups=self.films[self.train], cv=GroupKFold(5),
                                                 method="predict_proba", n_jobs=-1)
        parametres = {k.split("__")[1]: v for k, v in recherche.best_params_.items()}
        return utils.probabilites_ordonnees(recherche.best_estimator_, X), hors_pli, parametres


def main():
    lignes = []
    for jeu in JEUX:
        fusion = Fusion(jeu)
        suffixe = "" if jeu == "final" else "_gele"
        print(f"\n=== entrées : {jeu} ({', '.join(JEUX[jeu].values())}) ===")
        for m in utils.MODALITES:
            p, h = fusion.moyenne([m])
            lignes.append({"jeu": jeu, **fusion.ligne("seule", [m], p, h, {})})

        tous = utils.MODALITES
        resultats = {"moyenne": (*fusion.moyenne(tous), {}),
                     "ponderee": fusion.ponderee(tous),
                     "stacking": fusion.stacking(tous)}
        for methode, (p, h, parametres) in resultats.items():
            lignes.append({"jeu": jeu, **fusion.ligne(methode, tous, p, h, parametres)})
            utils.sauvegarder(f"tardive_{methode}{suffixe}", fusion.corpus, p, h,
                              fusion.y[fusion.dev], utils.predire(p[fusion.dev]))

        if jeu == "final":       # apport de chaque modalité
            for modalites in [["video", "texte"], ["video", "audio"], ["texte", "audio"]]:
                p, h, poids = fusion.ponderee(modalites)
                lignes.append({"jeu": jeu, **fusion.ligne("ponderee", modalites, p, h, poids)})

        tableau = pd.DataFrame([l for l in lignes if l["jeu"] == jeu])
        print(tableau.drop(columns="jeu").round(3).to_string(index=False))

    pd.DataFrame(lignes).to_csv(config.RESULTATS / "scores_tardive.csv", index=False)


if __name__ == "__main__":
    main()
