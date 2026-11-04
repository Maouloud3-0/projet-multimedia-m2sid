"""Option B — fusion précoce : on met bout à bout les vecteurs gelés des trois modalités, puis un seul classifieur.

Vecteurs (un encodeur par modalité, le meilleur seul au point 4, voir utils.py) :
  vidéo  PE moyenné sur 16 images (1 024) + visages ViT et HSEmotion résumés (2 084)
  texte  e5-large, description + dialogue (1 024) + indicateur de dialogue
  audio  Whisper large-v3, couche 16, moyenne dans le temps (1 280)
Classifieur : standardisation puis régression logistique ; C et poids des classes choisis en CV
(5 plis groupés par film, comme au point 4).

Deux variantes :
  brut        chaque colonne standardisée compte autant : la modalité qui a le plus de colonnes pèse le plus
  equilibre   chaque modalité est multipliée par racine(dimension moyenne / sa dimension) : même poids total
              pour les trois, quelle que soit leur taille
Pour la variante retenue en CV, on mesure aussi l'apport de chaque modalité.

Sorties : resultats/scores_precoce.csv, resultats/probas_precoce_<variante>.npz
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config
import utils

GRILLE = {"logisticregression__C": [0.0001, 0.0003, 0.001, 0.003, 0.01],
          "logisticregression__class_weight": ["balanced", None]}


def modele(blocs, variante):
    # facteur par colonne : 1 (brut) ou racine(dimension moyenne / dimension de la modalité) (equilibre) ;
    # l'échelle moyenne reste 1, pour que la grille de C garde le même sens dans les deux variantes
    moyenne = np.mean(list(blocs.values()))
    facteurs = np.concatenate([np.full(d, 1.0 if variante == "brut" else np.sqrt(moyenne / d)) for d in blocs.values()])
    return make_pipeline(StandardScaler(), FunctionTransformer(lambda X: X * facteurs),
                         LogisticRegression(max_iter=3000))


def main():
    corpus = utils.charger_corpus()
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    films = corpus["film_id"].to_numpy()
    vecteurs = {"video": np.hstack([utils.video_images().mean(1), utils.visages_resume()]),
                "texte": utils.texte_e5(corpus),
                "audio": utils.audio_resume()}
    print({m: v.shape[1] for m, v in vecteurs.items()})

    def evaluer(modalites, variante, enregistrer=False):
        blocs = {m: vecteurs[m].shape[1] for m in modalites}
        X = np.hstack([vecteurs[m] for m in modalites])
        recherche = GridSearchCV(modele(blocs, variante), GRILLE, scoring="f1_macro", cv=GroupKFold(5), n_jobs=-1)
        recherche.fit(X[train], y[train], groups=films[train])
        retenu = recherche.best_estimator_
        hors_pli = np.full((len(y), len(config.EMOTIONS)), np.nan)
        hors_pli[train] = cross_val_predict(retenu, X[train], y[train], groups=films[train], cv=GroupKFold(5),
                                            method="predict_proba", n_jobs=-1)
        probas = utils.probabilites_ordonnees(retenu, X)
        s = utils.scores(y[dev], utils.predire(probas[dev]))
        ligne = {"variante": variante, "modalites": "+".join(modalites), "dimension": X.shape[1],
                 "parametres": json.dumps({k.split("__")[1]: v for k, v in recherche.best_params_.items()}),
                 "f1_cv": utils.scores(y[train], utils.predire(hors_pli[train]))["f1_macro"],
                 "f1_dev": s["f1_macro"], "exactitude_dev": s["exactitude"]}
        print(f"{variante:10s} {ligne['modalites']:18s} F1 cv {ligne['f1_cv']:.3f}  dev {ligne['f1_dev']:.3f}", flush=True)
        if enregistrer:
            utils.sauvegarder(f"precoce_{variante}", corpus, probas, hors_pli, y[dev], utils.predire(probas[dev]))
        return ligne

    tous = utils.MODALITES
    lignes = [evaluer(["video"], "brut")]                       # référence : la vidéo seule
    lignes += [evaluer(tous, v, enregistrer=True) for v in ["brut", "equilibre"]]
    retenue = max(lignes[1:], key=lambda l: l["f1_cv"])["variante"]
    lignes += [evaluer(m, retenue) for m in [["video", "texte"], ["video", "audio"]]]

    tableau = pd.DataFrame(lignes)
    tableau.to_csv(config.RESULTATS / "scores_precoce.csv", index=False)
    print(tableau.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
