"""N1 : TF-IDF + régression logistique / SVM linéaire (réglages choisis en CV sur train)."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import config
import utils

ENTREES = ["description", "dialogue", "description+dialogue"]
VALEURS_C = [0.1, 0.3, 1, 3, 10]
ETAPES = {"logistique": "logisticregression", "svm": "linearsvc"}


def chaine(nom):
    # TF-IDF dans le pipeline pour qu'il soit réappris à chaque pli
    vectoriseur = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  stop_words="english")
    if nom == "logistique":
        return make_pipeline(vectoriseur, LogisticRegression(max_iter=3000))
    return make_pipeline(vectoriseur, LinearSVC())


def main():
    corpus = utils.charger_corpus()
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    films = corpus["film_id"].to_numpy()

    lignes = []
    modeles = {}
    for entree in ENTREES:
        textes = utils.textes(corpus, entree).to_numpy()
        for nom, etape in ETAPES.items():
            grille = {f"{etape}__C": VALEURS_C, f"{etape}__class_weight": ["balanced", None]}
            resultat = utils.recherche(chaine(nom), grille, textes[train], y[train], films[train])
            modele = resultat.best_estimator_
            modeles[(entree, nom)] = modele
            lignes.append({"entree": entree, "classifieur": nom,
                           "C": resultat.best_params_[f"{etape}__C"],
                           "class_weight": resultat.best_params_[f"{etape}__class_weight"],
                           "f1_cv": resultat.best_score_,
                           **utils.scores(y[dev], modele.predict(textes[dev]))})

    tableau = pd.DataFrame(lignes)
    tableau.to_csv(config.RESULTATS / "scores_tfidf.csv", index=False)
    print("Meilleur réglage par entrée et par classifieur (f1_cv : train, f1_macro : dev) :")
    print(tableau.round(3).to_string(index=False))

    # la fusion a besoin de probabilités -> on garde la logistique (pas le SVM)
    choix = tableau[tableau["classifieur"] == "logistique"].sort_values("f1_cv").iloc[-1]
    print(f"\nRetenu pour la fusion : {choix['entree']}, logistique, C={choix['C']}, "
          f"class_weight={choix['class_weight']} (F1 cv {choix['f1_cv']:.3f}, dev {choix['f1_macro']:.3f})")
    modele = modeles[(choix["entree"], "logistique")]
    textes = utils.textes(corpus, choix["entree"]).to_numpy()
    utils.sauvegarder("tfidf", corpus, utils.probabilites_ordonnees(modele, textes),
                      utils.probabilites_hors_pli(modele, textes, y, films, train),
                      y[dev], modele.predict(textes[dev]))


if __name__ == "__main__":
    main()
