"""N2 : embeddings de modèles pré-entraînés (gelés) + régression logistique."""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import utils

MODELES = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "e5": "intfloat/e5-large-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "goemotions": "SamLowe/roberta-base-go_emotions",
    "ekman": "j-hartmann/emotion-english-roberta-large",
}
ENTREES = ["description", "dialogue", "description+dialogue", "concatenation"]
VALEURS_C = [0.0003, 0.001, 0.003, 0.01, 0.03]

APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"


def encoder(nom, textes):
    # on garde les vecteurs dans data/ pour ne pas réencoder à chaque fois
    cache = config.DATA / f"emb_{nom}_{textes.name}.npy"
    if cache.exists():
        return np.load(cache)
    print(f"Encodage {nom} / {textes.name} sur {APPAREIL}")
    modele = SentenceTransformer(MODELES[nom], device=APPAREIL)
    liste = textes.tolist()
    if nom == "e5":
        liste = ["query: " + t for t in liste]   # préfixe demandé par E5
    vecteurs = modele.encode(liste, batch_size=32, show_progress_bar=True,
                             normalize_embeddings=True)
    np.save(cache, vecteurs)
    return vecteurs


def caracteristiques(corpus, nom, entree):
    # "concatenation" : description + dialogue dans un seul texte (pour comparer)
    if entree == "concatenation":
        texte = (corpus["caption"] + " " + corpus["dialogue"]).str.strip()
        return encoder(nom, texte.rename("concatenation"))
    description = encoder(nom, corpus["caption"])
    dialogue = encoder(nom, corpus["dialogue"])
    a_dialogue = corpus[["a_dialogue"]].to_numpy(dtype=float)
    dialogue = dialogue * a_dialogue   # vecteur nul si pas de dialogue
    if entree == "description":
        return description
    if entree == "dialogue":
        return np.hstack([dialogue, a_dialogue])
    return np.hstack([description, dialogue, a_dialogue])


GRILLE = {"logisticregression__C": VALEURS_C,
          "logisticregression__class_weight": ["balanced", None]}


def regression():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def main(modeles=MODELES):
    corpus = utils.charger_corpus()
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    films = corpus["film_id"].to_numpy()

    lignes = []
    for nom in modeles:
        print(f"\n=== {nom} ===")
        meilleur = None
        for entree in ENTREES:
            X = caracteristiques(corpus, nom, entree)
            resultat = utils.recherche(regression(), GRILLE, X[train], y[train], films[train])
            modele = resultat.best_estimator_
            ligne = {"modele": nom, "entree": entree,
                     "C": resultat.best_params_["logisticregression__C"],
                     "class_weight": resultat.best_params_["logisticregression__class_weight"],
                     "f1_cv": resultat.best_score_,
                     **utils.scores(y[dev], modele.predict(X[dev]))}
            lignes.append(ligne)
            print(f"{entree:22s} C={ligne['C']:<7} {str(ligne['class_weight']):9s} "
                  f"F1 cv {ligne['f1_cv']:.3f}  F1 dev {ligne['f1_macro']:.3f}")
            if meilleur is None or ligne["f1_cv"] > meilleur[0]:
                meilleur = (ligne["f1_cv"], modele, X)

        # on garde l'entrée qui a le meilleur score en CV
        _, modele, X = meilleur
        utils.sauvegarder(f"emb_{nom}", corpus, utils.probabilites_ordonnees(modele, X),
                          utils.probabilites_hors_pli(modele, X, y, films, train),
                          y[dev], modele.predict(X[dev]))

    tableau = pd.DataFrame(lignes)
    tableau.to_csv(config.RESULTATS / "scores_embeddings.csv", index=False)
    for colonne, titre in [("f1_cv", "validation croisée sur train"), ("f1_macro", "dev")]:
        print(f"\nF1 macro ({titre}) :")
        print(tableau.pivot(index="modele", columns="entree", values=colonne).round(3))


if __name__ == "__main__":
    main()
