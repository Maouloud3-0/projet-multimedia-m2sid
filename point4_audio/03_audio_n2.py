"""N2 : vecteurs des modèles gelés (02) + un classifieur appris sur train.

Tous les choix sont faits par validation croisée sur train (5 plis groupés par film) ;
dev ne sert qu'à mesurer. Même démarche que point4_video/05_video_n2.py.
Étape A, pour chaque encodeur, régression logistique :
    A1  couche : 7 couches réparties de la première à la dernière, ou moyenne de toutes les couches
        (résumé dans le temps : moyenne)
    A2  résumé dans le temps de la couche retenue : moyenne ou moyenne + écart-type
Étape B, avec le meilleur réglage de A, régression logistique, descripteurs ajoutés :
    aucun / n1 (descripteurs de 01) / audioset (probabilités AudioSet d'AST) / n1 + audioset
Étape D, pour « aucun » et le meilleur complément de B, deux autres classifieurs :
    lda  analyse discriminante linéaire, rétrécissement calculé automatiquement (Ledoit-Wolf)
    svm  SVM à noyau gaussien, C et gamma cherchés par Optuna (20 essais, chacun évalué en CV),
         puis calibration sigmoïde pour obtenir des probabilités
Étape C : les probabilités AudioSet seules, avec les trois classifieurs.
Pour chaque encodeur, la combinaison retenue est celle qui a le meilleur F1 en CV calculé sur
les probabilités (f1_cv_probas), c'est-à-dire sur ce que la fusion utilise.

Sorties : resultats/scores_n2.csv (une ligne par configuration, reprise possible après une coupure),
          resultats/probas_emb_<encodeur>.npz (configuration retenue), resultats/probas_audioset.npz.
"""

import itertools
import json

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

import config
import utils

ENCODEURS = ["whisper", "wavlm", "w2v2_emo", "ast", "clap"]
NB_COUCHES = 7
COMPLEMENTS = ["aucun", "n1", "audioset", "n1+audioset"]
AUTRES_CLASSIFIEURS = ["lda", "svm"]
VALEURS_C = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
ESSAIS_SVM = 20
FICHIER = config.RESULTATS / "scores_n2.csv"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def representation(couches, couche, regroupement):
    """couches : (extraits, couches, 2 x dimension), moyenne puis écart-type dans le temps."""
    dimension = couches.shape[2] // 2
    choisie = couches.astype(np.float32).mean(1) if couche == "moyenne" else couches[:, int(couche)].astype(np.float32)
    return choisie[:, :dimension] if regroupement == "moyenne" else choisie


def complements():
    n1 = np.load(config.DATA / "n1.npy").astype(np.float32)
    audioset = np.load(config.DATA / "audioset.npy").astype(np.float32)
    return {"n1": n1, "audioset": audioset, "n1+audioset": np.hstack([n1, audioset])}


def classifieur(nom, p, dimension=None, probabilites=True):
    """Le modèle (standardisation + classifieur) pour des réglages donnés."""
    if nom == "lr":
        fin = LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=3000)
    elif nom == "lda":
        priors = None if p["priors"] == "frequences" else np.full(len(config.EMOTIONS), 1 / len(config.EMOTIONS))
        fin = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
    else:
        fin = SVC(C=p["C"], gamma=p["gamma_relatif"] / dimension, class_weight="balanced")
        if probabilites:      # le SVM ne donne pas de probabilités : calibration sigmoïde (Platt)
            fin = CalibratedClassifierCV(fin, ensemble=False)
    return make_pipeline(StandardScaler(), fin)


def score_pli(modele, X, y, apprentissage, validation):
    modele.fit(X[apprentissage], y[apprentissage])
    return f1_score(y[validation], modele.predict(X[validation]), average="macro")


class Experience:
    def __init__(self):
        self.corpus = utils.charger_corpus()
        self.train = (self.corpus["partition"] == "train").to_numpy()
        self.dev = (self.corpus["partition"] == "dev").to_numpy()
        self.y = self.corpus["emotion"].to_numpy()
        self.films = self.corpus["film_id"].to_numpy()
        self.faits = pd.read_csv(FICHIER, dtype={"couche": str}) if FICHIER.exists() else pd.DataFrame()
        self.barre = tqdm(desc="configurations")

    def deja(self, **cle):
        """Indice de la ligne déjà calculée pour cette configuration, ou None."""
        if self.faits.empty:
            return None
        lignes = self.faits
        for nom, valeur in cle.items():
            lignes = lignes[lignes[nom].astype(str) == str(valeur)]
        return None if lignes.empty else lignes.index[0]

    def svm_calibre(self, X, reglages):
        """F1 cv et scores dev du SVM calibré (la classe prédite est la plus probable après calibration)."""
        Xt, yt, ft = X[self.train], self.y[self.train], self.films[self.train]
        plis = list(GroupKFold(5).split(Xt, yt, ft))
        f1_cv = np.mean(Parallel(n_jobs=-1)(delayed(score_pli)(classifieur("svm", reglages, X.shape[1]), Xt, yt, a, t)
                                            for a, t in plis))
        modele = classifieur("svm", reglages, X.shape[1]).fit(Xt, yt)
        return {"f1_cv_probas": f1_cv, **dict(zip(["f1_dev", "exactitude_dev"],
                                                  utils.scores(self.y[self.dev], modele.predict(X[self.dev])).values()))}

    def chercher(self, nom, X):
        """Réglages du classifieur choisis en CV sur train ; renvoie (réglages, F1 cv)."""
        Xt, yt, ft = X[self.train], self.y[self.train], self.films[self.train]
        if nom in ("lr", "lda"):
            grille = ({"C": VALEURS_C, "class_weight": ["balanced", None]} if nom == "lr"
                      else {"priors": ["frequences", "egales"]})
            essais = [dict(zip(grille, v)) for v in itertools.product(*grille.values())]
            plis = list(GroupKFold(5).split(Xt, yt, ft))
            # tous les couples (réglage, pli) en parallèle, pour occuper tous les cœurs
            scores = Parallel(n_jobs=-1)(delayed(score_pli)(classifieur(nom, p, X.shape[1], False), Xt, yt, a, t)
                                         for p in essais for a, t in plis)
            scores = np.array(scores).reshape(len(essais), len(plis)).mean(1)
            k = int(np.argmax(scores))
            return essais[k], scores[k]

        def objectif(essai):
            p = {"C": essai.suggest_float("C", 0.1, 100, log=True),
                 "gamma_relatif": essai.suggest_float("gamma_relatif", 0.03, 3, log=True)}
            return cross_val_score(classifieur("svm", p, X.shape[1], False), Xt, yt, groups=ft,
                                   cv=GroupKFold(5), scoring="f1_macro", n_jobs=-1).mean()

        etude = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.GRAINE))
        etude.optimize(objectif, n_trials=ESSAIS_SVM)
        return etude.best_params, etude.best_value

    def evaluer(self, X, classifieur_nom="lr", **cle):
        cle = {**cle, "classifieur": classifieur_nom}
        indice = self.deja(**cle)
        self.barre.update()
        if indice is not None:
            return self.faits.loc[indice]
        reglages, f1_cv = self.chercher(classifieur_nom, X)
        if classifieur_nom == "svm":
            calibre = self.svm_calibre(X, reglages)
        else:
            modele = classifieur(classifieur_nom, reglages, X.shape[1], False).fit(X[self.train], self.y[self.train])
            s = utils.scores(self.y[self.dev], modele.predict(X[self.dev]))
            calibre = {"f1_cv_probas": f1_cv, "f1_dev": s["f1_macro"], "exactitude_dev": s["exactitude"]}
        ligne = {**cle, "dimension": X.shape[1], "parametres": json.dumps(reglages), "f1_cv": f1_cv, **calibre}
        self.faits = pd.concat([self.faits, pd.DataFrame([ligne])], ignore_index=True)
        self.faits.to_csv(FICHIER, index=False)          # enregistré après chaque configuration
        self.barre.write(f"  {str(cle):100s} F1 cv {calibre['f1_cv_probas']:.3f}  F1 dev {calibre['f1_dev']:.3f}")
        return pd.Series(ligne)

    def enregistrer(self, nom, X, ligne):
        # modèle final avec les réglages retenus en CV, appris sur train ; probabilités hors pli sur train
        modele = classifieur(ligne["classifieur"], json.loads(ligne["parametres"]), X.shape[1])
        modele.fit(X[self.train], self.y[self.train])
        hors_pli = np.full((len(self.y), len(config.EMOTIONS)), np.nan)
        hors_pli[self.train] = cross_val_predict(modele, X[self.train], self.y[self.train],
                                                 groups=self.films[self.train], cv=GroupKFold(5),
                                                 method="predict_proba", n_jobs=-1)
        utils.sauvegarder(nom, self.corpus, utils.probabilites_ordonnees(modele, X), hors_pli,
                          self.y[self.dev], modele.predict(X[self.dev]))


def main():
    exp = Experience()
    ajouts = complements()

    for encodeur in ENCODEURS:
        exp.barre.write(f"\n=== {encodeur} ===")
        couches = np.load(config.DATA / f"couches_{encodeur}.npy")
        nb = couches.shape[1]
        candidates = ["moyenne"] if nb == 1 else \
            [str(c) for c in np.unique(np.linspace(0, nb - 1, NB_COUCHES).round().astype(int))] + ["moyenne"]
        commun = dict(encodeur=encodeur)

        # étape A1 : couche ; A2 : résumé dans le temps
        lignes_a = {(c, "moyenne"): exp.evaluer(representation(couches, c, "moyenne"), etape="A", couche=c,
                                                regroupement="moyenne", complement="aucun", **commun)
                    for c in candidates}
        c = max(lignes_a, key=lambda k: lignes_a[k]["f1_cv"])[0]
        lignes_a[(c, "moyenne+ecart")] = exp.evaluer(representation(couches, c, "moyenne+ecart"), etape="A",
                                                     couche=c, regroupement="moyenne+ecart", complement="aucun",
                                                     **commun)
        c, r = max(lignes_a, key=lambda k: lignes_a[k]["f1_cv"])
        base = representation(couches, c, r)
        entrees = {"aucun": base, **{nom: np.hstack([base, X]) for nom, X in ajouts.items()}}
        commun.update(couche=c, regroupement=r)

        # étape B : descripteurs ajoutés, régression logistique
        candidats = {("aucun", "lr"): lignes_a[(c, r)]}
        candidats.update({(k, "lr"): exp.evaluer(entrees[k], etape="B", complement=k, **commun)
                          for k in COMPLEMENTS[1:]})
        meilleur_b = max(COMPLEMENTS[1:], key=lambda k: candidats[(k, "lr")]["f1_cv"])
        # étape D : autres classifieurs, sans complément et avec le meilleur complément
        candidats.update({(k, nom): exp.evaluer(entrees[k], nom, etape="D", complement=k, **commun)
                          for k in ["aucun", meilleur_b] for nom in AUTRES_CLASSIFIEURS})
        k, nom = max(candidats, key=lambda x: candidats[x]["f1_cv_probas"])
        exp.barre.write(f"  retenu : couche {c}, {r}, complément {k}, {nom}  "
                        f"(F1 cv {candidats[(k, nom)]['f1_cv_probas']:.3f})")
        exp.enregistrer(f"emb_{encodeur}", entrees[k], candidats[(k, nom)])

    # étape C : probabilités AudioSet seules
    exp.barre.write("\n=== audioset seul ===")
    commun = dict(etape="C", encodeur="audioset", couche="moyenne", regroupement="moyenne", complement="audioset")
    candidats = {nom: exp.evaluer(ajouts["audioset"], nom, **commun) for nom in ["lr"] + AUTRES_CLASSIFIEURS}
    nom = max(candidats, key=lambda x: candidats[x]["f1_cv_probas"])
    exp.enregistrer("audioset", ajouts["audioset"], candidats[nom])

    exp.barre.close()
    tableau = exp.faits
    print("\nMeilleure configuration par encodeur (F1 cv) :")
    print(tableau.loc[tableau.groupby("encodeur")["f1_cv_probas"].idxmax()]
          [["encodeur", "couche", "regroupement", "complement", "classifieur", "f1_cv", "f1_cv_probas",
            "f1_dev", "exactitude_dev"]].sort_values("f1_cv_probas", ascending=False).round(3).to_string(index=False))
    print("\nMeilleur F1 cv (sur les probabilités) par encodeur et classifieur :")
    print(tableau.pivot_table(index="encodeur", columns="classifieur", values="f1_cv_probas", aggfunc="max").round(3))


if __name__ == "__main__":
    main()
