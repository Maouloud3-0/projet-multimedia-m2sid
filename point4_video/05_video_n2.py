"""N2 : vecteurs des modèles gelés (02) et des visages (03) + un classifieur appris sur train.

Tous les choix sont faits par validation croisée sur train (5 plis groupés par film) ;
dev ne sert qu'à mesurer.
Étape A, pour chaque encodeur, sans visages, régression logistique :
    nombre d'images 4 / 8 / 16 (une image sur 4, sur 2, toutes)  x  regroupement moyenne / moyenne + écart-type
Étape B, avec le meilleur réglage de A, régression logistique : visages aucun / vit / hse / les deux.
Étape D, mêmes entrées que B, deux autres classifieurs :
    lda  analyse discriminante linéaire, rétrécissement calculé automatiquement (Ledoit-Wolf)
    svm  SVM à noyau gaussien, C et gamma cherchés par Optuna (25 essais, chacun évalué en CV)
Étape C : les visages seuls (vit, hse, les deux), avec les trois classifieurs.
Pour chaque encodeur, la combinaison visages x classifieur retenue est celle qui a le meilleur F1 en CV
calculé sur les probabilités (colonne f1_cv_probas), c'est-à-dire sur ce que la fusion utilise :
pour la régression logistique et la LDA, la prédiction est la classe la plus probable (f1_cv_probas = f1_cv) ;
le SVM ne donne pas de probabilités, on les obtient par calibration sigmoïde (Platt), et la classe la plus
probable après calibration n'est pas toujours celle que le SVM prédit : son F1 est donc recalculé
sur le modèle calibré (C et gamma restent ceux choisis par Optuna).

Réglages cherchés dans la même CV :
    lr   C parmi 0,00001 à 0,1, poids des classes équilibrés ou non
    lda  probabilités a priori des classes égales ou selon leur fréquence
    svm  C de 0,1 à 100 et gamma de 0,03/d à 3/d (d = dimension), classes équilibrées

Visages, résumés par extrait sur les 16 images (tous les visages trouvés) :
    moyenne des vecteurs, moyenne et maximum des probabilités d'expression,
    moyenne valence / éveil (hse), part des images avec visage, nombre moyen de visages,
    aire moyenne du plus grand visage, indicateur « au moins un visage ».

Sorties : resultats/scores_n2.csv (une ligne par configuration, reprise possible après une coupure),
          resultats/probas_emb_<encodeur>.npz (configuration retenue), resultats/probas_visages.npz.
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
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

import config
import utils

ENCODEURS = ["clip_b", "clip_l", "clip_h", "siglip2", "pe", "dinov2", "videomae"]
NB_IMAGES = [4, 8, 16]
REGROUPEMENTS = ["moyenne", "moyenne+ecart"]
VISAGES = ["aucun", "vit", "hse", "vit+hse"]
AUTRES_CLASSIFIEURS = ["lda", "svm"]
VALEURS_C = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
ESSAIS_SVM = 25
FICHIER = config.RESULTATS / "scores_n2.csv"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def regrouper(vecteurs, nb_images, regroupement):
    choisis = vecteurs[:, ::config.NB_IMAGES // nb_images].astype(np.float32)
    if regroupement == "moyenne":
        return choisis.mean(1)
    return np.hstack([choisis.mean(1), choisis.std(1)])


def moyenne_masquee(valeurs, masque):
    # moyenne sur les visages présents (16 images x 2 emplacements), 0 si aucun visage
    m = masque[..., None].astype(np.float32)
    return (valeurs.astype(np.float32) * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)


def caracteristiques_visages(choix):
    v = np.load(config.DATA / "visages.npz")
    masque = v["masque"]
    present = masque[..., 0]
    grand = np.where(present, v["boites"][:, :, 0, 4], 0)
    blocs = [present.mean(1, keepdims=True),
             v["nb_visages"].mean(1, keepdims=True),
             grand.sum(1, keepdims=True) / np.maximum(present.sum(1, keepdims=True), 1),
             present.any(1, keepdims=True).astype(np.float32)]
    for modele in choix.split("+"):
        probas = v[f"{modele}_probas"].astype(np.float32)
        blocs += [moyenne_masquee(v[f"{modele}_vecteurs"], masque),
                  moyenne_masquee(probas, masque),
                  np.where(masque[..., None], probas, 0).max((1, 2))]
        if modele == "hse":
            blocs.append(moyenne_masquee(v["hse_va"], masque))
    return np.hstack(blocs)


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
        self.faits = pd.read_csv(FICHIER) if FICHIER.exists() else pd.DataFrame()
        if not self.faits.empty and "classifieur" not in self.faits:      # lignes d'une version sans la colonne
            self.faits["classifieur"] = "lr"
            self.faits["parametres"] = [json.dumps({"C": c, "class_weight": None if pd.isna(w) else w})
                                        for c, w in zip(self.faits["C"], self.faits["class_weight"])]
        if not self.faits.empty and "f1_cv_probas" not in self.faits:     # lignes d'avant la calibration
            self.faits["f1_cv_probas"] = self.faits["f1_cv"].where(self.faits["classifieur"] != "svm")
        a = sum(1 if e == "videomae" else len(NB_IMAGES) * len(REGROUPEMENTS) for e in ENCODEURS)
        total = a + len(ENCODEURS) * (3 + 4 * len(AUTRES_CLASSIFIEURS)) + 3 * (1 + len(AUTRES_CLASSIFIEURS))
        self.barre = tqdm(total=total, desc="configurations")

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
            if pd.isna(self.faits.loc[indice, "f1_cv_probas"]):      # SVM d'une version sans calibration
                for nom, valeur in self.svm_calibre(X, json.loads(self.faits.loc[indice, "parametres"])).items():
                    self.faits.loc[indice, nom] = valeur
                self.faits.to_csv(FICHIER, index=False)
                self.barre.write(f"  {str(cle):110s} SVM calibré : F1 cv {self.faits.loc[indice, 'f1_cv_probas']:.3f}")
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
        self.barre.write(f"  {str(cle):110s} F1 cv {calibre['f1_cv_probas']:.3f}  F1 dev {calibre['f1_dev']:.3f}")
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
    visages = {choix: caracteristiques_visages(choix) for choix in VISAGES[1:]}

    for encodeur in ENCODEURS:
        exp.barre.write(f"\n=== {encodeur} ===")
        vecteurs = np.load(config.DATA / f"emb_{encodeur}.npy")
        un_seul = vecteurs.shape[1] == 1          # VideoMAE : un vecteur par extrait

        def scene(n, r):
            return vecteurs[:, 0].astype(np.float32) if un_seul else regrouper(vecteurs, n, r)

        # étape A : nombre d'images et regroupement, sans visages
        reglages = [(16, "moyenne")] if un_seul else [(n, r) for n in NB_IMAGES for r in REGROUPEMENTS]
        lignes_a = {(n, r): exp.evaluer(scene(n, r), etape="A", encodeur=encodeur, nb_images=n,
                                        regroupement=r, visages="aucun") for n, r in reglages}
        n, r = max(lignes_a, key=lambda c: lignes_a[c]["f1_cv"])
        entrees = {"aucun": scene(n, r)}
        entrees.update({c: np.hstack([entrees["aucun"], visages[c]]) for c in VISAGES[1:]})
        commun = dict(encodeur=encodeur, nb_images=n, regroupement=r)
        # étape B : visages, régression logistique
        candidats = {("aucun", "lr"): lignes_a[(n, r)]}
        candidats.update({(c, "lr"): exp.evaluer(entrees[c], etape="B", visages=c, **commun) for c in VISAGES[1:]})
        # étape D : autres classifieurs sur les mêmes entrées
        candidats.update({(c, k): exp.evaluer(entrees[c], k, etape="D", visages=c, **commun)
                          for c in VISAGES for k in AUTRES_CLASSIFIEURS})
        c, k = max(candidats, key=lambda x: candidats[x]["f1_cv_probas"])
        exp.barre.write(f"  retenu : {n} images, {r}, visages {c}, {k}  (F1 cv {candidats[(c, k)]['f1_cv_probas']:.3f})")
        exp.enregistrer(f"emb_{encodeur}", entrees[c], candidats[(c, k)])

    # étape C : visages seuls
    exp.barre.write("\n=== visages seuls ===")
    commun = dict(etape="C", encodeur="visages", nb_images=16, regroupement="moyenne")
    candidats = {(c, k): exp.evaluer(X, k, visages=c, **commun)
                 for c, X in visages.items() for k in ["lr"] + AUTRES_CLASSIFIEURS}
    c, k = max(candidats, key=lambda x: candidats[x]["f1_cv_probas"])
    exp.enregistrer("visages", visages[c], candidats[(c, k)])

    exp.barre.close()
    tableau = exp.faits
    print("\nMeilleure configuration par encodeur (F1 cv) :")
    print(tableau.loc[tableau.groupby("encodeur")["f1_cv_probas"].idxmax()]
          [["encodeur", "nb_images", "regroupement", "visages", "classifieur", "parametres", "f1_cv", "f1_cv_probas",
            "f1_dev", "exactitude_dev"]].sort_values("f1_cv_probas", ascending=False).round(3).to_string(index=False))
    print("\nMeilleur F1 cv (sur les probabilités) par encodeur et classifieur :")
    print(tableau.pivot_table(index="encodeur", columns="classifieur", values="f1_cv_probas", aggfunc="max").round(3))


if __name__ == "__main__":
    main()
