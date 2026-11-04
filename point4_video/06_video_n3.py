"""N3 : couche d'attention sur les 16 images, apprise sur train (les encodeurs restent gelés).

Au lieu de faire la moyenne des 16 images (05), un petit réseau apprend quelles images comptent :
    par image   vecteur de l'encodeur (02) + visage le plus grand de l'image (03), 0 si aucun visage
                -> standardisation (sur les extraits d'apprentissage) -> linéaire -> GELU -> dropout
    attention   un score par image -> softmax -> moyenne pondérée des images
    sortie      linéaire -> 8 émotions
Apprentissage : AdamW, entropie croisée avec poids des classes équilibrés, nombre d'époques fixé
(pas d'arrêt précoce sur le pli de validation), images retirées au hasard pendant l'apprentissage.

Réglages choisis par Optuna (30 essais), chacun évalué en CV sur train (5 plis groupés par film) :
    visages, dimension cachée, dropout, taux d'apprentissage, weight decay, époques, part d'images retirées.
Le modèle retenu est ensuite appris avec 3 graines (probabilités moyennées) ; dev ne sert qu'à mesurer.

Sorties : resultats/scores_n3.csv (une ligne par encodeur, reprise possible après une coupure),
          resultats/essais_n3_<encodeur>.csv (tous les essais d'Optuna),
          resultats/probas_att_<encodeur>.npz (probabilités et probabilités hors pli pour la fusion).
"""

import json

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from tqdm import tqdm

import config
import utils

ENCODEURS = ["pe", "siglip2", "clip_l", "clip_h", "clip_b", "dinov2"]
VISAGES = ["aucun", "vit", "hse", "vit+hse"]
ESSAIS = 30
GRAINES = [0, 1, 2]
TAILLE_LOT = 64
FICHIER = config.RESULTATS / "scores_n3.csv"
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def visages_par_image(choix):
    """Pour chaque image, le plus grand visage : présence, aire, puis vecteur et probabilités du modèle choisi."""
    v = np.load(config.DATA / "visages.npz")
    present = v["masque"][:, :, 0]
    blocs = [present[..., None].astype(np.float32), v["boites"][:, :, 0, 4:5]]
    for modele in choix.split("+"):
        blocs += [v[f"{modele}_vecteurs"][:, :, 0], v[f"{modele}_probas"][:, :, 0]]
        if modele == "hse":
            blocs.append(v["hse_va"][:, :, 0])
    return np.concatenate([b.astype(np.float32) for b in blocs], axis=-1)


class Attention(nn.Module):
    def __init__(self, dimension, cachee, dropout):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(dimension, cachee), nn.GELU(), nn.Dropout(dropout))
        self.score = nn.Linear(cachee, 1)
        self.sortie = nn.Sequential(nn.Dropout(dropout), nn.Linear(cachee, len(config.EMOTIONS)))

    def forward(self, x, gardees=None):
        z = self.image(x)                                   # (lot, images, cachee)
        s = self.score(z).squeeze(-1)                       # (lot, images)
        if gardees is not None:
            s = s.masked_fill(~gardees, float("-inf"))
        poids = s.softmax(dim=1)
        return self.sortie((poids.unsqueeze(-1) * z).sum(1)), poids


def apprendre(X, y, apprentissage, evaluation, p, graine):
    """Apprend sur les extraits `apprentissage`, renvoie les probabilités sur `evaluation`."""
    torch.manual_seed(graine)
    generateur = torch.Generator(device=APPAREIL).manual_seed(graine)
    moyenne = X[apprentissage].mean((0, 1), keepdim=True)
    ecart = X[apprentissage].std((0, 1), keepdim=True) + 1e-6
    Xa, ya = (X[apprentissage] - moyenne) / ecart, y[apprentissage]
    effectifs = torch.bincount(ya, minlength=len(config.EMOTIONS)).float()
    perte = nn.CrossEntropyLoss(weight=len(ya) / (len(config.EMOTIONS) * effectifs.clamp(min=1)))
    modele = Attention(X.shape[-1], p["cachee"], p["dropout"]).to(APPAREIL)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

    modele.train()
    for _ in range(p["epoques"]):
        ordre = torch.randperm(len(ya), device=APPAREIL, generator=generateur)
        for debut in range(0, len(ya), TAILLE_LOT):
            lot = ordre[debut:debut + TAILLE_LOT]
            gardees = torch.rand(len(lot), X.shape[1], device=APPAREIL, generator=generateur) >= p["images_retirees"]
            gardees[torch.arange(len(lot)), torch.randint(X.shape[1], (len(lot),), device=APPAREIL,
                                                          generator=generateur)] = True   # au moins une image
            sorties, _ = modele(Xa[lot], gardees)
            optimiseur.zero_grad()
            perte(sorties, ya[lot]).backward()
            optimiseur.step()

    modele.eval()
    with torch.no_grad():
        sorties, _ = modele((X[evaluation] - moyenne) / ecart)
    return sorties.softmax(-1).cpu().numpy()


class Experience:
    def __init__(self):
        self.corpus = utils.charger_corpus()
        self.train = np.where(self.corpus["partition"] == "train")[0]
        self.dev = (self.corpus["partition"] == "dev").to_numpy()
        self.etiquettes = self.corpus["emotion"].to_numpy()
        self.y = torch.tensor([config.EMOTIONS.index(e) for e in self.etiquettes], device=APPAREIL)
        films = self.corpus["film_id"].to_numpy()[self.train]
        self.plis = [(self.train[a], self.train[t])
                     for a, t in GroupKFold(5).split(self.train, groups=films)]
        self.faits = pd.read_csv(FICHIER) if FICHIER.exists() else pd.DataFrame()
        self.visages = {c: torch.tensor(visages_par_image(c)) for c in VISAGES[1:]}

    def entrees(self, scenes, choix):
        # gardées en mémoire centrale, envoyées sur le GPU seulement pendant l'essai
        return scenes if choix == "aucun" else torch.cat([scenes, self.visages[choix]], dim=-1)

    def f1_cv(self, X, p, graines):
        """F1 macro moyen sur les 5 plis ; probabilités moyennées sur les graines."""
        hors_pli = np.full((len(self.etiquettes), len(config.EMOTIONS)), np.nan)
        scores = []
        for apprentissage, validation in self.plis:
            probas = np.mean([apprendre(X, self.y, apprentissage, validation, p, g) for g in graines], axis=0)
            hors_pli[validation] = probas
            scores.append(f1_score(self.etiquettes[validation], utils.predire(probas), average="macro"))
        return float(np.mean(scores)), hors_pli

    def traiter(self, encodeur):
        if not self.faits.empty and (self.faits["encodeur"] == encodeur).any():
            print(f"{encodeur} déjà fait")
            return
        scenes = torch.tensor(np.load(config.DATA / f"emb_{encodeur}.npy").astype(np.float32))
        entrees = {}

        def objectif(essai):
            choix = essai.suggest_categorical("visages", VISAGES)
            p = {"cachee": essai.suggest_categorical("cachee", [128, 256, 512]),
                 "dropout": essai.suggest_float("dropout", 0.1, 0.6),
                 "lr": essai.suggest_float("lr", 1e-4, 3e-3, log=True),
                 "weight_decay": essai.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
                 "epoques": essai.suggest_int("epoques", 5, 40),
                 "images_retirees": essai.suggest_float("images_retirees", 0.0, 0.5)}
            if choix not in entrees:
                entrees[choix] = self.entrees(scenes, choix)
            return self.f1_cv(entrees[choix].to(APPAREIL), p, GRAINES[:1])[0]

        etude = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.GRAINE))
        with tqdm(total=ESSAIS, desc=encodeur) as barre:
            etude.optimize(objectif, n_trials=ESSAIS,
                           callbacks=[lambda e, t: (barre.update(), barre.set_postfix(meilleur=f"{e.best_value:.3f}"))])
        etude.trials_dataframe().to_csv(config.RESULTATS / f"essais_n3_{encodeur}.csv", index=False)

        # modèle retenu : 3 graines, en CV (probabilités hors pli) puis sur tout train
        p = dict(etude.best_params)
        choix = p.pop("visages")
        X = entrees[choix].to(APPAREIL)
        f1_cv, hors_pli = self.f1_cv(X, p, GRAINES)
        tous = np.arange(len(self.etiquettes))
        probas = np.mean([apprendre(X, self.y, self.train, tous, p, g) for g in GRAINES], axis=0)
        s = utils.scores(self.etiquettes[self.dev], utils.predire(probas[self.dev]))
        utils.sauvegarder(f"att_{encodeur}", self.corpus, probas, hors_pli,
                          self.etiquettes[self.dev], utils.predire(probas[self.dev]))

        ligne = {"encodeur": encodeur, "visages": choix, "parametres": json.dumps(p),
                 "f1_cv_recherche": etude.best_value, "f1_cv": f1_cv,
                 "f1_dev": s["f1_macro"], "exactitude_dev": s["exactitude"]}
        self.faits = pd.concat([self.faits, pd.DataFrame([ligne])], ignore_index=True)
        self.faits.to_csv(FICHIER, index=False)
        print(f"  retenu : visages {choix}, {p}")
        print(f"  F1 cv {f1_cv:.3f} (3 graines)  F1 dev {s['f1_macro']:.3f}  exactitude dev {s['exactitude']:.3f}")


def main():
    print(f"appareil : {APPAREIL}")
    exp = Experience()
    for encodeur in ENCODEURS:
        exp.traiter(encodeur)
    print("\nRésultats N3 (F1 cv) :")
    print(exp.faits.drop(columns="parametres").sort_values("f1_cv", ascending=False)
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
