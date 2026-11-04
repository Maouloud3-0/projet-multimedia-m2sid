"""Option C — petit modèle multimodal : une tête d'attention sur les morceaux des trois modalités.

C'est la tête du N3 de point4_video/06_video_n3.py, étendue à trois modalités (les encodeurs restent gelés) :
    morceaux    16 images     PE + plus grand visage de l'image (ViT)        1 801 valeurs
                1 texte       e5-large, description + dialogue               1 025 valeurs
                16 tranches   Whisper, moyenne des couches sur la tranche    1 280 valeurs
    projection  par modalité : standardisation (sur les extraits d'apprentissage) -> linéaire -> GELU -> dropout,
                plus un vecteur appris qui indique la modalité
    mélange     0 ou 1 couche de self-attention (les morceaux se regardent entre eux : image <-> son <-> texte)
    attention   un score par morceau -> softmax sur les 33 morceaux -> moyenne pondérée
    sortie      linéaire -> 8 émotions
Pendant l'apprentissage, on retire au hasard des morceaux et parfois une modalité entière, pour que le modèle
ne s'appuie pas seulement sur la vidéo.

Réglages choisis par Optuna (30 essais), chacun évalué en CV sur train (5 plis groupés par film), puis
le modèle retenu est appris avec 3 graines (probabilités moyennées) ; dev ne sert qu'à mesurer.
On relève aussi la part d'attention donnée à chaque modalité sur dev.

Sorties : resultats/scores_attention.csv, resultats/essais_attention.csv, resultats/probas_attention.npz
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

ESSAIS = 30
GRAINES = [0, 1, 2]
TAILLE_LOT = 64
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Multimodal(nn.Module):
    def __init__(self, dimensions, cachee, dropout, couches):
        super().__init__()
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(d, cachee), nn.GELU(), nn.Dropout(dropout))
                                         for d in dimensions)
        self.modalite = nn.Parameter(torch.zeros(len(dimensions), cachee))
        self.melange = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cachee, 4, 2 * cachee, dropout, batch_first=True, norm_first=True),
            couches, enable_nested_tensor=False) if couches else None
        self.score = nn.Linear(cachee, 1)
        self.sortie = nn.Sequential(nn.Dropout(dropout), nn.Linear(cachee, len(config.EMOTIONS)))

    def forward(self, entrees, gardes):
        z = torch.cat([proj(x) + self.modalite[i] for i, (proj, x) in enumerate(zip(self.projections, entrees))], 1)
        if self.melange is not None:
            z = self.melange(z, src_key_padding_mask=~gardes)
        poids = self.score(z).squeeze(-1).masked_fill(~gardes, float("-inf")).softmax(dim=1)
        return self.sortie((poids.unsqueeze(-1) * z).sum(1)), poids


def apprendre(X, y, apprentissage, evaluation, p, graine, avec_poids=False):
    """X : liste de 3 tenseurs (extraits, morceaux, dimension). Renvoie les probabilités sur `evaluation`."""
    torch.manual_seed(graine)
    generateur = torch.Generator(device=APPAREIL).manual_seed(graine)
    normes = [(x[apprentissage].mean((0, 1), keepdim=True), x[apprentissage].std((0, 1), keepdim=True) + 1e-6)
              for x in X]
    normer = lambda indices: [(x[indices] - m) / s for x, (m, s) in zip(X, normes)]
    Xa, ya = normer(apprentissage), y[apprentissage]
    morceaux = [x.shape[1] for x in X]
    effectifs = torch.bincount(ya, minlength=len(config.EMOTIONS)).float()
    perte = nn.CrossEntropyLoss(weight=len(ya) / (len(config.EMOTIONS) * effectifs.clamp(min=1)))
    modele = Multimodal([x.shape[-1] for x in X], p["cachee"], p["dropout"], p["couches"]).to(APPAREIL)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

    modele.train()
    for _ in range(p["epoques"]):
        ordre = torch.randperm(len(ya), device=APPAREIL, generator=generateur)
        for debut in range(0, len(ya), TAILLE_LOT):
            lot = ordre[debut:debut + TAILLE_LOT]
            n = len(lot)
            gardes = torch.rand(n, sum(morceaux), device=APPAREIL, generator=generateur) >= p["morceaux_retires"]
            # modalité entière retirée : par extrait, avec la probabilité choisie
            retirees = torch.rand(n, len(X), device=APPAREIL, generator=generateur) < p["modalite_retiree"]
            gardes &= ~torch.repeat_interleave(retirees, torch.tensor(morceaux, device=APPAREIL), dim=1)
            gardes[torch.arange(n), torch.randint(sum(morceaux), (n,), device=APPAREIL,
                                                  generator=generateur)] = True     # au moins un morceau
            sorties, _ = modele([x[lot] for x in Xa], gardes)
            optimiseur.zero_grad()
            perte(sorties, ya[lot]).backward()
            optimiseur.step()

    modele.eval()
    with torch.no_grad():
        tous = torch.ones(len(evaluation), sum(morceaux), dtype=torch.bool, device=APPAREIL)
        sorties, poids = modele(normer(evaluation), tous)
    probas = sorties.softmax(-1).cpu().numpy()
    if not avec_poids:
        return probas
    bornes = np.cumsum([0] + morceaux)
    return probas, np.stack([poids[:, a:b].sum(1).cpu().numpy() for a, b in zip(bornes[:-1], bornes[1:])], 1)


def main():
    print(f"appareil : {APPAREIL}")
    corpus = utils.charger_corpus()
    train = np.where(corpus["partition"] == "train")[0]
    dev = (corpus["partition"] == "dev").to_numpy()
    etiquettes = corpus["emotion"].to_numpy()
    y = torch.tensor([config.EMOTIONS.index(e) for e in etiquettes], device=APPAREIL)
    films = corpus["film_id"].to_numpy()[train]
    plis = [(train[a], train[t]) for a, t in GroupKFold(5).split(train, groups=films)]
    X = [torch.tensor(np.concatenate([utils.video_images(), utils.visages_par_image()], -1)),   # même ordre
         torch.tensor(utils.texte_e5(corpus)[:, None]),                                       # que MODALITES
         torch.tensor(utils.audio_tranches())]
    X = [x.to(APPAREIL) for x in X]
    print("morceaux :", [tuple(x.shape[1:]) for x in X])

    def f1_cv(p, graines):
        hors_pli = np.full((len(etiquettes), len(config.EMOTIONS)), np.nan)
        scores = []
        for apprentissage, validation in plis:
            probas = np.mean([apprendre(X, y, apprentissage, validation, p, g) for g in graines], axis=0)
            hors_pli[validation] = probas
            scores.append(f1_score(etiquettes[validation], utils.predire(probas), average="macro"))
        return float(np.mean(scores)), hors_pli

    def objectif(essai):
        p = {"cachee": essai.suggest_categorical("cachee", [128, 256]),
             "couches": essai.suggest_categorical("couches", [0, 1]),
             "dropout": essai.suggest_float("dropout", 0.1, 0.6),
             "lr": essai.suggest_float("lr", 1e-4, 3e-3, log=True),
             "weight_decay": essai.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
             "epoques": essai.suggest_int("epoques", 5, 40),
             "morceaux_retires": essai.suggest_float("morceaux_retires", 0.0, 0.5),
             "modalite_retiree": essai.suggest_float("modalite_retiree", 0.0, 0.4)}
        return f1_cv(p, GRAINES[:1])[0]

    etude = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.GRAINE))
    with tqdm(total=ESSAIS, desc="attention") as barre:
        etude.optimize(objectif, n_trials=ESSAIS,
                       callbacks=[lambda e, t: (barre.update(), barre.set_postfix(meilleur=f"{e.best_value:.3f}"))])
    etude.trials_dataframe().to_csv(config.RESULTATS / "essais_attention.csv", index=False)

    # modèle retenu : 3 graines, en CV (probabilités hors pli) puis sur tout train
    p = dict(etude.best_params)
    score_cv, hors_pli = f1_cv(p, GRAINES)
    tous = np.arange(len(etiquettes))
    sorties = [apprendre(X, y, train, tous, p, g, avec_poids=True) for g in GRAINES]
    probas = np.mean([s[0] for s in sorties], axis=0)
    parts = np.mean([s[1] for s in sorties], axis=0)[dev].mean(0)     # part d'attention par modalité sur dev
    s = utils.scores(etiquettes[dev], utils.predire(probas[dev]))
    utils.sauvegarder("attention", corpus, probas, hors_pli, etiquettes[dev], utils.predire(probas[dev]))

    ligne = {"parametres": json.dumps(p), "f1_cv_recherche": etude.best_value,
             "f1_cv": utils.scores(etiquettes[train], utils.predire(hors_pli[train]))["f1_macro"],
             "f1_dev": s["f1_macro"], "exactitude_dev": s["exactitude"],
             **{f"attention_{m}": parts[i] for i, m in enumerate(utils.MODALITES)}}
    pd.DataFrame([ligne]).to_csv(config.RESULTATS / "scores_attention.csv", index=False)
    print(f"retenu : {p}")
    print(f"F1 cv {ligne['f1_cv']:.3f} (3 graines)  F1 dev {s['f1_macro']:.3f}  exactitude dev {s['exactitude']:.3f}")
    print("part d'attention sur dev : " + ", ".join(f"{m} {parts[i]:.0%}" for i, m in enumerate(utils.MODALITES)))


if __name__ == "__main__":
    main()
