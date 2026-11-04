"""N3 : fine-tuning de plusieurs modèles pré-entraînés.

Pour chaque modèle et chaque taux d'apprentissage : validation croisée à 5 plis
sur train (groupée par film). Le taux retenu est celui qui a le meilleur F1 en CV,
puis un modèle final est entraîné sur tout train avec ce taux.
Tout s'enchaîne : python 05_texte_finetuning.py
"""

import gc
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import config
import utils

# du plus rapide au plus long
MODELES = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "goemotions": "SamLowe/roberta-base-go_emotions",
    "ekman": "j-hartmann/emotion-english-roberta-large",
    "e5": "intfloat/e5-large-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "roberta": "FacebookAI/roberta-large",
}
# valeurs recommandées pour BERT / RoBERTa (Devlin et al., 2019 ; Liu et al., 2019)
TAUX = [1e-5, 2e-5, 3e-5]
EPOQUES = 4
LOT = 16
LONGUEUR_MAX = 256

APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"
DEMI = APPAREIL == "cuda"   # calcul en float16 sur carte graphique


def lots(tokenizer, descriptions, dialogues, etiquettes=None, melanger=False):
    # padding à la longueur du lot, pas à LONGUEUR_MAX
    indices = list(range(len(descriptions)))

    def assembler(selection):
        lot = tokenizer([descriptions[i] for i in selection], [dialogues[i] for i in selection],
                        truncation=True, max_length=LONGUEUR_MAX, padding=True, return_tensors="pt")
        if etiquettes is not None:
            lot["labels"] = torch.tensor([etiquettes[i] for i in selection])
        return lot

    return DataLoader(indices, batch_size=LOT, shuffle=melanger, collate_fn=assembler)


def entrainer(chemin, taux, descriptions, dialogues, etiquettes):
    tokenizer = AutoTokenizer.from_pretrained(chemin)
    # nouvelle dernière couche à 8 sorties
    modele = AutoModelForSequenceClassification.from_pretrained(
        chemin, num_labels=len(config.EMOTIONS), ignore_mismatched_sizes=True).to(APPAREIL)

    # même formule que class_weight="balanced"
    effectifs = np.bincount(etiquettes, minlength=len(config.EMOTIONS))
    poids = torch.tensor(len(etiquettes) / (len(config.EMOTIONS) * effectifs), dtype=torch.float)
    perte = torch.nn.CrossEntropyLoss(weight=poids.to(APPAREIL))

    donnees = lots(tokenizer, descriptions, dialogues, etiquettes, melanger=True)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux, weight_decay=0.01)
    etapes = EPOQUES * len(donnees)
    planning = get_linear_schedule_with_warmup(optimiseur, int(0.1 * etapes), etapes)
    precision = torch.amp.GradScaler(enabled=DEMI)

    modele.train()
    for epoque in range(EPOQUES):
        for lot in tqdm(donnees, desc=f"  époque {epoque + 1}/{EPOQUES}", leave=False):
            lot = lot.to(APPAREIL)
            etiquettes_lot = lot.pop("labels")
            with torch.autocast(device_type=APPAREIL, dtype=torch.float16, enabled=DEMI):
                logits = modele(**lot).logits
            erreur = perte(logits.float(), etiquettes_lot)
            optimiseur.zero_grad()
            precision.scale(erreur).backward()
            precision.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            precision.step(optimiseur)
            precision.update()
            planning.step()
    return tokenizer, modele


@torch.no_grad()
def probabilites(tokenizer, modele, descriptions, dialogues):
    modele.eval()
    sorties = []
    for lot in lots(tokenizer, descriptions, dialogues):
        with torch.autocast(device_type=APPAREIL, dtype=torch.float16, enabled=DEMI):
            logits = modele(**lot.to(APPAREIL)).logits
        sorties.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(sorties)


def vider_memoire():
    # sinon l'ancien modèle reste en mémoire pendant le chargement du suivant
    gc.collect()
    if APPAREIL == "cuda":
        torch.cuda.empty_cache()


def affiner(nom, corpus):
    chemin = MODELES[nom]
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    films = corpus["film_id"].to_numpy()
    descriptions = corpus["caption"].tolist()
    dialogues = corpus["dialogue"].tolist()   # "" si pas de dialogue
    classes = np.array([config.EMOTIONS.index(e) for e in y])

    def selection(masque):
        indices = np.where(masque)[0]
        return [descriptions[i] for i in indices], [dialogues[i] for i in indices], classes[indices]

    def f1(probas, partie):
        return utils.scores(y[partie], np.array(config.EMOTIONS)[probas[partie].argmax(1)])["f1_macro"]

    # 1) CV sur train pour chaque taux : chaque extrait est prédit par un modèle qui n'a pas vu son film
    indices_train = np.where(train)[0]
    plis = list(GroupKFold(n_splits=5).split(indices_train, groups=films[train]))
    lignes, hors_pli = [], {}
    for taux in TAUX:
        torch.manual_seed(config.GRAINE)
        debut = time.time()
        hors_pli[taux] = np.full((len(corpus), len(config.EMOTIONS)), np.nan)
        for pli, (app, val) in enumerate(plis):
            print(f"{nom}  lr {taux:g}  pli {pli + 1}/5")
            masque_app = np.zeros(len(corpus), bool)
            masque_app[indices_train[app]] = True
            masque_val = np.zeros(len(corpus), bool)
            masque_val[indices_train[val]] = True
            tokenizer, modele = entrainer(chemin, taux, *selection(masque_app))
            d, t, _ = selection(masque_val)
            hors_pli[taux][masque_val] = probabilites(tokenizer, modele, d, t)
            del modele
            vider_memoire()
        lignes.append({"modele": nom, "lr": taux, "f1_cv": f1(hors_pli[taux], train),
                       "minutes": (time.time() - debut) / 60})
        print(f"{nom}  lr {taux:g}  F1 cv {lignes[-1]['f1_cv']:.3f}  ({lignes[-1]['minutes']:.0f} min)")

    # 2) meilleur taux en CV, modèle final sur tout train -> probabilités de dev et test
    meilleur = max(lignes, key=lambda l: l["f1_cv"])["lr"]
    print(f"{nom}  modèle final avec lr {meilleur:g}")
    torch.manual_seed(config.GRAINE)
    tokenizer, modele = entrainer(chemin, meilleur, *selection(train))
    probas = probabilites(tokenizer, modele, descriptions, dialogues)
    del modele
    vider_memoire()

    predit_dev = np.array(config.EMOTIONS)[probas[dev].argmax(1)]
    s = utils.scores(y[dev], predit_dev)
    utils.sauvegarder(f"ft_{nom}", corpus, probas, hors_pli[meilleur], y[dev], predit_dev)
    for ligne in lignes:
        ligne["retenu"] = ligne["lr"] == meilleur
        ligne["f1_dev"] = s["f1_macro"] if ligne["retenu"] else np.nan
    print(f"ft_{nom} : lr {meilleur:g}  F1 cv {max(l['f1_cv'] for l in lignes):.3f}  "
          f"F1 dev {s['f1_macro']:.3f}  exactitude dev {s['exactitude']:.3f}\n")
    return lignes


def main():
    corpus = utils.charger_corpus()
    fichier = config.RESULTATS / "scores_finetuning.csv"
    # modèles déjà présents dans le tableau (par exemple après une coupure) : on ne les refait pas
    faits = set(pd.read_csv(fichier)["modele"]) if fichier.exists() else set()
    for nom in MODELES:
        if nom in faits:
            print(f"{nom} déjà affiné, ignoré")
            continue
        try:
            lignes = affiner(nom, corpus)
        except Exception as erreur:   # un modèle qui échoue ne doit pas arrêter les suivants
            print(f"{nom} : échec ({erreur})\n")
            vider_memoire()
            continue
        # on enregistre après chaque modèle pour ne rien perdre si la nuit s'arrête
        tableau = pd.DataFrame(lignes)
        if fichier.exists():
            tableau = pd.concat([pd.read_csv(fichier), tableau])
        tableau.to_csv(fichier, index=False)


if __name__ == "__main__":
    main()
