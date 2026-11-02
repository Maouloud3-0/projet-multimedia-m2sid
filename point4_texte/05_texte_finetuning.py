"""N3 : fine-tuning d'un modèle pré-entraîné (CV 5 plis sur train, puis modèle final sur train)."""

import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import config
import utils

# réglages classiques pour RoBERTa, pas de recherche (un entraînement = plusieurs minutes)
MODELE = "j-hartmann/emotion-english-roberta-large"
NOM = "ft_ekman"
EPOQUES = 4
TAUX = 2e-5
LOT = 16
LONGUEUR_MAX = 256

APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"


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


def entrainer(descriptions, dialogues, etiquettes):
    tokenizer = AutoTokenizer.from_pretrained(MODELE)
    # ekman a 7 classes, on remplace la dernière couche par 8 sorties
    modele = AutoModelForSequenceClassification.from_pretrained(
        MODELE, num_labels=len(config.EMOTIONS), ignore_mismatched_sizes=True).to(APPAREIL)

    # même formule que class_weight="balanced"
    effectifs = np.bincount(etiquettes, minlength=len(config.EMOTIONS))
    poids = torch.tensor(len(etiquettes) / (len(config.EMOTIONS) * effectifs), dtype=torch.float)
    perte = torch.nn.CrossEntropyLoss(weight=poids.to(APPAREIL))

    donnees = lots(tokenizer, descriptions, dialogues, etiquettes, melanger=True)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=TAUX, weight_decay=0.01)
    etapes = EPOQUES * len(donnees)
    planning = get_linear_schedule_with_warmup(optimiseur, int(0.1 * etapes), etapes)
    precision = torch.amp.GradScaler(enabled=APPAREIL == "cuda")

    modele.train()
    for epoque in range(EPOQUES):
        debut = time.time()
        for lot in tqdm(donnees, desc=f"époque {epoque + 1}/{EPOQUES}"):
            lot = lot.to(APPAREIL)
            etiquettes_lot = lot.pop("labels")
            with torch.autocast(device_type=APPAREIL, dtype=torch.float16, enabled=APPAREIL == "cuda"):
                logits = modele(**lot).logits
            erreur = perte(logits.float(), etiquettes_lot)
            optimiseur.zero_grad()
            precision.scale(erreur).backward()
            precision.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            precision.step(optimiseur)
            precision.update()
            planning.step()
        print(f"  époque {epoque + 1} : {time.time() - debut:.0f} s")
    return tokenizer, modele


@torch.no_grad()
def probabilites(tokenizer, modele, descriptions, dialogues):
    modele.eval()
    sorties = []
    for lot in lots(tokenizer, descriptions, dialogues):
        with torch.autocast(device_type=APPAREIL, dtype=torch.float16, enabled=APPAREIL == "cuda"):
            logits = modele(**lot.to(APPAREIL)).logits
        sorties.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(sorties)


def liberer(modele):
    del modele
    if APPAREIL == "cuda":
        torch.cuda.empty_cache()


def main():
    torch.manual_seed(config.GRAINE)
    corpus = utils.charger_corpus()
    train = (corpus["partition"] == "train").to_numpy()
    dev = (corpus["partition"] == "dev").to_numpy()
    y = corpus["emotion"].to_numpy()
    films = corpus["film_id"].to_numpy()
    descriptions = corpus["caption"].tolist()
    dialogues = corpus["dialogue"].tolist()   # "" si pas de dialogue
    classes = np.array([config.EMOTIONS.index(e) for e in y])
    print(f"Affinage de {MODELE} sur {APPAREIL}")

    def selection(masque):
        indices = np.where(masque)[0]
        return [descriptions[i] for i in indices], [dialogues[i] for i in indices], classes[indices]

    # 1) CV sur train : chaque extrait est prédit par un modèle qui n'a pas vu son film
    hors_pli = np.full((len(corpus), len(config.EMOTIONS)), np.nan)
    indices_train = np.where(train)[0]
    for pli, (app, val) in enumerate(GroupKFold(n_splits=5).split(indices_train, groups=films[train])):
        print(f"\nPli {pli + 1}/5")
        masque_app = np.zeros(len(corpus), bool)
        masque_app[indices_train[app]] = True
        masque_val = np.zeros(len(corpus), bool)
        masque_val[indices_train[val]] = True
        tokenizer, modele = entrainer(*selection(masque_app))
        d, t, _ = selection(masque_val)
        hors_pli[masque_val] = probabilites(tokenizer, modele, d, t)
        liberer(modele)
    f1_cv = utils.scores(y[train], np.array(config.EMOTIONS)[hors_pli[train].argmax(1)])["f1_macro"]

    # 2) modèle final sur tout train -> probabilités de dev et test
    print("\nModèle final (tout train)")
    tokenizer, modele = entrainer(*selection(train))
    probas = probabilites(tokenizer, modele, descriptions, dialogues)
    liberer(modele)

    predit_dev = np.array(config.EMOTIONS)[probas[dev].argmax(1)]
    s = utils.scores(y[dev], predit_dev)
    utils.sauvegarder(NOM, corpus, probas, hors_pli, y[dev], predit_dev)
    print(f"\n{NOM} : F1 cv {f1_cv:.3f}  F1 dev {s['f1_macro']:.3f}  exactitude dev {s['exactitude']:.3f}")


if __name__ == "__main__":
    main()
