"""Fonctions communes aux scripts du point 5, mêmes conventions que les modules du point 4."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config

MODALITES = ["video", "texte", "audio"]


def charger_corpus():
    corpus = pd.read_csv(config.CORPUS, dtype={"film_id": str}, keep_default_na=False)
    config.DATA.mkdir(exist_ok=True)
    config.RESULTATS.mkdir(exist_ok=True)
    return corpus


def scores(y_vrai, y_predit):
    return {"f1_macro": f1_score(y_vrai, y_predit, average="macro"),
            "exactitude": accuracy_score(y_vrai, y_predit)}


def predire(probas):
    return np.array(config.EMOTIONS)[probas.argmax(axis=1)]


def probabilites_ordonnees(modele, X):
    # remet les colonnes dans l'ordre de config.EMOTIONS
    sorties = modele.predict_proba(X)
    probas = np.zeros((X.shape[0], len(config.EMOTIONS)))
    for colonne, classe in enumerate(modele.classes_):
        probas[:, config.EMOTIONS.index(classe)] = sorties[:, colonne]
    return probas


def charger_probas(dossier, nom, corpus):
    """Probabilités (tous les extraits) et probabilités hors pli (train) d'un modèle du point 4 ou 5."""
    fichier = np.load(dossier / f"probas_{nom}.npz")
    if not (fichier["video_id"] == corpus["video_id"].to_numpy(dtype=str)).all():
        raise ValueError(f"probas_{nom}.npz n'est pas dans l'ordre du corpus")
    if list(fichier["emotions"]) != config.EMOTIONS:
        raise ValueError(f"probas_{nom}.npz : émotions dans un autre ordre")
    return fichier["probabilites"], fichier["probabilites_hors_pli"]


def sauvegarder(nom, corpus, probas, hors_pli, y_dev, y_predit_dev):
    np.savez_compressed(config.RESULTATS / f"probas_{nom}.npz",
                        video_id=corpus["video_id"].to_numpy(dtype=str),
                        partition=corpus["partition"].to_numpy(dtype=str),
                        probabilites=probas, probabilites_hors_pli=hors_pli,
                        emotions=np.array(config.EMOTIONS))
    matrice(nom, y_dev, y_predit_dev, "dev")


def matrice(nom, y_vrai, y_predit, partie):
    m = confusion_matrix(y_vrai, y_predit, labels=config.EMOTIONS)
    figure, axe = plt.subplots(figsize=(7, 6))
    axe.imshow(m, cmap="Blues")
    axe.set_xticks(range(8), config.EMOTIONS, rotation=45, ha="right")
    axe.set_yticks(range(8), config.EMOTIONS)
    axe.set_xlabel("Prédiction")
    axe.set_ylabel("Émotion réelle")
    axe.set_title(f"{nom} ({partie})")
    for i in range(8):
        for j in range(8):
            axe.text(j, i, m[i, j], ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(config.RESULTATS / f"confusion_{nom}{'' if partie == 'dev' else '_' + partie}.png", dpi=150)
    plt.close(figure)


def bootstrap_films(y, predit_a, predit_b, films, tirages=2000):
    # IC 95 % de F1(a) - F1(b), en tirant les films avec remise
    rng = np.random.default_rng(config.GRAINE)
    liste = np.unique(films)
    indices = {f: np.where(films == f)[0] for f in liste}
    ecarts = []
    for _ in range(tirages):
        tirage = np.concatenate([indices[f] for f in rng.choice(liste, len(liste))])
        ecarts.append(scores(y[tirage], predit_a[tirage])["f1_macro"]
                      - scores(y[tirage], predit_b[tirage])["f1_macro"])
    return np.percentile(ecarts, [2.5, 97.5])


# ---------- vecteurs gelés des trois modalités (options B et C) ----------
# On reprend pour chaque modalité l'encodeur gelé le meilleur seul au point 4, avec son réglage retenu en CV :
#   vidéo  PE, 16 images (+ visages ViT et HSEmotion)         point4_video, N2 0,430
#   texte  e5-large sur description + dialogue concaténés     point4_texte, N2 0,330
#   audio  Whisper large-v3, couche 16 de l'encodeur          point4_audio, N2 0,265

def video_images():
    """(extraits, 16, 1024) : PE, un vecteur par image."""
    return np.load(config.DATA_VIDEO / "emb_pe.npy").astype(np.float32)


def visages_resume():
    """Visages résumés sur les 16 images (comme point4_video/05_video_n2.py, choix vit+hse)."""
    v = np.load(config.DATA_VIDEO / "visages.npz")
    masque = v["masque"]
    m = masque[..., None].astype(np.float32)

    def moyenne(valeurs):
        return (valeurs.astype(np.float32) * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)

    present = masque[..., 0]
    grand = np.where(present, v["boites"][:, :, 0, 4], 0)
    blocs = [present.mean(1, keepdims=True), v["nb_visages"].mean(1, keepdims=True),
             grand.sum(1, keepdims=True) / np.maximum(present.sum(1, keepdims=True), 1),
             present.any(1, keepdims=True).astype(np.float32)]
    for modele in ["vit", "hse"]:
        probas = v[f"{modele}_probas"].astype(np.float32)
        blocs += [moyenne(v[f"{modele}_vecteurs"]), moyenne(probas), np.where(masque[..., None], probas, 0).max((1, 2))]
    blocs.append(moyenne(v["hse_va"]))
    return np.hstack(blocs)


def visages_par_image():
    """(extraits, 16, 777) : pour chaque image, le plus grand visage (ViT), 0 si aucun (comme 06_video_n3.py)."""
    v = np.load(config.DATA_VIDEO / "visages.npz")
    present = v["masque"][:, :, 0]
    blocs = [present[..., None], v["boites"][:, :, 0, 4:5], v["vit_vecteurs"][:, :, 0], v["vit_probas"][:, :, 0]]
    return np.concatenate([b.astype(np.float32) for b in blocs], axis=-1)


def audio_resume():
    """(extraits, 1280) : Whisper, couche 16, moyenne dans le temps."""
    return np.load(config.DATA_AUDIO / "couches_whisper.npy", mmap_mode="r")[:, 16, :1280].astype(np.float32)


def audio_tranches():
    """(extraits, 16, 1280) : Whisper, moyenne des couches sur 16 tranches de même durée."""
    return np.load(config.DATA_AUDIO / "seg_whisper.npy").astype(np.float32)


def texte_e5(corpus):
    """(extraits, 1025) : e5-large (description + dialogue, préfixe « query: »), plus l'indicateur de dialogue."""
    cache = config.DATA / "e5_concatenation.npy"
    if not cache.exists():
        import torch
        from sentence_transformers import SentenceTransformer
        modele = SentenceTransformer("intfloat/e5-large-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        textes = ("query: " + (corpus["caption"] + " " + corpus["dialogue"]).str.strip()).tolist()
        np.save(cache, modele.encode(textes, batch_size=32, show_progress_bar=True, normalize_embeddings=True))
    return np.hstack([np.load(cache), corpus[["a_dialogue"]].to_numpy(dtype=np.float32)])
