"""Synthese des configurations et matrices des meilleurs modeles sur dev."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

import config
import utils


def matrice(nom):
    with np.load(config.DATA / f"probas_{nom}_optimise.npz", allow_pickle=True) as p:
        probabilites = p["probabilites"]
        emotions = list(p["emotions"])
        partitions = p["partition"]
        y = p["y"]
    dev = partitions == "dev"
    prediction = np.array(emotions)[probabilites[dev].argmax(axis=1)]

    detail = pd.DataFrame({
        "emotion": emotions,
        "rappel_dev": recall_score(y[dev], prediction, labels=emotions,
                                   average=None, zero_division=0),
        "effectif_dev": [(y[dev] == e).sum() for e in emotions],
    })
    detail.to_csv(config.RESULTATS / f"rappels_{nom}_optimise.csv", index=False)

    cm = confusion_matrix(y[dev], prediction, labels=emotions)
    figure, axe = plt.subplots(figsize=(8, 7))
    image = axe.imshow(cm, cmap="Blues")
    axe.set_xticks(range(len(emotions)), emotions, rotation=45, ha="right")
    axe.set_yticks(range(len(emotions)), emotions)
    axe.set_xlabel("Prediction")
    axe.set_ylabel("Emotion reelle")
    axe.set_title(f"{nom} - meilleur classifieur (dev)")
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            axe.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axe)
    figure.tight_layout()
    figure.savefig(config.RESULTATS / f"confusion_{nom}_optimise.png", dpi=150)
    plt.close(figure)


def main():
    utils.preparer_dossiers()
    scores = pd.read_csv(config.RESULTATS / "comparaison_classifieurs.csv")
    meilleurs = scores.loc[scores.groupby("configuration")["f1_macro_dev"].idxmax()]
    meilleurs = meilleurs.sort_values("f1_macro_dev", ascending=False)
    meilleurs.to_csv(config.RESULTATS / "synthese.csv", index=False)
    print(meilleurs.to_string(index=False))

    for nom in ("audio_ast_brut_plus_n1", "video_clip_plus_n1"):
        matrice(nom)


if __name__ == "__main__":
    main()
