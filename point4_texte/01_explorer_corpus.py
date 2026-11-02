"""Statistiques du corpus texte : partition, émotions, présence et longueur des textes."""

import pandas as pd

import config


def charger_corpus():
    # keep_default_na=False : un dialogue absent reste une chaîne vide, pas NaN
    return pd.read_csv(config.CORPUS, dtype={"film_id": str}, keep_default_na=False)


def main():
    corpus = charger_corpus()
    ordre = ["train", "dev", "test"]

    print("=== Extraits et films par partition ===")
    resume = corpus.groupby("partition").agg(films=("film_id", "nunique"),
                                             extraits=("video_id", "count"))
    print(resume.loc[ordre])

    print("\n=== Émotions par partition (en %) ===")
    emotions = pd.crosstab(corpus["emotion"], corpus["partition"], normalize="columns") * 100
    print(emotions[ordre].round(1))

    print("\n=== Dialogue anglais disponible ===")
    dialogue = corpus.groupby("partition")["a_dialogue"].agg(["sum", "mean"])
    dialogue.columns = ["avec_dialogue", "proportion"]
    print(dialogue.loc[ordre].round(3))

    print("\n=== Dialogue disponible selon l'émotion (en %) ===")
    print((corpus.groupby("emotion")["a_dialogue"].mean() * 100).round(1))

    print("\n=== Longueur moyenne (mots) ===")
    mots_caption = corpus["caption"].str.split().str.len()
    avec = corpus[corpus["a_dialogue"] == 1]
    mots_dialogue = avec["dialogue"].str.split().str.len()
    print(f"description : {mots_caption.mean():.1f} mots (médiane {mots_caption.median():.0f})")
    print(f"dialogue    : {mots_dialogue.mean():.1f} mots (médiane {mots_dialogue.median():.0f}), "
          f"sur {len(avec)} extraits")

    print("\n=== Exemples ===")
    for _, ligne in corpus[corpus["a_dialogue"] == 1].sample(3, random_state=config.GRAINE).iterrows():
        print(f"[{ligne['emotion']}] {ligne['video_id']}")
        print(f"  description : {ligne['caption'][:150]}")
        print(f"  dialogue    : {ligne['dialogue'][:150]}")


if __name__ == "__main__":
    main()