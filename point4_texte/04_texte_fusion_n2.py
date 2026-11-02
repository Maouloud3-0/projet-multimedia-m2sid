"""Fusion N2 : moyenne des probabilités des k meilleurs modèles parmi le TF-IDF et les modèles gelés."""

import utils

MODELES = ["tfidf", "emb_mpnet", "emb_e5", "emb_bge", "emb_goemotions", "emb_ekman"]


if __name__ == "__main__":
    utils.fusionner("n2", MODELES, sortie="texte_n2")
