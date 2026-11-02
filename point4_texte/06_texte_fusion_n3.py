"""Fusion N3 (système final) : mêmes candidats qu'en N2, plus les modèles fine-tunés par 05."""

import utils

MODELES = ["tfidf", "emb_mpnet", "emb_e5", "emb_bge", "emb_goemotions", "emb_ekman",
           "ft_mpnet", "ft_e5", "ft_bge", "ft_goemotions", "ft_ekman", "ft_roberta"]


if __name__ == "__main__":
    utils.fusionner("n3", MODELES, sortie="texte")
