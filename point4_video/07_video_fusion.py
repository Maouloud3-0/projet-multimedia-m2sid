"""Fusion vidéo : moyenne des probabilités des k meilleurs modèles vidéo, comme pour le texte.

Candidats : N1 (04), les modèles gelés + classifieur (05), les visages seuls (05),
            les couches d'attention (06).
Classement des modèles et choix de k sur les probabilités hors pli de train (CV) ; dev ne sert qu'à mesurer.
Sorties : resultats/scores_fusion_video.csv, resultats/probas_video.npz (utilisé par la fusion des trois modalités).
"""

import utils

MODELES = ["n1",
           "emb_pe", "emb_siglip2", "emb_clip_b", "emb_clip_l", "emb_clip_h", "emb_dinov2", "emb_videomae",
           "visages",
           "att_pe", "att_siglip2", "att_clip_l", "att_clip_h", "att_clip_b", "att_dinov2"]


if __name__ == "__main__":
    utils.fusionner("video", MODELES, sortie="video")
