"""Fusion audio : moyenne des probabilités des k meilleurs modèles audio, comme pour le texte et la vidéo.

Candidats : N1 (01), les modèles gelés + classifieur (03), les probabilités AudioSet seules (03),
            les couches d'attention (04).
Classement des modèles et choix de k sur les probabilités hors pli de train (CV) ; dev ne sert qu'à mesurer.
Sorties : resultats/scores_fusion_audio.csv, resultats/probas_audio.npz (utilisé par la fusion des trois modalités).
"""

import config
import utils

CANDIDATS = ["n1",
             "emb_whisper", "emb_wavlm", "emb_w2v2_emo", "emb_ast", "emb_clap", "audioset",
             "att_whisper", "att_wavlm", "att_w2v2_emo", "att_ast"]


if __name__ == "__main__":
    # un modèle dont l'étape n'a pas été lancée est simplement ignoré
    utils.fusionner("audio", [m for m in CANDIDATS if (config.RESULTATS / f"probas_{m}.npz").exists()],
                    sortie="audio")
