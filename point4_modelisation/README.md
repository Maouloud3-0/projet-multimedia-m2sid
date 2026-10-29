# Point 4 — Modélisation audio et vidéo

Objectif : prédire l'une des huit émotions à partir de chaque modalité,
séparément. Le texte est traité dans un autre module.

## Protocole

- **Partition par film** : 250 films pour l'apprentissage, 50 pour le
  développement, 50 pour le test.
- **Aucun réglage sur le test** : les modèles et configurations sont comparés
  sur le développement.
- **F1 macro** comme métrique principale, car les classes sont déséquilibrées.
- Pondération des classes dans les classifieurs.
- Les probabilités des huit classes sont enregistrées pour la fusion du point 5.

## Niveau 1 — Baselines

| Script | Caractéristiques | Classifieurs |
|---|---|---|
| [01_audio_n1.py](01_audio_n1.py) | MFCC, chroma, centroïde, bande passante, rolloff, contraste spectral, énergie, ZCR, tempo | régression logistique, forêt aléatoire |
| [02_video_n1.py](02_video_n1.py) | brightness, colorfulness, hue + moyennes, écarts-types et histogrammes de couleurs sur 4 images | régression logistique, forêt aléatoire |

Les caractéristiques sont mises en cache dans `data/` pour ne pas les recalculer
à chaque entraînement.

## Niveau 2 — Embeddings pré-entraînés

- **Audio** : AST ou CLAP pour le son général, WavLM/emotion2vec pour la parole.
  Comparaison du mixage brut et de la piste voix Demucs.
- **Vidéo** : CLIP ou SigLIP sur 8 images, moyenne des embeddings.

Les gros modèles servent d'extracteurs ; seul un petit classifieur est entraîné
sur nos annotations.

## Commandes

Utiliser le même environnement Python que le point 3 :

```bash
pip install -r requirements.txt
python 01_audio_n1.py
python 02_video_n1.py
```

## Résultats sur le développement

| Modalité | Configuration | Meilleur classifieur | F1 macro | Exactitude |
|---|---|---|---|---|
| Audio | N1 descripteurs classiques | SVM | 0,243 | 0,244 |
| Audio | AST AudioSet | SVM | 0,248 | 0,256 |
| Audio | WavLM brut | régression logistique | 0,218 | 0,224 |
| Audio | WavLM sur voix Demucs | SVM | 0,206 | 0,208 |
| Audio | wav2vec2 SUPERB ER | SVM | 0,204 | 0,217 |
| **Audio** | **AST + N1** | **SVM** | **0,269** | **0,278** |
| Vidéo | N1 couleurs | régression logistique | 0,181 | 0,176 |
| Vidéo | CLIP, 8 images | SVM | 0,394 | 0,391 |
| **Vidéo** | **CLIP + N1** | **SVM** | **0,398** | **0,398** |

### Conclusions

- Les **SVM à noyau RBF** sont nettement meilleurs que la régression logistique
  et les forêts sur les embeddings.
- La combinaison des descripteurs N1 et des embeddings améliore légèrement les
  deux modalités.
- **Demucs ne fait pas progresser la reconnaissance d'émotion audio** avec
  WavLM : 0,206 contre 0,218 sur le mixage brut. La musique et les effets
  sonores contiennent de l'information utile.
- AST est plus utile que WavLM et wav2vec2 SUPERB ER sur ce corpus, surtout
  lorsqu'il est complété par les descripteurs classiques.
- CLIP apporte le gain le plus important : la vidéo est actuellement la
  modalité la plus performante.
- Le modèle audio ne reconnaît aucun extrait `disgust` sur dev ; la vidéo est
  plus équilibrée. Ce point devra guider la fusion.
- **N3 non retenu pour l'instant** : le fine-tuning n'est pas nécessaire pour
  obtenir ces résultats et le rapport coût/bénéfice est défavorable.

Le test n'a pas été utilisé pour choisir les modèles. Les scores sont dans
`resultats/synthese.csv`, les rappels par émotion dans `resultats/rappels_*.csv`
et les matrices de confusion en PNG. Les probabilités sont dans
`data/probas_*.npz`, dans l'ordre fixe : amusement, anger, awe, contentment,
disgust, excitement, fear, sadness.
