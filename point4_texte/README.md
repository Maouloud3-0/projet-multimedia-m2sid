# Texte : reconnaître l'émotion à partir de la description et des dialogues

## En bref

```
 description ──┐
               ├──> ekman fine-tuné ──┐
 dialogue ─────┤                      │
               ├──> e5 (gelé) ────────┼──> moyenne des probabilités ──> 8 émotions
               │                      │
               └──> ekman (gelé) ─────┘
```

| Système | F1 macro dev | Exactitude dev |
|---|---|---|
| Hasard | 0,125 | 12,5 % |
| TF-IDF + régression logistique | 0,295 | 29,9 % |
| Meilleur modèle seul (ekman fine-tuné) | 0,370 | 37,8 % |
| **Système final (moyenne de 3 modèles)** | **0,371** | **37,8 %** |

- Tous les choix (réglages, entrée, modèles fusionnés) sont faits par validation
  croisée sur train ; dev ne sert qu'à mesurer, test n'est pas utilisé.
- Les probabilités du système final sont dans `resultats/probas_texte.npz`.

## Les données

Entrée : [point3_transcription_verifiee/resultats/corpus.csv](../point3_transcription_verifiee/resultats/corpus.csv),
une ligne par extrait (`caption`, `dialogue`, `a_dialogue`, `emotion`, `partition`).

| | train | dev | test |
|---|---|---|---|
| Films | 250 | 50 | 50 |
| Extraits | 2 207 | 442 | 439 |
| Avec dialogue anglais | 1 450 (66 %) | 298 (67 %) | 277 (63 %) |

La description fait 53 mots en moyenne, le dialogue 16. Un tiers des extraits
n'a pas de dialogue, et seulement 41 % des extraits `fear`.

## Comment on mesure

- Partition par film commune à tous les groupes (250 / 50 / 50).
- Validation croisée à 5 plis sur train, groupée par film : un film n'est jamais
  à la fois dans l'apprentissage et l'évaluation d'un pli. Elle sert à choisir C,
  le poids des classes, l'entrée de chaque modèle et le nombre de modèles fusionnés.
- Dev (50 films) est trop petit pour choisir : en choisissant sur dev, le même
  système affichait 0,405 sur dev, contre 0,366 une fois réglé en CV.
- F1 macro comme métrique principale, classes pondérées.

## Les modèles

| Niveau | Script | Ce qui est entraîné |
|---|---|---|
| N1 | [02_texte_tfidf.py](02_texte_tfidf.py) | TF-IDF (mots et bigrammes) + régression logistique ou SVM linéaire |
| N2 | [03_texte_embeddings.py](03_texte_embeddings.py) | régression logistique sur les vecteurs de 5 modèles gelés : mpnet, e5-large, bge-large, goemotions, ekman |
| N3 | [05_texte_finetuning.py](05_texte_finetuning.py) | tout le modèle ekman (RoBERTa-large déjà entraîné sur des émotions), nouvelle couche de sortie à 8 classes |
| Fusion | [04_texte_fusion.py](04_texte_fusion.py) | rien : moyenne des probabilités des k meilleurs modèles en CV |

Pour N2, quatre façons de donner le texte au modèle :
`description`, `dialogue`, `description+dialogue` (encodés séparément, vecteur
du dialogue à zéro s'il n'y en a pas, plus l'indicateur `a_dialogue`) et
`concatenation` (les deux textes bout à bout, un seul vecteur).

Pour N3, les réglages sont fixés à des valeurs courantes pour RoBERTa
(4 époques, lr 2e-5, lots de 16, 256 tokens) : un entraînement prend plusieurs
minutes, une recherche sur grille n'était pas envisageable.

## Résultats détaillés

| Niveau | Configuration | F1 cv | F1 dev | Exactitude dev |
|---|---|---|---|---|
| N1 | TF-IDF, description + dialogue | 0,295 | 0,295 | 0,299 |
| N2 | mpnet, description + dialogue | 0,320 | 0,358 | 0,355 |
| N2 | e5, concaténation | 0,330 | 0,338 | 0,335 |
| N2 | ekman, description + dialogue | 0,321 | 0,355 | 0,357 |
| N3 | ekman fine-tuné | 0,335 | 0,370 | 0,378 |
| Fusion | ekman fine-tuné + e5 + ekman | 0,351 | 0,371 | 0,378 |

Tous les scores sont dans `resultats/scores_*.csv`.

## Ce qu'on retient

- Les modèles pré-entraînés dépassent le TF-IDF de 2 à 3,5 points en CV : ils
  connaissent les synonymes et le contexte, le TF-IDF doit tout apprendre sur 2 200 extraits.
- Le dialogue seul est faible (F1 ≈ 0,17 en CV). Ajouté à la description, il
  aide un peu en CV (jusqu'à +0,6 point) et davantage sur dev (mpnet : 0,312 → 0,358).
- Encoder description et dialogue séparément est un peu meilleur que les
  concaténer pour 4 modèles sur 5 (jusqu'à +1,2 point en CV).
- Le fine-tuning d'ekman gagne 1,3 point en CV et 1,5 point sur dev par rapport à ekman gelé.
- La fusion gagne 1,6 point en CV sur le meilleur modèle seul ; sur dev, le gain
  n'est pas significatif (IC 95 % par films : [−0,023 ; +0,026]).
- `awe`, `excitement` et `contentment` restent les plus confondues : la
  description dit ce qui se passe, rarement ce qu'on ressent.

## Lancer les scripts

```bash
pip install -r requirements.txt
python 01_explorer_corpus.py      # statistiques du corpus
python 02_texte_tfidf.py          # ~1 min
python 03_texte_embeddings.py     # ~3 min avec carte graphique
python 05_texte_finetuning.py     # ~30 min avec carte graphique
python 04_texte_fusion.py         # en dernier
```

`requirements.txt` installe PyTorch pour CPU ; la carte graphique est utilisée
automatiquement si PyTorch en voit une (CUDA ou ROCm). Les modèles sont
téléchargés depuis Hugging Face au premier lancement, les vecteurs sont gardés
dans `data/` (non versionné).

## Fichiers produits

Dans `resultats/` :

- `scores_tfidf.csv`, `scores_embeddings.csv`, `scores_fusion.csv` ;
- `confusion_<modèle>.png` : matrices de confusion sur dev ;
- `probas_<modèle>.npz`, dont `probas_texte.npz` pour le système final.
  Champs : `video_id`, `partition`, `probabilites` (3 088 × 8), `probabilites_hors_pli`
  (CV sur train, NaN ailleurs) et `emotions` : amusement, anger, awe,
  contentment, disgust, excitement, fear, sadness.