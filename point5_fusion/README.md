# Point 5 — Fusion des trois modalités

## En bref

```
 vidéo (point4_video) ──> 8 probabilités ──┐  × 3
 texte (point4_texte) ──> 8 probabilités ──┼─ × 1 ──> moyenne pondérée ──> 8 émotions
 audio (point4_audio) ──> 8 probabilités ──┘  × 1
```

| Système | F1 cv | F1 dev | F1 test |
|---|---|---|---|
| Audio seul | 0,271 | 0,270 | 0,297 |
| Texte seul | 0,362 | 0,387 | 0,365 |
| Vidéo seule | 0,450 | 0,461 | 0,413 |
| **Fusion : moyenne pondérée (vidéo 3, texte 1, audio 1)** | **0,455** | **0,486** | **0,419** |

- Trois façons de fusionner ont été comparées avec le même protocole : **A** combiner les probabilités,
  **B** mettre les vecteurs bout à bout, **C** un petit modèle multimodal à attention.
- La plus simple gagne : une moyenne pondérée des probabilités, poids choisis en CV sur train.
- Gain sur la vidéo seule : significatif sur dev (+2,5 points, IC 95 % par films [+0,001 ; +0,053]),
  pas sur test (+0,6 point, IC [−0,017 ; +0,030]).
- Les probabilités finales sont dans `resultats/probas_finale.npz`.

## Les entrées

Les probabilités des systèmes finaux du point 4 (`probas_video.npz`, `probas_texte.npz`, `probas_audio.npz`)
et, pour B et C, les vecteurs gelés du meilleur encodeur de chaque modalité, avec son réglage retenu au point 4 :

| Modalité | Encodeur | Vecteur | Seul au point 4 (F1 cv) |
|---|---|---|---|
| Vidéo | PE, 16 images + visages ViT et HSEmotion | `point4_video/data/emb_pe.npy`, `visages.npz` | 0,430 |
| Texte | e5-large, description + dialogue | calculé ici (`data/e5_concatenation.npy`) | 0,330 |
| Audio | Whisper large-v3, couche 16 | `point4_audio/data/couches_whisper.npy`, `seg_whisper.npy` | 0,265 |

Les vecteurs vidéo ont été recalculés avec les scripts `01` à `03` de [point4_video](../point4_video/), sans
modification : PE moyenné et régression logistique redonnent exactement le F1 du module vidéo (0,4241 en CV,
0,4110 sur dev).

## Comment on mesure

- Même protocole que le point 4 : partition par film 250 / 50 / 50, validation croisée à 5 plis groupés par
  film sur train pour tous les choix, dev ne sert qu'à mesurer.
- La fusion apprend sur les **probabilités hors pli** de train : chaque extrait y est prédit par un modèle qui
  ne l'a pas vu, comme le seront dev et test. C'est pour cela que chaque modalité les fournit.
- Système retenu : le meilleur F1 hors pli sur train, parmi toutes les variantes de A, B et C.
- **Test n'est utilisé qu'une fois**, dans [04_comparaison.py](04_comparaison.py), pour le système retenu et les
  trois modalités seules. Aucun choix n'est fait sur test.
- F1 cv : F1 macro des prédictions hors pli regroupées. Les réglages (poids, C, essais Optuna) sont choisis sur
  ces mêmes prédictions, d'où un léger optimisme ; dev et test servent de contrôle.

## Option A — fusion tardive : combiner les probabilités

[01_fusion_tardive.py](01_fusion_tardive.py). Chaque modalité a déjà voté ; on combine les votes :

- **moyenne** : les trois modalités comptent autant ;
- **pondérée** : un poids par modalité, de 0 à 1 par pas de 0,1, choisi en CV ;
- **stacking** : régression logistique sur les 24 log-probabilités (3 × 8), C choisi en CV.

| Méthode (systèmes finaux du point 4) | Poids ou réglage | F1 cv | F1 dev |
|---|---|---|---|
| Vidéo seule | — | 0,450 | 0,461 |
| Moyenne simple | 1 / 1 / 1 | 0,435 | 0,451 |
| **Moyenne pondérée** | **vidéo 0,3, texte 0,1, audio 0,1** | **0,455** | **0,486** |
| Stacking | C = 0,003, classes équilibrées | 0,412 | 0,452 |

Apport de chaque modalité (moyenne pondérée) :

| Modalités | Poids | F1 cv | F1 dev |
|---|---|---|---|
| Vidéo + texte | 0,7 / 0,1 | 0,451 | 0,474 |
| Vidéo + audio | 0,1 / 0 | 0,450 | 0,461 |
| Texte + audio | 0,2 / 0,1 | 0,371 | 0,390 |
| Vidéo + texte + audio | 0,3 / 0,1 / 0,1 | 0,455 | 0,486 |

- La moyenne simple fait **moins bien** que la vidéo seule : elle donne autant de poids à l'audio (0,271) et
  au texte (0,362) qu'à la vidéo (0,450).
- Le stacking a trop de paramètres pour 2 207 exemples : il fait moins bien que la moyenne simple en CV.
- Seul, l'audio n'apporte rien à la vidéo (poids 0) ; avec le texte, il gagne un poids égal à celui du texte.

## Option B — fusion précoce : mettre les vecteurs bout à bout

[02_fusion_precoce.py](02_fusion_precoce.py). Les vecteurs des trois modalités sont concaténés (5 413
valeurs), puis une seule régression logistique (C et poids des classes choisis en CV). Le classifieur peut en
principe apprendre des combinaisons (image sombre + voix calme ≠ image sombre + cris). Deux variantes : **brut**
(chaque colonne compte autant, donc la vidéo, 3 108 colonnes, pèse le plus) et **équilibré** (chaque modalité
reçoit le même poids total).

| Variante | Modalités | F1 cv | F1 dev |
|---|---|---|---|
| Brut | vidéo seule (référence) | 0,431 | 0,441 |
| Brut | vidéo + texte + audio | 0,428 | 0,438 |
| Équilibré | vidéo + texte + audio | 0,415 | 0,438 |
| Brut | vidéo + texte | 0,428 | 0,469 |
| Brut | vidéo + audio | 0,423 | 0,432 |

Mettre les vecteurs bout à bout **ne bat pas la vidéo seule** en CV : avec 2 207 exemples, les colonnes du
texte et de l'audio ajoutent surtout du bruit, et le classifieur ne peut pas apprendre à leur donner moins de
poids sans perdre aussi de l'information vidéo (une seule régularisation pour toutes les colonnes).
Équilibrer les modalités aggrave le problème.

## Option C — petit modèle multimodal à attention

[03_fusion_attention.py](03_fusion_attention.py). La tête d'attention du N3 vidéo, étendue aux trois
modalités ; les encodeurs restent gelés. Chaque extrait devient une suite de 33 morceaux : 16 images
(PE + plus grand visage), 1 texte (e5), 16 tranches de son (Whisper). Chaque modalité est projetée dans un
même espace avec un vecteur appris qui l'identifie ; une couche de self-attention est possible (les morceaux
se regardent entre eux) ; puis un score par morceau donne une moyenne pondérée, et une couche de sortie
prédit les 8 émotions. Pendant l'apprentissage, des morceaux et parfois une modalité entière sont retirés au
hasard, pour que le modèle ne s'appuie pas que sur la vidéo. 30 essais Optuna en CV, puis 3 graines.

| | F1 cv | F1 dev | Part d'attention sur dev (vidéo / texte / audio) |
|---|---|---|---|
| Attention multimodale | 0,439 | 0,450 | 95 % / 3 % / 2 % |

Réglage retenu : dimension 256, **aucune couche de self-attention**, 16 époques. Le modèle dépasse la vidéo
seule avec les mêmes entrées (0,431) et la fusion précoce (0,428), mais il regarde presque uniquement les
images. C'est cohérent avec le corpus : selon l'article EmoVid, 80 % des émotions ont été annotées par un
modèle qui ne voit que les images (NVILA-Lite-2B).

## Comparaison et système final

[04_comparaison.py](04_comparaison.py). Pour comparer à armes égales, A a aussi été appliqué aux mêmes
entrées que B et C (un encodeur gelé par modalité, ligne « mêmes entrées »).

| Option | Système | F1 cv | F1 dev |
|---|---|---|---|
| A | moyenne pondérée, systèmes finaux du point 4 | **0,455** | **0,486** |
| A | moyenne pondérée, mêmes entrées que B et C | 0,448 | 0,454 |
| C | attention multimodale | 0,439 | 0,450 |
| B | concaténation, brut | 0,428 | 0,438 |

Même à entrées égales, la fusion tardive reste devant : avec 2 207 exemples, apprendre trois poids est plus
sûr qu'apprendre des milliers de paramètres sur des vecteurs de plusieurs milliers de valeurs. Elle profite
en plus des améliorations faites dans chaque modalité au point 4 (fine-tuning du texte, combinaisons de
modèles), ce que B et C ne peuvent pas faire.

**Système retenu : A, moyenne pondérée** (meilleur F1 cv).

| | F1 macro | Exactitude | Gain sur la vidéo seule (IC 95 % par films) |
|---|---|---|---|
| dev | 0,486 | 48,2 % | +0,025 [+0,001 ; +0,053] |
| test | 0,419 | 44,4 % | +0,006 [−0,017 ; +0,030] |

Les 50 films de test sont plus difficiles que ceux de dev pour toutes les modalités (vidéo : 0,461 sur dev,
0,413 sur test). Sur test, la fusion reste devant chaque modalité seule, mais son avance sur la vidéo n'est
plus significative.

## Ce qu'on retient

- La vidéo porte l'essentiel de l'information, en partie à cause de l'annotation du corpus (faite sur les
  images).
- Texte et audio apportent un petit gain, à condition de leur donner peu de poids : une moyenne simple dégrade.
- Avec 2 207 extraits d'apprentissage, la fusion la plus simple est la meilleure ; les méthodes qui apprennent
  plus de paramètres (stacking, concaténation, attention) n'en profitent pas.
- Un grand modèle multimodal pré-entraîné (de type Qwen2.5-Omni) n'a pas été essayé : il ne tient pas sur une
  carte de 8 Go, et l'affiner sur 2 207 exemples poserait le même problème.

## Lancer les scripts

Après les modules du point 4 (leurs `resultats/` sont dans Git ; B et C ont aussi besoin de leurs `data/` :
`point4_video` 01 à 03 pour PE et les visages, `point4_audio` 02 pour Whisper). Depuis `point5_fusion/` :

```bash
pip install -r requirements.txt
python 01_fusion_tardive.py      # A, quelques secondes
python 02_fusion_precoce.py      # B, ~10 min (processeur) ; encode d'abord le texte avec e5
python 03_fusion_attention.py    # C, ~30 min sur GPU
python 04_comparaison.py         # comparaison, système final, test
```

## Fichiers produits

Dans `data/` (exclu de Git) : `e5_concatenation.npy`, journaux.

Dans `resultats/` :

- `scores_tardive.csv`, `scores_precoce.csv`, `scores_attention.csv`, `essais_attention.csv`,
  `scores_comparaison.csv` ;
- `confusion_<système>.png` : matrices de confusion sur dev, et `confusion_finale_test.png` sur test ;
- `probas_<système>.npz`, dont **`probas_finale.npz`** : `video_id`, `partition`, `probabilites` (3 088 × 8),
  `probabilites_hors_pli` (train, NaN ailleurs), `emotions`.
