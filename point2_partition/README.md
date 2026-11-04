# Point 2 — Analyse du corpus et partition

Le corpus compte **350 films et 3 088 extraits**, avec des émotions déséquilibrées : `anger` représente
21,9 % des extraits, `amusement` 7,9 %.

Les extraits d'un même film se ressemblent (mêmes acteurs, même lumière, même musique) : la partition se
fait donc **par film**, pour qu'aucun film ne soit à la fois dans l'apprentissage et dans l'évaluation.

## Partition retenue

Chaque groupe a proposé une partition ; une seule a été retenue et est utilisée par tous les groupes.
C'est celle de ce dossier : [train.txt](train.txt), [dev.txt](dev.txt), [test.txt](test.txt)
(un identifiant de film par ligne). Elle n'est pas celle de notre groupe, qui visait 12,5 % de chaque émotion
en dev et en test.

[statistiques.py](statistiques.py) donne la répartition obtenue :

| | train | dev | test | total |
|---|---:|---:|---:|---:|
| Films | 250 | 50 | 50 | 350 |
| Extraits | 2 207 | 442 | 439 | 3 088 |
| amusement | 7,9 % | 7,9 % | 8,0 % | 7,9 % |
| anger | 21,9 % | 22,2 % | 21,9 % | 21,9 % |
| awe | 9,3 % | 9,0 % | 9,1 % | 9,3 % |
| contentment | 12,0 % | 12,2 % | 12,3 % | 12,1 % |
| disgust | 9,6 % | 9,7 % | 9,8 % | 9,6 % |
| excitement | 11,4 % | 11,5 % | 11,6 % | 11,5 % |
| fear | 16,8 % | 16,7 % | 16,9 % | 16,8 % |
| sadness | 11,1 % | 10,6 % | 10,5 % | 10,9 % |

Les trois ensembles gardent les proportions du corpus, à moins d'un point près, et aucun film n'est dans
deux ensembles. Le déséquilibre des classes est donc traité dans les modèles (poids des classes, F1 macro),
pas dans la partition.

## Utilisation dans la suite

[point3_transcription/09_construire_corpus.py](../point3_transcription/09_construire_corpus.py) lit ces
listes et écrit la colonne `partition` de `corpus.csv`, que tous les modèles utilisent :

- **train** : apprentissage, et tous les choix par validation croisée à 5 plis groupés par film ;
- **dev** : mesure des modèles retenus, jamais pour choisir ;
- **test** : utilisé une seule fois, pour le système final (point 5).

```bash
python statistiques.py     # besoin de movie/annotation/ (variable EMOVID_DIR)
```
