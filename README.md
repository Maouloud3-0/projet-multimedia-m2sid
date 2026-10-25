# Projet données multimédia — M2 SID

Reconnaissance de la catégorie d'émotion (8 classes) dans des extraits de film
du corpus **EmoVid** : 350 films, 3 088 extraits.

Ce dépôt regroupe le travail des trois parties à coder du projet.

## Organisation

| Dossier | Contenu |
|---|---|
| [point2_analyse_partition/](point2_analyse_partition/) | Analyse du corpus et proposition de partition 250 / 50 / 50 films (tâche 2) |
| [point3_transcription/](point3_transcription/) | **Traitement de l'audio : transcription des extraits en texte** |
| [point4_modelisation/](point4_modelisation/) | Modélisation séparée du texte, de l'audio et de la vidéo |
| [point5_fusion/](point5_fusion/) | Fusion des trois modalités |
| [splits/](splits/) | Listes de films des ensembles d'apprentissage, de développement et de test (communes à tous les groupes) |

Un seul dépôt avec un dossier par partie : l'historique reste lisible, les
listes communes (splits/) sont partagées, et chaque étudiant peut travailler
sur sa propre branche.

## Les données

Le corpus (579 Mo d'audio, ~20 Go de vidéo) n'est pas versionné.
Indiquer son emplacement dans [point3_transcription/config.py](point3_transcription/config.py)
ou avec la variable d'environnement `EMOVID_DIR`.

Structure attendue :

```
movie/
  audio/<film_id>/<video_id>.mp3
  annotation/<film_id>/<video_id>.json
  video/<film_id>/<video_id>.mp4
```

## Partition du corpus

Définie une fois pour tous les groupes, par film (aucun film n'est à cheval
sur deux ensembles) :

- **apprentissage** : 250 films — [splits/train.txt](splits/train.txt)
- **développement** : 50 films — [splits/dev.txt](splits/dev.txt)
- **test** : 50 films — [splits/test.txt](splits/test.txt)
