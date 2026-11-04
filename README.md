# Reconnaissance d'émotions dans des extraits de film — M2 SID, groupe 4

Projet Données multimédia 2026 : reconnaître la catégorie d'émotion (8 classes : amusement, anger, awe,
contentment, disgust, excitement, fear, sadness) d'extraits de film du corpus
[EmoVid](https://zane-zyqiu.github.io/EmoVid/) (section *movie* : 350 films, 3 088 extraits), à partir du
texte, de l'audio et de la vidéo, puis en fusionnant les trois.

## Résultats

F1 macro (8 classes, hasard : 0,125). **cv** : validation croisée sur train ; **dev** : mesure ;
**test** : une seule fois, à la fin.

| Système | F1 cv | F1 dev | F1 test |
|---|---:|---:|---:|
| Audio — Whisper et WavLM gelés, régression logistique et attention | 0,271 | 0,270 | 0,297 |
| Texte — description + dialogue, 4 modèles affinés et 2 gelés | 0,362 | 0,387 | 0,365 |
| Vidéo — PE, SigLIP 2, CLIP gelés + visages, régression logistique et attention | 0,450 | 0,461 | 0,413 |
| **Fusion — moyenne pondérée des trois (vidéo 0,6, texte 0,2, audio 0,2)** | **0,455** | **0,486** | **0,419** |

La vidéo domine : selon l'article EmoVid, 80 % des émotions ont été annotées par un modèle qui ne voit que les
images. La fusion gagne 2,5 points sur dev (significatif) et 0,6 point sur test (non significatif).

## Organisation

| Dossier | Point de l'énoncé | Contenu |
|---|---|---|
| [point2_partition/](point2_partition/) | 2. Analyse et partition | Listes de films train / dev / test communes à tous les groupes, statistiques |
| [point3_transcription/](point3_transcription/) | 3. Transcription | Whisper large-v3 + Qwen3-ASR, vérification à l'écoute des désaccords → `corpus.csv` |
| [point4_texte/](point4_texte/) | 4. Modèle texte | TF-IDF, encodeurs gelés, fine-tuning, combinaison |
| [point4_audio/](point4_audio/) | 4. Modèle audio | eGeMAPS, encodeurs gelés (Whisper, WavLM…), attention, combinaison |
| [point4_video/](point4_video/) | 4. Modèle vidéo | Images, encodeurs gelés, visages, attention, combinaison |
| [point5_fusion/](point5_fusion/) | 5. Fusion | Fusion tardive, fusion précoce, modèle multimodal à attention, système final |

Chaque dossier a son README : ce qui a été essayé, les scores, comment relancer.

## Protocole commun

- **Partition par film** (250 / 50 / 50) : aucun film n'est dans deux ensembles.
- **Tous les choix sur train**, par validation croisée à 5 plis groupés par film (réglages, couches,
  combinaisons de modèles, poids de la fusion). **Dev ne sert qu'à mesurer**, test une seule fois au point 5.
- **F1 macro**, classes pondérées pendant l'apprentissage (les classes vont de 7,9 % à 21,9 %).
- Chaque modèle enregistre ses probabilités dans `resultats/probas_<modèle>.npz` : `video_id`, `partition`,
  `probabilites` (3 088 × 8), `probabilites_hors_pli` (train) et `emotions`. Les probabilités **hors pli**
  (chaque extrait de train prédit par un modèle qui ne l'a pas vu) servent à apprendre les combinaisons et la
  fusion sans toucher à dev.

## Données

Le corpus (vidéos, audios, annotations) n'est pas versionné. Par défaut, les scripts le cherchent dans
l'archive du cours décompressée **à côté du dépôt** :

```
PROJET_SID_2026/PROJET_SID_2026/movie/
  annotation/<film_id>/<video_id>.json
  audio/<film_id>/<video_id>.mp3
  video/<film_id>/<video_id>.mp4
projet-multimedia-gitlab/        (ce dépôt)
```

Sinon : variable `EMOVID_DIR` (chemin du dossier `movie/`) pour les points 2 et 4, `PROJET_SID_DIR`
(dossier qui contient `movie/`) pour le point 3.

Les résultats de chaque point (`resultats/`) sont versionnés : on peut lire les scores, ou relancer un point,
sans refaire les précédents. Les fichiers intermédiaires lourds (images, vecteurs) vont dans `data/`, non
versionné.

## Relancer

Python 3.11, une carte graphique pour les encodeurs (8 Go suffisent). Les points 3 à 5 ont chacun leur `requirements.txt` ; le point 2 n'utilise que la bibliothèque standard de Python.
Ordre : point 3 (ou directement son `corpus.csv` versionné), puis les trois modules du point 4 dans
n'importe quel ordre (la vidéo d'abord si l'on relance aussi l'audio, qui réutilise `point4_video/resultats/fin_pub.csv`),
puis le point 5. Les commandes sont dans le README de chaque dossier.

## Répartition du travail

| Partie | Auteur |
|---|---|
| Points 3, 4 texte et 4 vidéo | Hanchen LI |
| Points 4 audio et 5 | Gaoussou |
