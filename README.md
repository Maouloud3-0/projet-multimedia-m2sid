# Reconnaissance d'émotions dans des extraits de film — M2 SID, groupe 4

**Groupe 4 — Gaoussou Maouloud KONTA, Hanchen LI**

Dépôt : https://gitlab.com/Maouloud3-0/projet-multimedia-m2sid

Projet Données multimédia 2026 : reconnaître la catégorie d'émotion (8 classes : amusement, anger, awe,
contentment, disgust, excitement, fear, sadness) d'extraits de film du corpus
[EmoVid](https://zane-zyqiu.github.io/EmoVid/) (section *movie* : 350 films, 3 088 extraits), à partir du
texte, de l'audio et de la vidéo, puis en fusionnant les trois.

Le **rapport complet** est dans [RAPPORT.pdf](RAPPORT.pdf) (54 pages, 38 figures : schémas d'architecture,
matrices de confusion, analyses) ; sa source lisible en ligne est [RAPPORT.md](RAPPORT.md). Chaque dossier a
aussi son README détaillé.

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
| [rapport/](rapport/) | rapport | Figures, schémas et génération du PDF à partir des résultats versionnés |

Chaque dossier a son README : ce qui a été essayé, les scores, comment relancer.

## Méthodes en bref

Chaque modalité va du plus simple au plus riche, avec trois niveaux :

| | N1 : descripteurs classiques | N2 : encodeurs pré-entraînés gelés + classifieur | N3 : partie apprise en plus | Système final |
|---|---|---|---|---|
| **Texte** | TF-IDF — 0,295 | mpnet, e5, bge, modèles d'émotion — 0,330 (e5) | fine-tuning de 6 modèles — 0,341 (ekman) | moyenne de 6 modèles — **0,362** |
| **Audio** | eGeMAPS + MFCC — 0,212 | Whisper, WavLM, wav2vec2, AST, CLAP — 0,265 (Whisper, couche 16) | attention sur 16 tranches de temps — 0,258 | moyenne de 4 modèles — **0,271** |
| **Vidéo** | couleurs, luminosité — 0,192 | PE, SigLIP 2, CLIP, DINOv2… + visages — 0,430 (PE) | attention sur 16 images — 0,421 | moyenne de 6 modèles — **0,450** |

(F1 cv du meilleur modèle de chaque niveau.)

- **Encodeur gelé** : un modèle pré-entraîné sur de grandes quantités de données transforme l'extrait en
  vecteur ; on ne modifie pas ses poids, il ne voit aucune étiquette, et il ne tourne qu'une fois sur les
  3 088 extraits. Seul le petit classifieur posé dessus est appris.
- **Système final d'une modalité** : moyenne des probabilités des *k* meilleurs modèles, *k* choisi en
  validation croisée.

Pour la **fusion** (point 5), trois approches ont été comparées :

| Approche | Principe | F1 cv | F1 dev |
|---|---|---:|---:|
| **A, fusion tardive** | moyenne pondérée des probabilités des trois modalités | **0,455** | **0,486** |
| C, modèle multimodal | tête d'attention sur 16 images, 1 texte et 16 tranches de son | 0,439 | 0,450 |
| B, fusion précoce | vecteurs des trois modalités mis bout à bout, une régression logistique | 0,428 | 0,438 |

## Comment lire les scores

- **F1 macro** : pour chaque émotion, le F1 combine la précision (quand le modèle dit « peur », a-t-il
  raison ?) et le rappel (retrouve-t-il les vraies peurs ?). Le F1 macro est la moyenne sur les 8 émotions,
  chacune comptant pareil malgré le déséquilibre des classes. Hasard : 0,125.
- **F1 cv** : les 250 films de train sont répartis en 5 paquets ; on apprend sur 4 paquets et on prédit le
  5ᵉ, cinq fois. Chaque extrait de train est ainsi prédit par un modèle qui n'a jamais vu son film
  (prédiction **hors pli**). Le F1 cv est calculé sur ces 2 207 prédictions. Il sert à faire tous les choix.
- **F1 dev** : le modèle final, appris sur tout train, prédit les 50 films de dev. Ce score ne sert qu'à
  vérifier que le F1 cv était fiable (les deux sont proches dans tous nos modules).
- **F1 test** : calculé une seule fois, à la fin, pour le système retenu.

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

Le rapport se régénère depuis `rapport/` : `dessiner_figures.py` (figures et schémas), `construire_rapport.py`
(mise en page du PDF à partir de `RAPPORT.md`) et `verifier_resultats.py` (contrôle des scores) ; voir
[rapport/README.md](rapport/README.md).

## Analyse des résultats

- **La vidéo domine** (0,461 sur dev), devant le texte (0,387) et l'audio (0,270). D'après l'article EmoVid,
  80 % des émotions ont été annotées par un modèle vision-langage qui ne voit que les images : les étiquettes
  reflètent surtout l'image.
- **Chaque modalité a ses émotions** : le texte reconnaît le mieux le dégoût (F1 dev 0,43 contre 0,27 pour la
  vidéo), l'audio a sa meilleure émotion avec la tristesse (0,40).
- **Les encodeurs gelés font l'essentiel du travail.** Pour l'audio, la meilleure couche de Whisper est celle
  du milieu (16 sur 32) : les dernières couches se spécialisent dans la transcription et perdent le ton de la
  voix.
- **Pour la fusion, la méthode la plus simple gagne** : une moyenne simple dégrade la vidéo seule, mais des
  poids choisis en validation croisée (vidéo 0,6, texte 0,2, audio 0,2) la dépassent. Avec 2 207 extraits
  d'apprentissage, apprendre trois poids est plus sûr qu'apprendre des milliers de paramètres (fusion précoce,
  attention multimodale), même à entrées égales.
- **Test est plus difficile que dev** pour la vidéo et le texte : la fusion reste la meilleure sur test
  (0,419), mais son gain sur la vidéo seule n'y est plus significatif.

## Conclusion et limites

Un protocole commun aux trois modalités (tous les choix en validation croisée par film sur train, dev pour
mesurer, test une seule fois) donne des scores comparables et fiables. Le meilleur système fusionne les trois
modalités par une moyenne pondérée : F1 macro 0,486 sur dev et 0,419 sur test.

Limites : annotation surtout visuelle, qui avantage la vidéo ; peu de données (2 207 extraits
d'apprentissage, 50 films en dev et en test) ; transcription non évaluée par un taux d'erreur sur les mots ;
réglages choisis sur les mêmes prédictions hors pli, donc F1 cv légèrement optimiste. Pistes : des poids de
fusion par émotion (plus de texte pour le dégoût), l'affinage d'un encodeur de parole, un grand modèle
multimodal avec plus de mémoire graphique.

## Répartition du travail

| Partie | Auteur |
|---|---|
| Points 3, 4 texte et 4 vidéo | Hanchen LI |
| Points 4 audio et 5 | Gaoussou Maouloud KONTA |
