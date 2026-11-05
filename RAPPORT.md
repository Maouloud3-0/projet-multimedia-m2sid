# Reconnaissance d'émotions dans des extraits de film à partir du texte, de l'audio et de la vidéo

**Projet Données multimédia — M2 SID, groupe 4 — novembre 2026**

---

## Sommaire

1. [Introduction](#1-introduction)
2. [Présentation des données](#2-présentation-des-données-point-2)
3. [Transcription de l'audio](#3-transcription-de-laudio-point-3)
4. [Méthodes monomodales](#4-méthodes-monomodales-point-4)
5. [Méthodes multimodales](#5-méthodes-multimodales-point-5)
6. [Analyse des résultats](#6-analyse-des-résultats)
7. [Conclusion et limites](#7-conclusion-et-limites)
8. [Annexe : organisation du dépôt et reproduction](#8-annexe--organisation-du-dépôt-et-reproduction)

---

## 1. Introduction

L'objectif est de reconnaître la catégorie d'émotion d'un extrait de film parmi huit : *amusement*,
*anger* (colère), *awe* (émerveillement), *contentment* (satisfaction), *disgust* (dégoût), *excitement*,
*fear* (peur) et *sadness* (tristesse). Chaque extrait offre trois sources d'information : les images, la
bande son, et un texte (une description de la scène fournie avec le corpus, et les dialogues, que nous avons
transcrits).

Le travail suit les points de l'énoncé :

| Point | Travail | Dossier du dépôt |
|---|---|---|
| 2 | Analyse des données et partition train / dev / test | `point2_partition/` |
| 3 | Transcription des dialogues | `point3_transcription/` |
| 4 | Un modèle par modalité : texte, audio, vidéo | `point4_texte/`, `point4_audio/`, `point4_video/` |
| 5 | Fusion des trois modalités | `point5_fusion/` |

**Résultat principal.** Le meilleur système fusionne les trois modalités par une moyenne pondérée de leurs
probabilités. Il atteint un F1 macro de **0,486 sur dev** et **0,419 sur test** (hasard : 0,125), contre 0,461
et 0,413 pour la vidéo seule, la meilleure modalité.

---

## 2. Présentation des données (point 2)

### 2.1 Le corpus

Le corpus est la section *movie* d'EmoVid (Qiu et al., 2025, arXiv 2511.11002), restreinte pour le projet.
Les données disponibles comptent **350 films et 3 088 extraits**. Chaque extrait comprend :

- une vidéo MP4 (H264, 1280 × 720) ;
- la bande son séparée en MP3 (44,1 kHz, stéréo) ;
- un fichier JSON avec l'émotion, une description textuelle de la scène (*caption*), la durée et trois
  indicateurs d'image (luminosité, colorimétrie, teinte).

| Caractéristique | Valeur |
|---|---|
| Durée des extraits | 4 à 30 s, médiane 6,5 s |
| Description (*caption*) | 53 mots en moyenne |
| Extraits avec dialogue anglais (après transcription) | 2 025 (66 %), 16 mots en moyenne |
| Extraits avec au moins un visage détecté | 2 849 (92 %) |

Les descriptions décrivent ce que l'on voit (« un homme aux cheveux roux, en chemise bleue, devant un
grillage »), rarement ce qu'on ressent.

### 2.2 Une annotation surtout visuelle

D'après l'article EmoVid, environ 80 % des émotions ont été attribuées automatiquement par un modèle
vision-langage, NVILA-Lite-2B, qui ne voit que les images, puis contrôlées par des annotateurs. **Les
étiquettes reflètent donc surtout le contenu visuel.** Ce point explique une grande partie des résultats :
la vidéo est la modalité la plus performante, et l'audio la plus faible.

### 2.3 Déséquilibre des classes

| Émotion | Part des extraits |
|---|---:|
| anger | 21,9 % |
| fear | 16,8 % |
| contentment | 12,1 % |
| excitement | 11,5 % |
| sadness | 10,9 % |
| disgust | 9,6 % |
| awe | 9,3 % |
| amusement | 7,9 % |

La classe la plus fréquente est près de trois fois plus représentée que la plus rare. Un modèle qui
prédirait toujours *anger* aurait 22 % d'exactitude. Nous utilisons donc le **F1 macro**, qui donne le même
poids aux huit émotions, et nous pondérons les classes pendant l'apprentissage.

### 2.4 Partition

Les extraits d'un même film partagent les acteurs, les décors et la musique. Un modèle évalué sur un film
déjà vu en apprentissage serait jugé trop favorablement. **La partition se fait donc par film** : 250 films
pour l'apprentissage, 50 pour le développement et 50 pour le test.

Chaque groupe a proposé une partition ; une seule a été retenue, commune à tous les groupes. Ce n'est pas
celle de notre groupe, qui visait 12,5 % de chaque émotion en dev et en test. La partition retenue garde au
contraire les proportions du corpus dans chaque ensemble :

| | train | dev | test |
|---|---:|---:|---:|
| Films | 250 | 50 | 50 |
| Extraits | 2 207 | 442 | 439 |
| anger | 21,9 % | 22,2 % | 21,9 % |
| amusement | 7,9 % | 7,9 % | 8,0 % |
| Écart maximal à la proportion du corpus, toutes émotions | 0,2 pt | 0,3 pt | 0,4 pt |

Aucun film n'est dans deux ensembles (vérifié par `point2_partition/statistiques.py`).

### 2.5 Nettoyage : les écrans de fin

Certains extraits se terminent par un écran publicitaire de la chaîne Movieclips (bandeau gris, bouton bleu).
Un script les détecte sans utiliser les étiquettes (`point4_video/00_nettoyer_pub.py`) : 33 extraits sont
concernés. Les images et le son sont arrêtés avant cet écran dans tous les modules.

---

## 3. Transcription de l'audio (point 3)

Les dialogues sont une information que le corpus ne fournit pas. Il n'existe aucune transcription de
référence, et seulement environ 4 heures de parole : entraîner un système de reconnaissance de la parole est
hors de portée. Nous utilisons des modèles pré-entraînés, et le travail porte sur le contrôle de leur qualité.

### 3.1 Premier essai : un seul modèle

Whisper `large-v3-turbo`, avec détection de parole (Silero VAD) et filtrage des segments douteux, a été
appliqué à l'audio brut et à la voix isolée par Demucs :

| Traitement | Extraits avec parole | Textes modifiés par rapport au brut |
|---|---:|---:|
| Audio brut | 2 119 (68,6 %) | — |
| Voix isolée (Demucs) | 2 167 (70,2 %) | 1 332 (43,1 %) |
| Filtre passe-bande 300–3 400 Hz | mêmes textes (12 extraits testés) | — |

La séparation de la voix modifie 43 % des textes, mais sans référence on ne peut pas savoir lesquels sont
meilleurs : elle retrouve parfois un mot couvert par la musique et en déforme parfois un autre. Le filtre
fréquentiel ne sert à rien, la musique occupant la même bande que la voix.

### 3.2 Chaîne retenue : deux modèles qui se contrôlent, et vérification humaine

1. Deux modèles indépendants transcrivent chaque extrait : **Whisper large-v3** et **Qwen3-ASR-1.7B**.
2. La présence de parole et de musique est mesurée (Silero VAD, YAMNet).
3. Pour les extraits difficiles, la voix est séparée (BandIt) puis retranscrite.
4. Si les deux modèles sont d'accord et qu'il y a de la parole, le texte est accepté ; sinon l'extrait part
   en **vérification à l'écoute**, avec la vidéo, dans Label Studio.
5. Les textes sont nettoyés et la langue est détectée ; seul le dialogue anglais est gardé, car les modèles
   texte utilisés ensuite sont anglais.

| Résultat | Extraits |
|---|---:|
| Vérifiés à l'écoute | 1 032 |
| … dont texte de Qwen / de Whisper retenu | 690 / 234 |
| … dont pas de dialogue / corrigé à la main | 63 / 17 |
| Avec dialogue anglais (corpus final) | 2 025 (66 %) |
| Sans dialogue | 994 |
| Autre langue ou incertain | 69 |

Le résultat, `corpus.csv`, réunit pour chaque extrait la partition, l'émotion, la description et le dialogue.
C'est l'entrée commune de tous les modèles du point 4.

**Limites.** Les 2 056 extraits acceptés automatiquement peuvent contenir des erreurs ; aucun taux d'erreur
sur les mots n'a été mesuré, faute de référence indépendante. Les cris, rires et pleurs ne sont pas des mots
et ne sont pas transcrits, alors qu'ils portent beaucoup d'émotion : c'est le rôle du modèle audio.

---

## 4. Méthodes monomodales (point 4)

### 4.1 Protocole commun

Les trois modalités suivent exactement le même protocole, ce qui rend leurs scores comparables et permet la
fusion.

- **Tous les choix sont faits sur train**, par validation croisée à 5 plis **groupés par film** : réglages
  des classifieurs, couches des encodeurs, nombre de modèles combinés. **Dev ne sert qu'à mesurer** les
  modèles retenus. **Test** n'est utilisé qu'une fois, au point 5.
- Dev ne compte que 50 films : y choisir des réglages fausse la mesure. Par exemple, le système texte N2
  affichait 0,405 sur dev quand il était réglé sur dev, contre 0,366 une fois réglé en validation croisée.
- **F1 macro** comme mesure principale, classes pondérées pendant l'apprentissage.
- **Trois niveaux** de méthodes, du plus simple au plus riche :
  - **N1** : descripteurs classiques et régression logistique ;
  - **N2** : encodeurs pré-entraînés **gelés** (non modifiés) qui transforment l'extrait en vecteur, puis un
    classifieur (régression logistique, analyse discriminante ou SVM). Un encodeur est une pile de couches :
    chacune transforme les vecteurs de la précédente, et l'on peut lire les vecteurs à la sortie de
    n'importe laquelle. L'encodeur tourne une seule fois sur les 3 088 extraits, sans voir les étiquettes, et
    l'on enregistre les vecteurs de toutes ses couches ; la validation croisée porte ensuite uniquement sur le
    classifieur, pour choisir la couche et les autres réglages ;
  - **N3** : une partie apprise en plus de l'encodeur : fine-tuning (texte) ou tête d'attention sur la suite
    d'images ou de tranches de son (vidéo, audio). Au lieu de résumer l'extrait par une moyenne, la tête
    d'attention donne une note d'importance à chacun des 16 morceaux (une image, ou une tranche de temps),
    fait une moyenne pondérée par ces notes, puis prédit l'émotion : elle peut ainsi privilégier le moment
    d'un cri ou l'image d'un visage.
- **Combinaison finale** de chaque modalité : les modèles sont classés par F1 en validation croisée, puis on
  fait la moyenne des probabilités des *k* meilleurs ; *k* est choisi en validation croisée.
- Chaque modèle enregistre ses probabilités sur les 3 088 extraits, et des **probabilités hors pli** sur
  train : chaque extrait de train y est prédit par le modèle appris sur les quatre autres plis. Ces
  prédictions « honnêtes » servent à choisir les combinaisons et la fusion sans toucher à dev.

### 4.2 Texte

**Entrée** : la description et le dialogue.

| Niveau | Méthode | F1 cv | F1 dev |
|---|---|---:|---:|
| N1 | TF-IDF (mots et bigrammes) + régression logistique | 0,295 | 0,295 |
| N2 | 5 encodeurs gelés (mpnet, e5-large, bge-large, deux modèles d'émotion) + TF-IDF, combinés | 0,352 | 0,366 |
| N3 | Fine-tuning de 6 modèles (taux d'apprentissage choisi en CV), combinaison de 4 affinés et 2 gelés | **0,362** | **0,387** |

- Les encodeurs pré-entraînés dépassent le TF-IDF (e5 : +3,5 points en CV) : ils connaissent les synonymes
  et le contexte.
- Le dialogue seul est faible (F1 ≈ 0,17) ; ajouté à la description, il aide un peu.
- Le fine-tuning aide les grands modèles (ekman, un RoBERTa-large déjà entraîné sur les émotions : +1,9
  point), pas les petits.
- La combinaison progresse parce que les modèles affinés font des erreurs différentes.

### 4.3 Vidéo

**Entrée** : 16 images par extrait, prises au milieu de 16 intervalles égaux (avant l'écran de fin).

| Niveau | Méthode | F1 cv | F1 dev |
|---|---|---:|---:|
| N1 | Couleurs (HSV), luminosité, variation entre images + régression logistique | 0,192 | 0,200 |
| N2 | PE (Perception Encoder) gelé, moyenne des 16 images + visages + régression logistique | 0,430 | 0,441 |
| N3 | PE + visages, tête d'attention sur les 16 images | 0,421 | 0,454 |
| Final | Moyenne de 6 modèles (PE, SigLIP 2, CLIP-B, CLIP-H, en N2 et N3) | **0,450** | **0,461** |

- Sept encodeurs d'images ont été comparés (CLIP-B/L/H, SigLIP 2, PE, DINOv2, VideoMAE). Les modèles
  image-texte récents (PE, SigLIP 2) sont les meilleurs ; DINOv2 (images seules) et VideoMAE (mouvement) sont
  derrière.
- Les **visages** sont détectés (YuNet) et décrits par deux modèles d'expression faciale (ViT entraîné sur
  FER2013, HSEmotion entraîné sur AffectNet). Seuls, ils donnent 0,304 ; ajoutés aux images, ils aident
  toutes les configurations retenues.
- La tête d'attention apprend quelles images comptent. Elle ne bat pas toujours la simple moyenne, mais elle
  fait des erreurs différentes, utiles à la combinaison.

### 4.4 Audio

**Entrée** : la bande son, arrêtée avant l'écran de fin, 30 s au plus.

**N1 — descripteurs classiques (221 valeurs).** eGeMAPS v02 (openSMILE, 88 valeurs : hauteur, intensité,
qualité de la voix, rythme ; c'est le jeu de référence pour l'émotion dans la voix), complété par des
descripteurs librosa (MFCC et leurs dérivées, chroma, contraste spectral, centroïde, énergie, passages par
zéro, attaques, tempo).

**N2 — cinq encodeurs gelés**, choisis pour couvrir la parole, l'émotion vocale et les sons :

| Encodeur | Spécialité | Meilleure configuration | F1 cv | F1 dev |
|---|---|---|---:|---:|
| Whisper large-v3 (encodeur) | parole multilingue, robuste au bruit | couche 16 sur 32 | **0,265** | 0,267 |
| WavLM large | parole, auto-supervisé | moyenne des couches + N1 | 0,256 | 0,240 |
| wav2vec2, affiné sur l'émotion | valence, éveil, dominance | moyenne des couches + N1 + AudioSet | 0,249 | 0,263 |
| CLAP | son-texte, musique et parole | — | 0,245 | 0,229 |
| AST | sons et musique (AudioSet) | couche 4 sur 12 | 0,241 | 0,219 |
| Probabilités AudioSet seules | 527 catégories de sons | — | 0,213 | 0,214 |

Pour chaque encodeur, on garde, couche par couche, la moyenne et l'écart-type dans le temps. On compare
ensuite, en validation croisée : la couche, le résumé dans le temps, l'ajout des descripteurs N1 et AudioSet,
et trois classifieurs (régression logistique, analyse discriminante, SVM réglé par Optuna). Au total,
76 configurations ont été évaluées.

**N3 — tête d'attention sur 16 tranches de temps**, la même que pour la vidéo, les 16 images étant remplacées
par 16 tranches de même durée. 30 essais Optuna par encodeur, puis 3 graines.

| Encodeur | N2 (F1 cv) | N3 attention (F1 cv) | N3 (F1 dev) |
|---|---:|---:|---:|
| Whisper | 0,265 | 0,258 | 0,268 |
| WavLM | 0,256 | 0,258 | 0,255 |
| wav2vec2 émotion | 0,249 | 0,242 | 0,265 |
| AST | 0,241 | 0,235 | 0,243 |

**Système audio final** : moyenne de 4 modèles (Whisper et WavLM, chacun en N2 et N3), *k* = 4 choisi en
validation croisée. **F1 cv 0,271, dev 0,270**, exactitude 28,5 %.

Ce que l'on retient de l'audio :

- **Les couches du milieu de Whisper sont les plus utiles** : 0,265 pour la couche 16, contre 0,245 pour la
  dernière et 0,186 pour la première. Les dernières couches se spécialisent dans la transcription, celles du
  milieu gardent le ton de la voix.
- **Les modèles de parole** (Whisper, WavLM) **dépassent les modèles de sons et de musique** (AST, CLAP).
- **La régression logistique est meilleure partout** ; le SVM tombe à 0,16–0,20 après calibration.
- Une fois les encodeurs utilisés, les descripteurs N1 et AudioSet n'apportent presque rien.
- Une première version (AST + N1 + SVM) affichait 0,269 sur dev, mais avec des réglages choisis sur dev et
  sans probabilités hors pli : elle a été refaite avec le protocole commun, pour un score équivalent mais
  mesuré honnêtement et utilisable pour la fusion.

### 4.5 Bilan monomodal

| Modalité | F1 cv | F1 dev | Exactitude dev |
|---|---:|---:|---:|
| Audio | 0,271 | 0,270 | 28,5 % |
| Texte | 0,362 | 0,387 | 38,9 % |
| Vidéo | 0,450 | 0,461 | 45,7 % |

L'ordre vidéo > texte > audio est cohérent avec l'annotation, faite à partir des images. Le texte profite de
la description, qui décrit elle aussi l'image.

---

## 5. Méthodes multimodales (point 5)

### 5.1 Trois façons de fusionner

| Option | Principe | Ce qui est appris |
|---|---|---|
| **A — fusion tardive** | Chaque modalité donne ses 8 probabilités ; on les combine | 3 poids, ou un petit classifieur |
| **B — fusion précoce** | On met bout à bout les vecteurs des trois modalités, puis un seul classifieur | Un classifieur sur 5 413 valeurs |
| **C — modèle multimodal à attention** | Une tête d'attention regarde ensemble les images, le texte et les tranches de son | Un petit réseau |

Le même protocole s'applique : tous les choix en validation croisée sur train, à partir des probabilités ou
des vecteurs hors pli ; dev ne sert qu'à mesurer ; le système retenu est celui qui a le meilleur F1 en
validation croisée ; test est utilisé une seule fois, à la fin.

Pour B et C, chaque modalité est représentée par son meilleur encodeur gelé : PE + visages pour la vidéo,
e5-large pour le texte, Whisper (couche 16) pour l'audio. Les vecteurs vidéo ont été recalculés à partir des
scripts du module vidéo ; ils redonnent exactement ses scores (0,4241 en CV, 0,4110 sur dev pour PE seul, sans les visages),
ce qui confirme que le module se reproduit.

### 5.2 Option A — fusion tardive

Trois variantes : **moyenne simple** ; **moyenne pondérée**, avec un poids par modalité choisi de 0 à 1 par
pas de 0,1 ; **stacking**, une régression logistique sur les 24 log-probabilités (3 × 8).

| Méthode | Poids ou réglage | F1 cv | F1 dev |
|---|---|---:|---:|
| Vidéo seule | — | 0,450 | 0,461 |
| Moyenne simple | 1 / 1 / 1 | 0,435 | 0,451 |
| **Moyenne pondérée** | **vidéo 0,3, texte 0,1, audio 0,1 (soit 0,6 / 0,2 / 0,2)** | **0,455** | **0,486** |
| Stacking | C = 0,003 | 0,412 | 0,452 |

Apport de chaque modalité (moyenne pondérée) :

| Modalités | F1 cv | F1 dev |
|---|---:|---:|
| Vidéo + texte | 0,451 | 0,474 |
| Vidéo + audio | 0,450 | 0,461 |
| Vidéo + texte + audio | **0,455** | **0,486** |

- La **moyenne simple dégrade** la vidéo seule : elle donne autant de poids à l'audio (0,271) qu'à la vidéo
  (0,450).
- Le **stacking** a trop de paramètres pour 2 207 exemples : moins bon que la moyenne simple en CV.
- **Seul, l'audio n'apporte rien à la vidéo** (poids choisi : 0) ; **avec le texte, il gagne un poids égal à
  celui du texte**, et le trio dépasse la paire vidéo + texte.

### 5.3 Option B — fusion précoce

Les vecteurs sont concaténés (vidéo 3 108 valeurs, texte 1 025, audio 1 280), standardisés, puis une
régression logistique est apprise. Deux variantes : **brute** (chaque colonne compte autant) et
**équilibrée** (chaque modalité reçoit le même poids total).

| Variante | Modalités | F1 cv | F1 dev |
|---|---|---:|---:|
| Brute | vidéo seule (référence) | 0,431 | 0,441 |
| Brute | vidéo + texte + audio | 0,428 | 0,438 |
| Équilibrée | vidéo + texte + audio | 0,415 | 0,438 |
| Brute | vidéo + texte | 0,428 | 0,469 |

En principe, cette fusion peut apprendre des combinaisons (image sombre + voix calme ≠ image sombre + cris).
En pratique, **elle ne bat pas la vidéo seule** : avec 2 207 exemples, les colonnes du texte et de l'audio
ajoutent surtout du bruit, et une seule régularisation pour toutes les colonnes ne permet pas de leur donner
moins de poids. Donner le même poids aux trois modalités aggrave le problème.

### 5.4 Option C — modèle multimodal à attention

Chaque extrait devient une suite de 33 « morceaux » : 16 images (PE + plus grand visage), 1 texte (e5) et 16
tranches de son (Whisper). Chaque modalité est projetée dans un même espace, avec un vecteur appris qui
indique la modalité. Une couche de *self-attention* entre morceaux est possible. Un score par morceau donne
ensuite une moyenne pondérée, puis une couche de sortie prédit les 8 émotions. Pendant l'apprentissage, des
morceaux, et parfois une modalité entière, sont retirés au hasard, pour éviter que le modèle ne s'appuie que
sur la vidéo. Les encodeurs restent gelés. 30 essais Optuna en validation croisée, puis 3 graines.

| | F1 cv | F1 dev | Part de l'attention sur dev (vidéo / texte / audio) |
|---|---:|---:|---|
| Attention multimodale | 0,439 | 0,450 | 95 % / 3 % / 2 % |

Optuna retient un modèle **sans self-attention**. Le modèle dépasse la vidéo seule avec les mêmes entrées
(0,431) et la fusion précoce (0,428), mais **il regarde presque uniquement les images** : un résultat
cohérent avec l'annotation du corpus.

Un grand modèle multimodal pré-entraîné (du type Qwen2.5-Omni) n'a pas été essayé : il ne tient pas sur une
carte graphique de 8 Go, et l'affiner sur 2 207 exemples poserait le même problème de données.

### 5.5 Comparaison et système retenu

| Option | Système | F1 cv | F1 dev |
|---|---|---:|---:|
| A | Moyenne pondérée des systèmes finaux du point 4 | **0,455** | **0,486** |
| A | Moyenne pondérée, mêmes entrées que B et C (un encodeur gelé par modalité) | 0,448 | 0,454 |
| C | Attention multimodale | 0,439 | 0,450 |
| B | Concaténation, brute | 0,428 | 0,438 |

**À entrées égales, la fusion tardive reste devant** (0,448 contre 0,439 et 0,428) : apprendre trois poids
est plus sûr qu'apprendre des milliers de paramètres avec 2 207 exemples. Elle a en plus l'avantage de
profiter de tout le travail du point 4 (fine-tuning du texte, combinaisons de modèles), ce que B et C ne
peuvent pas faire.

**Système retenu : option A, moyenne pondérée (vidéo 0,6, texte 0,2, audio 0,2).**

---

## 6. Analyse des résultats

### 6.1 Scores finaux, test compris

Le test a été calculé une seule fois, pour le système retenu et pour les trois modalités seules :

| Système | F1 cv | F1 dev | F1 test |
|---|---:|---:|---:|
| Audio seul | 0,271 | 0,270 | 0,297 |
| Texte seul | 0,362 | 0,387 | 0,365 |
| Vidéo seule | 0,450 | 0,461 | 0,413 |
| **Fusion (moyenne pondérée)** | **0,455** | **0,486** | **0,419** |

| | F1 macro | Exactitude | Gain sur la vidéo seule (IC 95 % par films) |
|---|---:|---:|---|
| dev | 0,486 | 48,2 % | +0,025 [+0,001 ; +0,053] |
| test | 0,419 | 44,4 % | +0,006 [−0,017 ; +0,030] |

Les intervalles de confiance sont obtenus en rééchantillonnant les **films** (2 000 tirages), puisque les
extraits d'un même film ne sont pas indépendants.

- **Dev et validation croisée sont cohérents** pour toutes les modalités : le protocole ne surestime pas les
  scores.
- **Test est plus difficile que dev** pour la vidéo (0,413 contre 0,461) et le texte (0,365 contre 0,387),
  mais pas pour l'audio (0,297 contre 0,270). Avec 50 films par ensemble, quelques films difficiles suffisent
  à déplacer le score de plusieurs points.
- **La fusion reste la meilleure sur test**, mais son avance sur la vidéo n'y est plus significative.

### 6.2 Par émotion

F1 par émotion sur dev :

| Émotion | Audio | Texte | Vidéo | Fusion |
|---|---:|---:|---:|---:|
| amusement | 0,29 | 0,53 | 0,53 | **0,58** |
| anger | 0,35 | 0,36 | 0,42 | **0,46** |
| awe | 0,17 | 0,34 | 0,46 | **0,49** |
| contentment | 0,30 | 0,26 | **0,53** | 0,50 |
| disgust | 0,07 | **0,43** | 0,27 | 0,32 |
| excitement | 0,28 | 0,32 | 0,48 | **0,52** |
| fear | 0,31 | 0,48 | 0,54 | **0,56** |
| sadness | **0,40** | 0,38 | 0,46 | **0,46** |

- La fusion améliore ou égale la vidéo sur 7 émotions sur 8.
- **Le texte est la meilleure modalité pour *disgust*** (0,43 contre 0,27 pour la vidéo) : le dégoût se dit
  ou se décrit plus qu'il ne se voit. La fusion, qui donne peu de poids au texte, n'en récupère qu'une partie
  (0,32).
- **L'audio est le plus proche de la vidéo pour *sadness*** (0,40) : la tristesse s'entend dans la voix et
  la musique. Il est en revanche presque aveugle au dégoût (0,07) et à l'émerveillement (0,17).
- *awe* et *excitement* restent parmi les émotions les moins bien reconnues sur test (0,29 et 0,32 pour la
  fusion). Avec *contentment*, ce sont trois émotions positives proches, que le module texte confond le plus
  souvent entre elles.

### 6.3 Ce que chaque modalité apporte

| Constat | Explication |
|---|---|
| La vidéo domine | Annotation faite à partir des images ; encodeurs image-texte très performants |
| Le texte aide surtout grâce à la description | La description décrit l'image ; le dialogue seul est faible (F1 ≈ 0,17) et absent d'un tiers des extraits |
| L'audio est faible seul, mais utile en complément | Il porte des indices différents (voix, musique), surtout pour la tristesse |
| Les fusions qui apprennent beaucoup échouent | 2 207 extraits d'apprentissage ne suffisent pas pour des milliers de paramètres |

---

## 7. Conclusion et limites

### Conclusion

Nous avons construit, pour chaque modalité, une chaîne allant de descripteurs classiques (N1) à des
encodeurs pré-entraînés (N2) et à des modèles partiellement appris (N3), avec un protocole commun : tous les
choix par validation croisée groupée par film sur train, dev réservé à la mesure, test utilisé une seule
fois.

- La **vidéo** est la meilleure modalité (F1 dev 0,461), devant le **texte** (0,387) et l'**audio** (0,270).
- Les **encodeurs pré-entraînés gelés** font l'essentiel du travail ; le fine-tuning et l'attention apportent
  surtout de la diversité, utile aux combinaisons.
- La **fusion tardive par moyenne pondérée** est la meilleure façon de combiner les trois modalités. Elle
  dépasse la vidéo seule (0,486 contre 0,461 sur dev, gain significatif ; 0,419 contre 0,413 sur test), alors
  que la fusion précoce et le modèle multimodal à attention, plus ambitieux, ne font pas mieux avec si peu de
  données.

### Limites

- **Biais de l'annotation.** 80 % des étiquettes viennent d'un modèle qui ne voit que les images : nos scores
  mesurent en partie la capacité à retrouver le jugement de ce modèle, ce qui avantage la vidéo et pénalise
  l'audio.
- **Peu de données.** 2 207 extraits d'apprentissage limitent les modèles qui apprennent beaucoup de
  paramètres (stacking, fusion précoce, attention, fine-tuning des petits modèles).
- **Petits ensembles d'évaluation.** 50 films en dev comme en test : les écarts de un ou deux points ne sont
  pas significatifs, et test s'est révélé plus difficile que dev.
- **Transcription non évaluée.** Aucun taux d'erreur sur les mots n'a été mesuré ; les 2 056 transcriptions
  acceptées automatiquement peuvent contenir des erreurs. Les cris, rires et pleurs ne sont pas transcrits.
- **Choix des réglages sur les mêmes prédictions hors pli.** Les scores de validation croisée sont donc
  légèrement optimistes ; dev et test servent de contrôle.
- **Pas de modèle déployable.** Les scripts gardent les vecteurs et les probabilités, pas les poids des
  classifieurs : prédire une vidéo extérieure demanderait une chaîne d'inférence dédiée.

### Pistes

- Pondérer la fusion **par émotion** (par exemple, plus de poids au texte pour *disgust*), avec une
  validation croisée pour éviter le surapprentissage.
- Mesurer la qualité de la transcription sur un échantillon de dev transcrit à la main.
- Affiner un encodeur audio de parole (WavLM ou Whisper) plutôt que de le garder gelé.
- Tester un grand modèle multimodal avec plus de mémoire graphique, en le gardant gelé.

---

## 8. Annexe : organisation du dépôt et reproduction

```
point2_partition/       listes train / dev / test, statistiques.py
point3_transcription/   01 à 09 : préparation, transcription, vérification, corpus.csv
point4_texte/           01 à 06 : TF-IDF, encodeurs gelés, fine-tuning, combinaisons
point4_video/           00 à 07 : écrans de fin, images, encodeurs, visages, N1, N2, N3, combinaison
point4_audio/           01 à 05 : N1, encodeurs, N2, N3, combinaison
point5_fusion/          01 à 04 : fusion tardive, précoce, attention, comparaison et test
```

Chaque dossier a son README (méthode, tableaux complets, commandes) et son `requirements.txt`. Les résultats
(`resultats/` : scores, matrices de confusion, probabilités) sont versionnés ; les fichiers intermédiaires
lourds (`data/`) ne le sont pas.

**Données.** Par défaut, les scripts cherchent le corpus dans l'archive du cours décompressée à côté du
dépôt (`../PROJET_SID_2026/PROJET_SID_2026/movie/`), sinon dans la variable `EMOVID_DIR` (points 2 et 4) ou
`PROJET_SID_DIR` (point 3).

**Ordre.** Point 3 (ou directement son `corpus.csv` versionné), puis les trois modules du point 4 (la vidéo
avant l'audio, qui réutilise la détection des écrans de fin), puis le point 5.

**Matériel.** Python 3.11, une carte graphique de 8 Go (RTX 5070 Laptop). Durées indicatives : transcription
2 à 3 h, fine-tuning du texte 3 h 30, encodeurs vidéo et audio environ 1 h chacun, fusion moins de 45 min.

**Reproductibilité vérifiée.** Depuis ce dépôt, `corpus.csv` se reconstruit à l'identique, et les
combinaisons finales des trois modalités et la fusion redonnent exactement les scores de ce rapport.
