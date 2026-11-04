# Audio : reconnaître l'émotion à partir de la bande son

## En bref

```
                 ┌──> Whisper (couche 16) + régression logistique ──┐
 bande son  ─────┼──> Whisper + attention ─────────────────────────┼──> moyenne des probabilités ──> 8 émotions
 (≤ 30 s)        ├──> WavLM + N1 + régression logistique ──────────┤
                 └──> WavLM + attention ───────────────────────────┘
```

| Étape | Système | F1 cv | F1 dev | Exactitude dev |
|---|---|---|---|---|
| — | Hasard | — | 0,125 | 12,5 % |
| N1 | eGeMAPS + descripteurs librosa + régression logistique | 0,212 | 0,224 | 22,6 % |
| N2 | Whisper large-v3 gelé (couche 16) + régression logistique | 0,269 | 0,267 | 28,1 % |
| N3 | Whisper gelé + tête d'attention sur 16 tranches | 0,264 | 0,268 | 28,1 % |
| **Final** | **moyenne de 4 modèles de N2 et N3** | **0,271** | **0,270** | **28,5 %** |

- Même protocole que [point4_texte](../point4_texte/) et [point4_video](../point4_video/) :
  tous les choix par validation croisée sur train, dev ne sert qu'à mesurer, test non utilisé.
- Les probabilités finales sont dans `resultats/probas_audio.npz`.
- L'audio reste la modalité la plus faible : selon l'article EmoVid, 80 % des émotions ont été
  annotées par un modèle qui regarde les images (NVILA-Lite-2B), sans le son.

## Les données

Entrées : les MP3 de `movie/audio/` et
[point3_transcription_verifiee/resultats/corpus.csv](../point3_transcription_verifiee/resultats/corpus.csv)
(identifiants, émotion, partition). Les extraits durent de 4 à 30 s (6,4 s en médiane). Le son est
arrêté avant l'écran de fin Movieclips repéré par
[point4_video/00_nettoyer_pub.py](../point4_video/00_nettoyer_pub.py) (33 extraits), et à 30 s au plus.

| | train | dev | test |
|---|---|---|---|
| Films | 250 | 50 | 50 |
| Extraits | 2 207 | 442 | 439 |

Le chemin vers `movie/` se règle dans [config.py](config.py), ou avec `EMOVID_DIR`.

## Comment on mesure

- Partition par film commune aux autres modalités.
- Validation croisée à 5 plis sur train, groupée par film ; standardisation et classifieurs réappris
  dans chaque pli.
- F1 macro comme métrique principale, classes pondérées.
- Dans les tableaux, **F1 cv** est calculé sur les probabilités hors pli regroupées (comme dans `05`).

## Étape 1 — N1 : descripteurs classiques

[01_audio_n1.py](01_audio_n1.py) : **eGeMAPS v02** (openSMILE, 88 valeurs : hauteur, intensité, qualité
de voix, rythme ; le jeu de référence pour l'émotion dans la voix) et des descripteurs librosa (MFCC
et dérivées, chroma, contraste spectral, centroïde, largeur de bande, rolloff, platitude, énergie,
passages par zéro, attaques ; moyenne et écart-type), plus le tempo : 221 caractéristiques.
Régression logistique, `C = 0.003`, classes équilibrées. F1 cv 0,212, dev 0,224.

## Étape 2 — N2 : encodeurs gelés

[02_encodeurs.py](02_encodeurs.py) transforme le son en vecteurs, sans utiliser les étiquettes ni
modifier les encodeurs. Pour chaque couche, on garde la moyenne et l'écart-type dans le temps.

| Encodeur | Modèle | Ce qu'il représente |
|---|---|---|
| Whisper | `openai/whisper-large-v3` (encodeur) | parole multilingue, robuste au bruit |
| WavLM | `microsoft/wavlm-large` | parole, auto-supervisé |
| wav2vec2 émotion | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | voix, affiné sur valence / éveil / dominance |
| AST | `MIT/ast-finetuned-audioset-10-10-0.4593` | sons, musique et voix (AudioSet) |
| CLAP | `laion/larger_clap_music_and_speech` | son-texte, musique et parole |

AST donne aussi ses probabilités sur les 527 catégories AudioSet (« musique », « cri », « rire »…),
utilisées comme descripteurs.

[03_audio_n2.py](03_audio_n2.py) compare, pour chaque encodeur : la couche (7 couches réparties ou
moyenne des couches), le résumé dans le temps (moyenne ou moyenne + écart-type), les descripteurs
ajoutés (N1, AudioSet, les deux) et trois classifieurs (régression logistique, LDA, SVM réglé par
Optuna puis calibré). 76 configurations au total.

| Sortie retenue | Couche | Ajout | Classifieur | F1 cv | F1 dev |
|---|---|---|---|---|---|
| Whisper | 16 sur 32 | — | rég. logistique | **0,265** | 0,267 |
| WavLM | moyenne des couches | N1 | rég. logistique | 0,256 | 0,240 |
| wav2vec2 émotion | moyenne (+ écart-type) | N1 + AudioSet | rég. logistique | 0,249 | 0,263 |
| CLAP | — | — | rég. logistique | 0,245 | 0,229 |
| AST | 4 sur 12 | — | rég. logistique | 0,241 | 0,219 |
| AudioSet seul | — | — | rég. logistique | 0,213 | 0,214 |

(F1 cv moyen par pli, d'où de petits écarts avec le tableau de `05`.)

Ce qu'on retient :

- **Les couches du milieu de Whisper sont les plus utiles** : 0,265 pour la couche 16, contre 0,245
  pour la dernière et 0,186 pour la première. Les dernières couches se spécialisent dans la
  transcription ; celles du milieu gardent le ton de la voix.
- Les modèles de parole (Whisper, WavLM) dépassent les modèles de sons et de musique (AST, CLAP).
- La régression logistique est meilleure partout ; le SVM tombe à 0,16–0,20 en CV.
- Les descripteurs N1 et AudioSet n'apportent presque rien une fois les encodeurs utilisés.

## Étape 3 — N3 : tête d'attention

[04_audio_n3.py](04_audio_n3.py) reprend la tête de [point4_video/06_video_n3.py](../point4_video/06_video_n3.py),
les 16 images étant remplacées par 16 tranches de même durée (moyenne des couches de l'encodeur sur
chaque tranche). 30 essais Optuna par encodeur, puis 3 graines.

| Encodeur | N2 (F1 cv) | N3 attention (F1 cv) | F1 dev de N3 |
|---|---|---|---|
| Whisper | **0,265** | 0,258 | 0,268 |
| WavLM | 0,256 | **0,258** | 0,255 |
| wav2vec2 émotion | **0,249** | 0,242 | 0,265 |
| AST | **0,241** | 0,235 | 0,243 |

Comme pour la vidéo, l'attention ne bat pas systématiquement la moyenne dans le temps, mais elle
donne des sorties différentes, utiles pour la combinaison.

## Système final — combinaison des modèles audio

[05_audio_fusion.py](05_audio_fusion.py) classe les 11 sorties (N1, six N2, quatre N3) par F1 hors pli
sur train, puis moyenne les probabilités des `k` premiers ; `k` est choisi en CV.

| k | 1 | 2 | 3 | **4** | 6 | 8 | 11 |
|---|---|---|---|---|---|---|---|
| F1 cv | 0,269 | 0,269 | 0,268 | **0,271** | 0,266 | 0,268 | 0,261 |

Retenus : Whisper et WavLM, chacun avec régression logistique et avec attention.
F1 cv 0,271, dev 0,270, exactitude dev 28,5 %. Le gain sur Whisper seul n'est pas significatif
(IC 95 % par films [−0,022 ; +0,028]).

Comparaison avec la première version de l'audio (`point4_modelisation/`, AST + N1 + SVM) :
0,269 sur dev, mais réglages choisis sur dev et pas de probabilités hors pli.

## Lancer les scripts

Environnement avec PyTorch et [requirements.txt](requirements.txt). Depuis `point4_audio/` :

```bash
python 01_audio_n1.py        # N1 (~30 min, processeur)
python 02_encodeurs.py       # vecteurs des 5 encodeurs (~30 min sur GPU 8 Go, ~6 Go de modèles)
python 03_audio_n2.py        # N2 (~1 h, processeur)
python 04_audio_n3.py        # N3 (~40 min sur GPU)
python 05_audio_fusion.py    # système final
```

`01` et `02` réutilisent les fichiers déjà présents dans `data/`, `03` reprend les configurations de
son CSV et `04` saute les encodeurs terminés.

## Fichiers produits

Dans `data/` (exclu de Git) : `n1.npy`, `couches_<encodeur>.npy`, `seg_<encodeur>.npy`, `audioset.npy`.

Dans `resultats/` :

- `scores_n1.csv`, `scores_n2.csv`, `scores_n3.csv`, `essais_n3_<encodeur>.csv`, `scores_fusion_audio.csv` ;
- `confusion_<modèle>.png` : matrices de confusion sur dev ;
- `probas_<modèle>.npz`, dont **`probas_audio.npz`** pour la fusion des modalités : `video_id`,
  `partition`, `probabilites` (3 088 × 8), `probabilites_hors_pli` (train, NaN ailleurs), `emotions`.
