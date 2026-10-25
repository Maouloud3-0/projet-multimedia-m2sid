# Point 3 — Transcription de l'audio

Transformer les 3 088 extraits audio (MP3) en texte, en limitant les erreurs
dues à la musique et aux bruits de fond des films.

## Pourquoi ne pas entraîner notre propre modèle ?

Entraîner un système de reconnaissance de la parole demanderait des milliers
d'heures de parole **déjà transcrite**. Le corpus en contient environ 4 heures,
et les annotations décrivent l'image, pas les dialogues : il n'existe aucune
transcription de référence pour apprendre.

On utilise donc un modèle **pré-entraîné de Hugging Face** (Whisper) et tout le
travail porte sur le **traitement des données** : préparation de l'audio,
séparation de la voix, détection de la parole, filtrage des sorties inventées,
puis mesure de la qualité.

## Chaîne de traitement

```
MP3 44,1 kHz stéréo
  │ 01  conversion          → WAV 16 kHz mono
  │ 02  séparation          → Demucs : on garde la piste "vocals"
  │ 03  transcription       → Whisper (+ Silero VAD) + filtrage
  │ 04  mise en tableau     → resultats/transcriptions.csv
  │ 05  évaluation          → WER + détection de parole
```

| Script | Rôle | Sortie |
|---|---|---|
| [01_convertir_audio.py](01_convertir_audio.py) | Décodage ffmpeg en 16 kHz mono | `data/wav/` |
| [02_separer_voix.py](02_separer_voix.py) | Demucs : voix conservée, musique et bruits écartés | `data/voix/` |
| [03_transcrire.py](03_transcrire.py) | Whisper + VAD + filtrage des segments douteux | `data/transcriptions/` |
| [04_construire_csv.py](04_construire_csv.py) | Assemblage avec émotion et ensemble (train/dev/test) | `resultats/transcriptions.csv` |
| [05_evaluer_wer.py](05_evaluer_wer.py) | Taux d'erreur sur les mots | affichage |

Les paramètres se règlent dans un seul fichier : [config.py](config.py).

## Installation

Python 3.10 à 3.12 recommandé.

```bash
cd point3_transcription
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Sur une carte graphique récente (RTX 50xx), installer PyTorch pour CUDA 12.8 :

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Utilisation

```bash
python 01_convertir_audio.py          # ~5 min
python 02_separer_voix.py             # ~30 min sur carte graphique
python 03_transcrire.py               # ~20 min sur carte graphique
python 04_construire_csv.py
python 05_evaluer_wer.py
```

Chaque étape saute les extraits déjà traités : on peut interrompre et relancer.

Pour un essai rapide sur 20 extraits :

```bash
python 03_transcrire.py --limite 20
```

## Comparaison des configurations

Les choix se font sur l'ensemble de **développement**, jamais sur le test.
Conserver une copie du CSV à chaque essai :

```bash
python 03_transcrire.py --source wav                 # audio brut
copy resultats\transcriptions.csv resultats\config_A.csv

python 03_transcrire.py                              # piste voix
copy resultats\transcriptions.csv resultats\config_C.csv
```

Remplir à la main 50 à 100 extraits dans
[reference/transcriptions_manuelles.csv](reference/transcriptions_manuelles.csv)
(trois colonnes : `video_id`, `texte`, `a_parole`), puis :

```bash
python 05_evaluer_wer.py --csv resultats/config_A.csv
```

| Configuration | Traitement | Modèle | WER | Détection de parole |
|---|---|---|---|---|
| A | audio brut | whisper-small | à mesurer | à mesurer |
| B | audio brut | large-v3-turbo | à mesurer | à mesurer |
| C | Demucs | large-v3-turbo | à mesurer | à mesurer |
| D | Demucs + VAD + filtrage | large-v3-turbo | à mesurer | à mesurer |

## Ce que contient le CSV

| Colonne | Signification |
|---|---|
| `video_id`, `film_id` | identifiants de l'extrait et du film |
| `partition` | `train`, `dev` ou `test` |
| `emotion` | émotion annotée (cible du point 4) |
| `langue` | langue détectée par Whisper |
| `a_parole` | 1 si des dialogues ont été trouvés, 0 sinon |
| `ratio_parole` | part de l'extrait couverte par de la parole |
| `no_speech_prob` | confiance de Whisper sur l'absence de parole |
| `logprob_moyen` | confiance moyenne du texte produit |
| `texte` | la transcription |

## À retenir pour la suite

Beaucoup d'extraits de 5 secondes ne contiennent **aucun dialogue**. Le modèle
texte du point 4 devra donc combiner la transcription et la description
(`caption`) du fichier JSON, sinon il sera aveugle sur ces extraits.
