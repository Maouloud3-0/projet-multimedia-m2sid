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
| [05_comparer_configurations.py](05_comparer_configurations.py) | Compare deux configurations sans référence | `resultats/comparaison_*.txt` |
| [06_evaluer_wer.py](06_evaluer_wer.py) | Taux d'erreur sur les mots (référence manuelle) | affichage |
| [telecharger_modele.py](telecharger_modele.py) | Télécharge un modèle Whisper sans passer par `huggingface_hub` | `.cache/` |

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

### Dépannage Windows

**`Library cublas64_12.dll is not found or cannot be loaded`**
CTranslate2 (le moteur de faster-whisper) a besoin de cuBLAS et cuDNN, que
PyTorch installe pour lui-même sans les rendre visibles. On copie les
bibliothèques dans le dossier de CTranslate2 :

```powershell
$torch = ".venv\Lib\site-packages\torch\lib"
$ct2   = ".venv\Lib\site-packages\ctranslate2"
Copy-Item "$torch\cublas64_12.dll","$torch\cublasLt64_12.dll","$torch\cudnn*64_9.dll","$torch\cudart*.dll" $ct2 -Force
```

**Modèle déjà téléchargé ailleurs**
Si les poids de Whisper sont déjà sur le disque, on évite un nouveau
téléchargement de 1,5 Go :

```powershell
$env:WHISPER_MODELE = "C:\chemin\vers\whisper-large-v3-turbo"
```

## Utilisation

```bash
python 01_convertir_audio.py                   # ~5 min
python 02_separer_voix.py                      # ~45 min sur carte graphique
python 03_transcrire.py                        # ~20 min sur carte graphique
python 04_construire_csv.py --source voix
python 05_comparer_configurations.py
python 06_evaluer_wer.py
```

Chaque étape saute les extraits déjà traités : on peut interrompre et relancer.

Pour un essai rapide sur 20 extraits :

```bash
python 03_transcrire.py --limite 20
```

## Choix techniques et justification

| Choix | Pourquoi |
|---|---|
| **Whisper pré-entraîné** plutôt qu'un modèle entraîné par nos soins | Entraîner un système de reconnaissance de la parole demande des milliers d'heures de parole **déjà transcrite**. Le corpus en contient environ 4 heures et **aucune transcription de référence** : il n'y a rien pour apprendre. C'est donc hors de portée d'un projet de cette taille. |
| **`large-v3-turbo`** | Meilleur compromis qualité / vitesse : environ 809 M paramètres, moins de 3 Go de mémoire, robuste au bruit et au multilingue. `small` sert de point de comparaison, `large-v3` de borne supérieure. |
| **Demucs (`htdemucs`)** | La musique et les effets sonores d'un film masquent les dialogues. Demucs isole la piste voix ; c'est le levier le plus efficace sur ce corpus. |
| **Silero VAD** | Beaucoup d'extraits de 5 s n'ont aucun dialogue. Sans détection de parole, Whisper **invente** du texte sur ces extraits (« Thank you for watching »…). Le VAD coupe ces passages avant la transcription. |
| **`condition_on_previous_text=False`** | Empêche le modèle de propager une erreur ou une invention d'un segment au suivant. |
| **Seuils de filtrage** (0,6 / −1,0 / 2,4) | Valeurs usuelles : au-delà, Whisper signale lui-même un doute (absence de parole, faible confiance, texte qui se répète en boucle). |

### Limites assumées

- Sans transcription de référence, on ne peut pas calculer un vrai WER ; on
  mesure ce qui est observable (détection de parole, stabilité entre
  configurations) et on annote un échantillon à la main.
- Les cris, rires et pleurs ne sont pas des mots : ils ne seront pas
  transcrits, alors qu'ils portent une forte charge émotionnelle. C'est au
  modèle audio du point 4 de les exploiter.
- Les extraits sans dialogue produiront un texte vide : le modèle texte du
  point 4 devra s'appuyer aussi sur la description (`caption`) du JSON.

## Comparaison des configurations

Les choix se font sur l'ensemble de **développement**, jamais sur le test.
Chaque configuration range ses résultats séparément, dans
`data/transcriptions/<source>/` et `resultats/transcriptions_<source>.csv` :

```bash
python 03_transcrire.py --source wav            # audio brut
python 04_construire_csv.py --source wav        # -> transcriptions_wav.csv

python 03_transcrire.py                         # piste voix de Demucs
python 04_construire_csv.py --source voix       # -> transcriptions_voix.csv
```

Remplir à la main 50 à 100 extraits dans
[reference/transcriptions_manuelles.csv](reference/transcriptions_manuelles.csv)
(trois colonnes : `video_id`, `texte`, `a_parole`), puis :

```bash
python 06_evaluer_wer.py --csv resultats/transcriptions_wav.csv
```

| Configuration | Traitement | Modèle | WER | Détection de parole |
|---|---|---|---|---|
| A | audio brut | whisper-small | à mesurer | à mesurer |
| B | audio brut | large-v3-turbo | à mesurer | à mesurer |
| C | Demucs | large-v3-turbo | à mesurer | à mesurer |
| D | Demucs + VAD + filtrage | large-v3-turbo | à mesurer | à mesurer |

## Ce que contient le CSV — `resultats/transcriptions_voix.csv`

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
