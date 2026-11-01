# Point 3 (variante) — Transcription vérifiée à la main

Transcription des dialogues des 3 088 extraits avec deux modèles indépendants
(Whisper large-v3 et Qwen3-ASR-1.7B), puis vérification manuelle dans Label Studio
des extraits où les modèles ne sont pas d'accord ou où la musique gêne.
Le résultat sert d'entrée au modèle texte ([point4_texte](../point4_texte/)).

Le corpus n'est pas versionné : son emplacement se règle avec la variable
d'environnement `PROJET_SID_DIR` (dossier qui contient `movie/`).

## Chaîne de traitement

```
movie/ (audio, vidéo, annotations d'origine)
  │ 01–05  automatique : WAV 16 kHz → Whisper + Qwen → parole / musique → séparation de la voix → aiguillage
  │ 06     vérification manuelle des extraits douteux dans Label Studio
  │ 07–08  décisions appliquées, nettoyage, détection de la langue → transcription finale
  │ 09     + annotations (émotion, description) + partition splits/
  ▼
resultats/corpus.csv
```

| Script | Rôle | Sortie |
|---|---|---|
| `01_preparer_audio.py` | Conversion des MP3 en WAV 16 kHz mono | `work/audio/` |
| `02_transcrire.py` | Whisper large-v3 et Qwen3-ASR-1.7B, extrait complet | `work/asr/` |
| `03_detecter_sons.py` | Parole, musique, chant : Silero VAD et YAMNet | `work/evidence/` |
| `04_separer_voix.py` | Extraits difficiles : séparation de la voix (BandIt) puis Qwen sur la voix seule | `work/recovery/` |
| `05_aiguiller.py` | Texte accepté, pas de dialogue, ou à vérifier | `resultats/aiguillage.jsonl` |
| `06_verification.py` | Vérification manuelle dans Label Studio (vidéo, audio, transcriptions) | `resultats/decisions_verification.jsonl` |
| `07_nettoyer.py` | Décisions appliquées, nettoyage léger, détection de la langue | `work/cleaned.jsonl` |
| `08_exporter.py` | Transcription finale | `resultats/transcriptions_verifiees.csv` |
| `09_construire_corpus.py` | Corpus : partition, émotion, description, dialogue | `resultats/corpus.csv` |

`work/` (dans le dossier du corpus) contient les fichiers intermédiaires lourds ;
`resultats/` est versionné.

## Reproduire le corpus

À partir des résultats versionnés (quelques secondes, résultat identique) :

```powershell
cd point3_transcription_verifiee
python 07_nettoyer.py
python 08_exporter.py
python 09_construire_corpus.py
```

Chaîne complète depuis `movie/` (environ 2 à 3 h sur GPU) :

```powershell
python 01_preparer_audio.py
python 02_transcrire.py
python 03_detecter_sons.py
python 04_separer_voix.py
python 05_aiguiller.py
python 06_verification.py preparer      # puis vérification dans Label Studio
python 06_verification.py lancer
python 06_verification.py exporter --input export_label_studio.json
python 07_nettoyer.py
python 08_exporter.py
python 09_construire_corpus.py
```

La vérification manuelle ne se refait pas : ses décisions sont conservées dans
`resultats/decisions_verification.jsonl` et réappliquées par `07_nettoyer.py`.
Whisper et Qwen ne sont pas parfaitement déterministes : une chaîne complète
peut aiguiller différemment quelques extraits, qui reprennent alors la proposition
automatique. `--limite 20` permet un essai rapide.

Les réglages sont dans `config/pipeline.json`, les chemins des modèles dans
`config/modeles.json`.

## Environnements

- `requirements.txt` : environnement principal.
- `requirements-dml.txt` : environnement `.venv-dml` pour BandIt (onnxruntime-directml).
- `requirements-review.txt` : environnement `.venv-review` pour Label Studio.
- `requirements-language.txt` : langid, installé dans `runtime/language_dependencies`.

Whisper et Qwen tournent avec whisper.cpp et llama.cpp (Vulkan).

## Résultats

Vérification manuelle : 1 005 extraits relus. Décisions : Qwen 690, Whisper 234,
pas de dialogue 63, Qwen sur la voix séparée 28, texte corrigé à la main 17.

| Ensemble | Films | Extraits | Dialogue anglais |
|---|---:|---:|---:|
| train | 250 | 2 207 | 1 450 |
| dev | 50 | 442 | 298 |
| test | 50 | 439 | 277 |
| **total** | **350** | **3 088** | **2 025** |

Sans dialogue : 994 extraits ; autre langue : 67 ; langue incertaine : 2.
Seul le dialogue anglais est gardé dans la colonne `dialogue` de `corpus.csv`.
