"""03 - Transcrire les extraits avec Whisper et filtrer les segments douteux.

Whisper est un modele pre-entraine (Hugging Face / OpenAI). On ne l'entraine
pas : on le fait tourner sur nos extraits et on jette les sorties peu fiables,
car sur des extraits sans dialogue il invente du texte.

Un fichier JSON est ecrit par extrait dans data/transcriptions/.
Un extrait deja traite est saute : on peut arreter et relancer.

Usage :
    python 03_transcrire.py                 # piste voix de Demucs si elle existe
    python 03_transcrire.py --source wav    # audio brut converti (comparaison)
    python 03_transcrire.py --sans-vad      # sans detection de parole (comparaison)
    python 03_transcrire.py --limite 20     # essai sur 20 extraits
"""

import argparse
import time

from faster_whisper import WhisperModel

import config
import utils

# Phrases que Whisper invente sur les extraits sans parole.
PHRASES_INVENTEES = (
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "sous-titres",
    "subtitles by",
    "amara.org",
    "transcription by",
)


def choisir_source(source):
    """Dossier contenant les fichiers a transcrire."""
    if source == "voix" and any(config.VOIX.glob("*.wav")):
        return config.VOIX
    return config.WAV


def segment_valide(segment):
    """Garde un segment seulement s'il est fiable et pas invente."""
    if segment.no_speech_prob > config.SEUIL_SANS_PAROLE:
        return False
    if segment.avg_logprob < config.SEUIL_LOGPROB:
        return False
    if getattr(segment, "compression_ratio", 0) > config.SEUIL_COMPRESSION:
        return False

    texte = segment.text.strip().lower()
    if any(phrase in texte for phrase in PHRASES_INVENTEES):
        return False
    return len(texte) > 1


def transcrire(modele, chemin, avec_vad):
    """Transcrit un extrait et renvoie un dictionnaire de resultats."""
    morceaux, infos = modele.transcribe(
        str(chemin),
        language=config.LANGUE,
        vad_filter=avec_vad,
        vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False,   # limite les inventions
        beam_size=5,
    )
    morceaux = list(morceaux)
    gardes = [m for m in morceaux if segment_valide(m)]
    texte = " ".join(m.text.strip() for m in gardes).strip()

    duree = infos.duration or 0.0
    # Part de l'extrait ou le detecteur de parole a entendu quelqu'un.
    if avec_vad and duree:
        ratio = infos.duration_after_vad / duree
    else:
        ratio = None

    sans_parole = [m.no_speech_prob for m in morceaux]
    logprobs = [m.avg_logprob for m in gardes]

    return {
        "langue": infos.language,
        "duree": round(duree, 2),
        "texte": texte,
        "a_parole": bool(texte) and (ratio is None or ratio >= config.SEUIL_RATIO_PAROLE),
        "ratio_parole": None if ratio is None else round(ratio, 3),
        "no_speech_prob": round(max(sans_parole, default=0.0), 3),
        "logprob_moyen": round(sum(logprobs) / len(logprobs), 3) if logprobs else None,
    }


def main():
    analyseur = argparse.ArgumentParser(description="Transcription des extraits audio")
    analyseur.add_argument("--source", choices=["voix", "wav"], default="voix")
    analyseur.add_argument("--sans-vad", action="store_true",
                           help="desactive la detection de parole")
    analyseur.add_argument("--limite", type=int, default=0,
                           help="ne traiter que les N premiers extraits")
    options = analyseur.parse_args()

    utils.preparer_dossiers()
    dossier = choisir_source(options.source)
    avec_vad = not options.sans_vad

    # Une sortie par configuration, pour pouvoir les comparer sans les melanger.
    etiquette = dossier.name if avec_vad else dossier.name + "_sans_vad"
    dossier_sortie = config.TRANSCRIPTIONS / etiquette
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    print(f"Modele {config.CHEMIN_MODELE} | source {dossier.name} | VAD {avec_vad}")
    modele = WhisperModel(config.CHEMIN_MODELE, device=config.DISPOSITIF,
                          compute_type="float16" if config.DISPOSITIF == "cuda" else "int8")

    liste = utils.extraits()
    if options.limite:
        liste = liste[:options.limite]

    depart = time.time()
    faits = 0
    for extrait in liste:
        sortie = dossier_sortie / f"{extrait['video_id']}.json"
        if sortie.exists():
            continue

        wav = dossier / f"{extrait['video_id']}.wav"
        if not wav.exists():
            print("fichier manquant :", wav)
            continue

        resultat = {"video_id": extrait["video_id"], "film_id": extrait["film_id"]}
        resultat.update(transcrire(modele, wav, avec_vad))
        utils.ecrire_json(sortie, resultat)

        faits += 1
        if faits % 100 == 0:
            print(f"  {faits} extraits | {time.time() - depart:.0f} s")

    print(f"{faits} extraits transcrits ->", dossier_sortie)


if __name__ == "__main__":
    main()
