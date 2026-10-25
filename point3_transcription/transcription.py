"""Coeur de la transcription : chargement du modele et filtrage des sorties.

Ce module est partage par 03_transcrire.py et par 08_experience_filtres.py,
pour n'avoir qu'une seule definition de ce qu'est une transcription valable.
"""

from faster_whisper import WhisperModel

import config

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


def charger_modele():
    """Charge le modele une seule fois (l'operation la plus lente)."""
    return WhisperModel(
        config.CHEMIN_MODELE,
        device=config.DISPOSITIF,
        compute_type="float16" if config.DISPOSITIF == "cuda" else "int8",
    )


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


def transcrire(modele, chemin, avec_vad=True):
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
