"""Signaux d'alerte sur une transcription et choix des extraits à reprendre après séparation."""
import difflib
import re
import unicodedata

from .routage import mots


def signaux(resultat):
    """Repère les sorties suspectes : texte vide, limite de longueur, boucle de répétition…"""
    alertes = []
    if not resultat["text"]:
        alertes.append("empty_output")
    if resultat.get("finish_reason") == "length":
        alertes.append("token_limit")
    if not resultat.get("format_valid", True):
        alertes.append("unexpected_response_format")
    if resultat["text"] and resultat.get("language") is None:
        alertes.append("unknown_language_label")
    if repetition(resultat["text"]):
        alertes.append("long_repetition_candidate")
    return sorted(set(alertes))


def repetition(texte):
    """Whisper part parfois en boucle (« thank you thank you … ») : on cherche un motif
    de 1 à 8 mots répété au moins 3 fois et couvrant au moins 20 mots ou 60 % du texte."""
    liste = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", texte).casefold())
    for taille in range(1, min(8, len(liste) // 3) + 1):
        for decalage in range(min(taille, len(liste))):
            blocs = [liste[i:i + taille] for i in range(decalage, len(liste) - taille + 1, taille)]
            suite = 1
            for i in range(1, len(blocs)):
                suite = suite + 1 if blocs[i] == blocs[i - 1] else 1
                if suite >= 3 and suite * taille >= max(20, 0.6 * len(liste)):
                    return True
    return False


def raisons_de_reprise(whisper, qwen, indices, reglages):
    """Raisons d'envoyer un extrait à la séparation de voix (liste vide = extrait fiable)."""
    a, b = mots(whisper["text"]), mots(qwen["text"])
    similarite = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0
    seuils = reglages["diagnostics"]
    seuil_yamnet = reglages["yamnet"]["evidence_threshold"]
    groupes = indices["yamnet"]["groups"]
    parole = indices["silero"]["speech_frame_fraction"]

    raisons = []
    if whisper["status"] != "completed" or qwen["status"] != "completed":
        raisons.append("asr_operational_failure")
    if bool(a) != bool(b):
        raisons.append("one_sided_empty")
    if a and b and similarite < seuils["text_similarity_threshold"]:
        raisons.append("text_disagreement")
    if a and b and whisper.get("language") != qwen.get("language"):
        raisons.append("language_disagreement")
    for alerte in ["token_limit", "long_repetition_candidate", "unexpected_response_format", "unknown_language_label"]:
        if alerte in whisper.get("flags", []) or alerte in qwen.get("flags", []):
            raisons.append(alerte)
    if (a or b) and groupes["music"]["peak"] >= seuil_yamnet:
        raisons.append("music_with_model_text")
    if (a or b) and parole < seuils["low_speech_fraction"] and groupes["speech"]["peak"] < seuils["low_yamnet_speech_peak"]:
        raisons.append("low_speech_evidence_with_text")
    if not a and not b and (parole > seuils["empty_speech_fraction"] or groupes["speech"]["peak"] >= seuil_yamnet):
        raisons.append("both_empty_with_speech_evidence")
    return sorted(set(raisons))
