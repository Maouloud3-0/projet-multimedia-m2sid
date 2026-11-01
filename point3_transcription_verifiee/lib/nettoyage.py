"""Nettoyage léger du texte et décision « anglais / autre langue / vide ».

Le nettoyage garde les mots, la ponctuation, les interjections et les paroles de
chanson ; il ne retire que les symboles de musique et les espaces en trop.
"""
from pathlib import Path
import re
import sys
import unicodedata

from .routage import mots


class DetecteurLangue:
    """langid, installé à part dans runtime/language_dependencies (voir requirements-language.txt)."""

    def __init__(self, racine):
        sys.path.insert(0, str(Path(racine) / "runtime" / "language_dependencies"))
        from langid.langid import LanguageIdentifier, model
        self.classifieur = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    def detect(self, texte):
        langue, score = self.classifieur.classify(texte)
        return dict(language=langue, normalized_score=float(score), word_tokens=len(mots(texte)))


def normaliser(texte, regles, retirer_musique=True):
    if texte is None:
        return None, []
    valeur = unicodedata.normalize(regles["unicode_normalization"], texte)
    actions = ["unicode_nfc"] if valeur != texte else []
    if retirer_musique:
        sans_symboles = valeur.translate(str.maketrans({c: " " for c in regles["music_symbols"]}))
        if sans_symboles != valeur:
            actions.append("remove_music_symbols")
        valeur = sans_symboles
    propre = re.sub(r"\s+", " ", valeur).strip()
    if propre != valeur:
        actions.append("normalize_whitespace")
    if not mots(propre):
        if propre:
            actions.append("symbol_only_to_empty")
        propre = ""
    return propre, actions


def decider_langue(texte, source, transcriptions, detecte, regles):
    """Combine le détecteur de langue et la langue annoncée par Whisper et Qwen.

    Les deux sorties de Qwen (audio d'origine et voix séparée) viennent du même
    modèle : elles ne comptent que pour un seul avis.
    """
    annoncee = transcriptions.get(source, {}).get("language")
    cle_qwen = "separated_qwen" if source == "separated_qwen" else "original_qwen"
    avis = []
    for cle in ("whisper", cle_qwen):
        t = transcriptions.get(cle, {})
        if mots(t.get("text", "")) and t.get("language") and t.get("language") != "unknown":
            avis.append(t["language"])
    langue, score = detecte["language"], detecte["normalized_score"]
    assez_long = len(mots(texte)) >= regles["detector_minimum_word_tokens"]
    sur = score >= regles["detector_score_threshold"]
    lettres = [c for c in texte if c.isalpha()]
    non_latin = sum("LATIN" not in unicodedata.name(c, "") for c in lettres) / len(lettres) if lettres else 0
    remarques = []
    if annoncee and langue != annoncee:
        remarques.append("detector_and_selected_metadata_disagree")
    if not assez_long:
        remarques.append("short_text_language_evidence")

    def decider(statut, raison):
        return dict(status=statut, language="en" if statut == "english" else langue, reason=raison,
                    selected_asr_language=annoncee, asr_family_languages=avis, detector=detecte,
                    non_latin_letter_fraction=non_latin, diagnostics=remarques)

    if lettres and non_latin > 0.5:
        return decider("non_english", "non_latin_script_text")
    if langue == "en" and sur and assez_long:
        return decider("english", "english_text_detector_supported")
    if len(avis) == 2 and all(a != "en" for a in avis):
        return decider("non_english", "both_asr_families_non_english")
    if langue != "en" and sur and ((annoncee and annoncee != "en") or (assez_long and langue in avis)):
        return decider("non_english", "non_english_asr_and_text_evidence")
    # Phrase courte : le détecteur se trompe souvent, on suit la langue annoncée par l'ASR.
    if annoncee == "en":
        return decider("english", "selected_asr_english_short_or_conflicting_detector")
    if source == "human_edit" and (avis.count("en") == 2 or ("en" in avis and langue == "en")):
        return decider("english", "human_edit_with_english_asr_context")
    return decider("language_uncertain", "insufficient_language_evidence")


def nettoyer(texte_brut, source, transcriptions, detecteur, regles):
    propre, actions = normaliser(texte_brut, regles)
    if not propre:
        return dict(raw_text=texte_brut, clean_text=propre, asr_text="", asr_text_available=False,
                    text_status="unknown" if propre is None else "empty",
                    exclusion_reason="unknown_selected_text" if propre is None else "empty_selected_text",
                    cleaning_actions=actions, language_evidence=None)
    langue = decider_langue(propre, source, transcriptions, detecteur.detect(propre), regles)
    anglais = langue["status"] == "english"
    return dict(raw_text=texte_brut, clean_text=propre, asr_text=propre if anglais else "",
                asr_text_available=anglais, text_status=langue["status"],
                exclusion_reason=None if anglais else langue["reason"],
                cleaning_actions=actions, language_evidence=langue)
