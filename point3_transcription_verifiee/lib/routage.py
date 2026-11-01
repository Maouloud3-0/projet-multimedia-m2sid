"""Aiguillage de chaque extrait à partir des transcriptions et des indices acoustiques.

Trois sorties possibles :
- text_candidate : Whisper et Qwen sont d'accord et on entend de la parole ;
- no_dialogue_candidate : pas de parole ;
- uncertain : à vérifier à la main dans Label Studio.
L'accord entre deux modèles ne garantit pas que le texte est juste : en cas de doute,
l'extrait part en vérification.
"""
from collections import Counter
import difflib
import re
import unicodedata


def mots(texte):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", texte).casefold())


def mots_lexicaux(texte, regles):
    """Comme mots(), mais « [music] », « silence »… comptent comme un texte vide."""
    liste = mots(texte)
    if " ".join(liste) in regles["nonlexical_markers"]:
        return []
    return liste


def mots_sensibles(texte, regles):
    """Négations et nombres : une seule différence change le sens de la phrase."""
    texte = unicodedata.normalize("NFKC", texte).casefold().replace("’", "'")
    texte = re.sub(r"\bcannot\b", "can not", texte)
    texte = re.sub(r"n\'t\b", " not", texte)
    sensibles = set(regles["protected_words"])
    return Counter(m for m in mots(texte) if m in sensibles or any(c.isdigit() for c in m))


def problemes(resultat, regles):
    if resultat is None:
        return ["missing_result"]
    liste = []
    if resultat.get("status") != "completed":
        liste.append("operational_failure")
    if not resultat.get("format_valid", True):
        liste.append("unexpected_response_format")
    liste.extend(set(resultat.get("flags", [])) & set(regles["blocking_flags"]))
    return sorted(set(liste))


def trois_vides(whisper, qwen, separe, regles):
    """Les trois transcriptions existent, sont saines et ne contiennent aucun mot."""
    return all(r is not None and not problemes(r, regles)
               and isinstance(r.get("text"), str) and not mots(r["text"])
               for r in (whisper, qwen, separe))


def comparer(gauche, droite, regles):
    """Compare deux transcriptions ; « eligible » si elles peuvent être acceptées sans vérification."""
    a, b = mots_lexicaux(gauche["text"], regles), mots_lexicaux(droite["text"], regles)
    similarite = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0
    rapport = min(len(a), len(b)) / max(len(a), len(b)) if a or b else 1.0
    raisons = []
    if not a or not b:
        raisons.append("empty_member")
    if problemes(gauche, regles) or problemes(droite, regles):
        raisons.append("technical_flag")
    if not gauche.get("language") or gauche.get("language") != droite.get("language"):
        raisons.append("language_disagreement_or_unknown")
    if similarite < regles["similarity_threshold"]:
        raisons.append("text_disagreement")
    if rapport < regles["minimum_token_length_ratio"]:
        raisons.append("length_disagreement")
    # Pour les phrases très courtes, on exige un accord exact.
    if regles["short_text_requires_exact_match"] and min(len(a), len(b)) <= regles["short_text_max_tokens"] and a != b:
        raisons.append("short_text_not_exact")
    if mots_sensibles(gauche["text"], regles) != mots_sensibles(droite["text"], regles):
        raisons.append("protected_word_or_number_disagreement")
    return dict(eligible=not raisons, normalized_exact=a == b, similarity=similarite,
                token_length_ratio=rapport, reasons=raisons)


def aiguiller(whisper, qwen, separe, indices, regles, separation_obligatoire=False):
    """separe = transcription Qwen après séparation de la voix (None si l'extrait n'a pas été repris)."""
    groupes = indices["yamnet"]["groups"]
    parole = indices["silero"]["speech_frame_fraction"]
    pic_parole = groupes["speech"]["peak"]
    chant = groupes["singing"]["peak"] >= regles["singing_peak_requires_uncertain"]
    parole_confirmee = parole >= regles["supported_silero_fraction"] or pic_parole >= regles["supported_yamnet_speech_peak"]
    peu_de_parole = parole <= regles["low_silero_fraction"] and pic_parole < regles["low_yamnet_speech_peak"]

    origine = comparer(whisper, qwen, regles)
    whisper_separe = comparer(whisper, separe, regles) if separe is not None else None
    stabilite = comparer(qwen, separe, regles) if separe is not None else None
    energie = separe.get("separation", {}).get("speech_to_mixture_rms_ratio") if separe is not None else None
    seuil_energie = regles["minimum_recovered_speech_rms_ratio"]
    vide_w = not mots_lexicaux(whisper["text"], regles)
    vide_q = not mots_lexicaux(qwen["text"], regles)
    vide_s = separe is not None and not mots_lexicaux(separe["text"], regles)

    resultat = dict(route="uncertain", reasons=[], proposed_text=None, proposed_source=None, language=None,
                    agreement_kind=None, text_modality_candidate=False,
                    original_pair=origine, whisper_recovered_pair=whisper_separe,
                    qwen_recovered_stability=stabilite, independent_asr_models=2,
                    acoustic_support=parole_confirmee, low_speech_evidence=peu_de_parole,
                    singing_evidence=chant, speech_to_mixture_rms_ratio=energie,
                    recovery_needed_for_routing=not (origine["eligible"] and parole_confirmee and not chant),
                    recovery_conflict_observed=False)

    def decider(route, raisons, texte=None, source=None, langue=None, exact=None):
        resultat.update(route=route, reasons=raisons, proposed_text=texte, proposed_source=source,
                        language=langue, text_modality_candidate=route == "text_candidate",
                        agreement_kind=("exact" if exact else "near") if route == "text_candidate" else None)
        return resultat

    if regles.get("all_three_empty_is_no_dialogue", False) and trois_vides(whisper, qwen, separe, regles):
        return decider("no_dialogue_candidate", ["all_three_healthy_outputs_empty"], "", "all_three_empty_policy")
    if indices.get("status", "completed") != "completed":
        return decider("uncertain", ["acoustic_operational_failure"])

    # Cas principal : les deux modèles d'origine sont d'accord et on entend de la parole.
    # La séparation ne peut pas annuler cet accord.
    if regles["original_supported_pair_has_priority"] and origine["eligible"] and parole_confirmee and not chant:
        raisons = ["independent_original_models_agree", "speech_supported", "original_supported_pair_has_priority"]
        if separe is not None:
            confirme = (not problemes(separe, regles) and stabilite["eligible"] and whisper_separe["eligible"]
                        and energie is not None and energie >= seuil_energie)
            resultat["recovery_conflict_observed"] = not confirme
            raisons.append("recovery_corroborates" if confirme else "recovery_did_not_veto_original_pair")
        return decider("text_candidate", raisons, qwen["text"], "original_qwen", qwen["language"], origine["normalized_exact"])

    if separation_obligatoire and separe is None:
        return decider("uncertain", ["required_recovery_missing"])
    if separe is not None and problemes(separe, regles):
        return decider("uncertain", ["recovery_technical_flag", *problemes(separe, regles)])

    # Pas de dialogue : modèles vides et peu de parole détectée.
    if (peu_de_parole and vide_w and vide_q and (separe is None or vide_s)
            and not problemes(whisper, regles) and not problemes(qwen, regles)):
        return decider("no_dialogue_candidate", ["original_models_empty", "low_speech_evidence"], "", "joint_empty_evidence")
    if peu_de_parole and vide_s and energie is not None and energie < seuil_energie:
        return decider("no_dialogue_candidate",
                       ["recovery_empty", "very_low_recovered_energy", "low_speech_evidence",
                        "original_text_unsubstantiated" if not (vide_w and vide_q) else "original_models_empty"],
                       "", "separation_and_acoustic_evidence")

    if chant and (not vide_w or not vide_q or (separe is not None and not vide_s)):
        return decider("uncertain", ["singing_or_lyrics_ambiguity"])
    if not parole_confirmee:
        return decider("uncertain", ["insufficient_speech_support"])
    if origine["eligible"]:
        if separe is None or (stabilite["eligible"] and whisper_separe["eligible"]
                              and energie is not None and energie >= seuil_energie):
            return decider("text_candidate",
                           ["independent_original_models_agree", "speech_supported"]
                           + (["recovery_corroborates"] if separe is not None else []),
                           qwen["text"], "original_qwen", qwen["language"], origine["normalized_exact"])
        return decider("uncertain", ["recovery_conflicts_with_original_agreement"])

    # Qwen d'origine en échec, mais Whisper et Qwen sur la voix séparée sont d'accord.
    if whisper_separe is not None and whisper_separe["eligible"] and energie is not None and energie >= seuil_energie:
        return decider("text_candidate",
                       ["whisper_and_recovered_qwen_agree", "speech_supported", "original_qwen_not_corroborated"],
                       separe["text"], "separated_qwen", separe["language"], whisper_separe["normalized_exact"])

    raisons = ["no_supported_independent_pair"] + ["original_" + r for r in origine["reasons"]]
    if whisper_separe is not None:
        raisons += ["recovered_" + r for r in whisper_separe["reasons"]]
    return decider("uncertain", sorted(set(raisons)))
