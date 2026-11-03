"""Interface Label Studio et lecture des décisions de la vérification manuelle."""

CHOIX_MODELE = {"Use Whisper": "whisper", "Use original Qwen": "original_qwen", "Use separated Qwen": "separated_qwen"}
CHOIX_VIDE = {"Music only / empty text", "No dialogue / other sounds"}


def interface_xml(modele, separe, video=True):
    """Complète le gabarit config/review.xml : la vidéo et la voix séparée n'apparaissent que si elles existent."""
    bloc_video = ('<Header value="Movie clip (synchronized with original audio)" size="4"/>'
                  '<Video name="clip_video" value="$video" frameRate="$video_fps" sync="original_media" '
                  'height="240" timelineHeight="32" muted="true"/>'
                  '<Text name="video_help" value="Video plays with original audio. The separated track is '
                  'independent. Transcribe only words you hear; visual context is supplementary."/>')
    bloc_audio = ('<Header value="Separated speech (BandIt; may remove real speech)" size="4"/>'
                  '<Audio name="separated_audio" value="$separated_audio" hotkey="alt+s" height="100"/>')
    bloc_texte = ('<View className="review-card"><Header value="Qwen: separated audio" size="4"/>'
                  '<Text name="separated_meta" value="$separated_meta"/>'
                  '<View className="review-transcript"><Text name="separated_text" value="$separated_display"/></View></View>')
    return (modele
            .replace("<!-- ORIGINAL_VIDEO -->", bloc_video if video else "")
            .replace("ORIGINAL_SYNC", 'sync="original_media"' if video else "")
            .replace("<!-- SEPARATED_AUDIO -->", bloc_audio if separe else "")
            .replace("<!-- SEPARATED_TEXT -->", bloc_texte if separe else "")
            .replace("<!-- SEPARATED_CHOICE -->", '<Choice value="Use separated Qwen" hotkey="5"/>' if separe else ""))


def lire_decision(tache, annotation):
    """Transforme une annotation Label Studio en texte final : le texte d'un modèle choisi,
    le texte corrigé à la main, ou un texte vide (pas de dialogue)."""
    resultat = annotation.get("result", [])
    decision = next(r["value"]["choices"][0] for r in resultat if r.get("from_name") == "decision")

    texte, source = None, None
    if decision in CHOIX_VIDE:
        texte = ""
    elif decision in CHOIX_MODELE:
        source = CHOIX_MODELE[decision]
        texte = tache["raw_transcripts"][source]["text"]
    elif decision == "Edit transcript":
        texte = next(r["value"]["text"][0] for r in resultat if r.get("from_name") == "edited_text")
        source = "human_edit"

    notes = [t for r in resultat if r.get("from_name") == "review_notes" for t in r.get("value", {}).get("text", [])]
    return dict(review_decision=decision, final_text=texte, selected_source=source,
                human_reviewed=True, human_confirmed=decision != "Uncertain",
                review_notes=notes, annotation_id=annotation.get("id"),
                reviewer=annotation.get("completed_by"), updated_at=annotation.get("updated_at"))
