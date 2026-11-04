"""Compare les transcriptions et les indices acoustiques, puis classe chaque extrait :
texte accepté, pas de dialogue, ou à vérifier à la main."""
from collections import Counter

from tqdm import tqdm

from pathlib import Path

from lib.commun import RESULTATS, arguments, chemin_extrait, ecrire_jsonl, extraits, lire_json
from lib.routage import aiguiller


def main():
    parseur = arguments(__doc__)
    parseur.add_argument("--sortie", type=Path, default=RESULTATS / "aiguillage.jsonl")
    args = parseur.parse_args()
    regles = lire_json(args.config)["routing"]
    sortie = []
    for ligne in tqdm(extraits(args), desc="Aiguillage"):
        whisper = lire_json(chemin_extrait(args.work / "asr" / "whisper", ligne))
        qwen = lire_json(chemin_extrait(args.work / "asr" / "qwen", ligne))
        indices = lire_json(chemin_extrait(args.work / "evidence", ligne))
        fichier_separe = chemin_extrait(args.work / "recovery", ligne)
        separe = lire_json(fichier_separe) if fichier_separe.exists() else None

        decision = aiguiller(whisper, qwen, separe, indices, regles)
        brutes = {nom: dict(text=r["text"], language=r.get("language"), flags=r.get("flags", []))
                  for nom, r in [("whisper", whisper), ("original_qwen", qwen), ("separated_qwen", separe)]
                  if r is not None}
        sortie.append(dict(film_id=ligne["film_id"], video_id=ligne["video_id"],
                           duration_seconds=ligne["duration_seconds"],
                           route=decision["route"], reasons=decision["reasons"],
                           proposed_text=decision["proposed_text"], proposed_source=decision["proposed_source"],
                           raw_transcripts=brutes, original_pair=decision["original_pair"],
                           separated_audio=separe["separation"]["recognition_path"] if separe else None))
    ecrire_jsonl(args.sortie, sortie)
    print(dict(Counter(r["route"] for r in sortie)))


if __name__ == "__main__":
    main()
