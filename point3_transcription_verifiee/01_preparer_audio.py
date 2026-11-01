"""Liste les 3 088 extraits et convertit chaque MP3 d'origine en WAV 16 kHz mono."""
from tqdm import tqdm

from lib.audio import chemin_mp3, preparer_audio
from lib.commun import arguments, chemin_extrait, ecrire_jsonl, lire_json


def lister_extraits(donnees):
    lignes = []
    for fichier in sorted((donnees / "annotation").glob("*/*.json")):
        if fichier.name.startswith("._"):      # fichiers cachés créés par macOS
            continue
        annotation = lire_json(fichier)
        lignes.append(dict(film_id=fichier.parent.name, video_id=annotation["video_id"],
                           duration_seconds=annotation["duration"]))
    return lignes


def main():
    args = arguments(__doc__).parse_args()
    lignes = lister_extraits(args.data)
    if args.limite:
        lignes = lignes[:args.limite]
    ecrire_jsonl(args.work / "clips.jsonl", lignes)
    for ligne in tqdm(lignes, desc="Conversion"):
        cible = chemin_extrait(args.work / "audio", ligne, ".wav")
        if not cible.exists():
            preparer_audio(chemin_mp3(ligne, args.data), cible)
    print(f"{len(lignes)} extraits prêts dans {args.work / 'audio'}")


if __name__ == "__main__":
    main()
