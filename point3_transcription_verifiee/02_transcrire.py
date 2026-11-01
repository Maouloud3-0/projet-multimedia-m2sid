"""Transcrit chaque extrait complet avec Whisper large-v3 puis Qwen3-ASR (GPU).

Les deux modèles sont indépendants : leur accord sert ensuite à repérer les
transcriptions fiables (05_aiguiller.py).
"""
from tqdm import tqdm

from lib.asr import ServeurASR
from lib.commun import CODE, arguments, chemin_extrait, deja_fait, ecrire_json, extraits, lire_json, modeles
from lib.qualite import signaux


def main():
    parseur = arguments(__doc__)
    parseur.add_argument("--modele", choices=["whisper", "qwen"], help="un seul modèle (par défaut : les deux)")
    args = parseur.parse_args()
    lignes = extraits(args)
    reglages = lire_json(args.config)["asr"]
    codes = lire_json(CODE / "config" / "languages.json")
    chemins = modeles("asr")

    for modele in [args.modele] if args.modele else ["whisper", "qwen"]:
        dossier = args.work / "asr" / modele
        restants = [l for l in lignes if not deja_fait(chemin_extrait(dossier, l))]
        if not restants:
            print(f"{modele} : déjà terminé")
            continue
        with ServeurASR(modele, chemins, reglages, codes, args.work / "logs") as serveur:
            for ligne in tqdm(restants, desc=modele):
                resultat = serveur.transcrire(chemin_extrait(args.work / "audio", ligne, ".wav"))
                resultat.update(film_id=ligne["film_id"], video_id=ligne["video_id"], status="completed")
                resultat["flags"] = signaux(resultat)
                ecrire_json(chemin_extrait(dossier, ligne), resultat)
    print("Transcription terminée.")


if __name__ == "__main__":
    main()
