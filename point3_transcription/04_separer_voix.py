"""Pour les extraits difficiles (désaccord des modèles, musique, texte vide malgré de la parole…),
sépare la voix avec BandIt puis transcrit la voix seule avec Qwen."""
from tqdm import tqdm

from lib.asr import ServeurASR
from lib.commun import (CODE, RACINE, arguments, chemin_extrait, deja_fait, ecrire_json, extraits,
                        lire_json, modeles)
from lib.qualite import raisons_de_reprise, signaux
from lib.separation import SeparateurVoix


def main():
    args = arguments(__doc__).parse_args()
    config = lire_json(args.config)
    a_reprendre = []
    for ligne in extraits(args):
        whisper = lire_json(chemin_extrait(args.work / "asr" / "whisper", ligne))
        qwen = lire_json(chemin_extrait(args.work / "asr" / "qwen", ligne))
        indices = lire_json(chemin_extrait(args.work / "evidence", ligne))
        raisons = raisons_de_reprise(whisper, qwen, indices, config["acoustic"])
        if raisons and not deja_fait(chemin_extrait(args.work / "recovery", ligne)):
            a_reprendre.append((ligne, raisons))
    print(f"{len(a_reprendre)} extraits à reprendre")
    if not a_reprendre:
        return

    codes = lire_json(CODE / "config" / "languages.json")
    bandit = modeles("separation")["bandit"]
    with SeparateurVoix(RACINE, bandit, config["separation"], args.work / "logs") as separateur, \
            ServeurASR("qwen", modeles("asr"), config["asr"], codes, args.work / "logs") as qwen:
        for ligne, raisons in tqdm(a_reprendre, desc="Séparation"):
            source = args.data / "audio" / ligne["film_id"] / (ligne["video_id"] + ".mp3")
            separation = separateur.separer(source, args.work / "separated" / ligne["film_id"] / ligne["video_id"])
            resultat = qwen.transcrire(separation["recognition_path"])
            resultat.update(film_id=ligne["film_id"], video_id=ligne["video_id"], status="completed",
                            separation=separation, risk_reasons=raisons)
            resultat["flags"] = signaux(resultat)
            if separation["speech_to_mixture_rms_ratio"] < 0.01:
                resultat["flags"].append("very_low_estimated_speech_energy")
            ecrire_json(chemin_extrait(args.work / "recovery", ligne), resultat)


if __name__ == "__main__":
    main()
