"""Mesure la parole, la musique et le chant dans chaque extrait (Silero VAD + YAMNet).

Aucun extrait n'est écarté ici : ces mesures servent seulement d'indices pour l'aiguillage.
"""
import soundfile as sf
from tqdm import tqdm

from lib.acoustique import ModelesAcoustiques
from lib.commun import arguments, chemin_extrait, deja_fait, ecrire_json, extraits, lire_json, modeles


def main():
    args = arguments(__doc__).parse_args()
    lignes = extraits(args)
    dossier = args.work / "evidence"
    restants = [l for l in lignes if not deja_fait(chemin_extrait(dossier, l))]
    if not restants:
        print("Mesures acoustiques déjà terminées.")
        return
    analyse = ModelesAcoustiques(modeles("acoustique"), lire_json(args.config)["acoustic"])
    for ligne in tqdm(restants, desc="Sons"):
        audio, _ = sf.read(chemin_extrait(args.work / "audio", ligne, ".wav"), dtype="float32")
        resultat = analyse.analyser(audio)
        resultat.update(film_id=ligne["film_id"], video_id=ligne["video_id"], status="completed")
        ecrire_json(chemin_extrait(dossier, ligne), resultat)
    print(f"Mesures enregistrées pour {len(lignes)} extraits.")


if __name__ == "__main__":
    main()
