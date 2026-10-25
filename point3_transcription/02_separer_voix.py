"""02 - Isoler la voix de la musique et des effets sonores (Demucs).

Dans un film, la musique et les bruits couvrent souvent les dialogues.
Demucs separe chaque extrait en deux pistes : la voix et le reste.
On ne garde que la piste voix, qui sert ensuite a la transcription.

Le modele est charge une seule fois, puis on boucle sur les extraits.
Les extraits deja traites sont sautes : on peut relancer le script.

Usage :
    python 02_separer_voix.py
    python 02_separer_voix.py --limite 10      # essai sur 10 extraits
"""

import argparse
import time

import config
import separation
import utils


def main():
    analyseur = argparse.ArgumentParser(description="Separation voix / reste")
    analyseur.add_argument("--limite", type=int, default=0,
                           help="ne traiter que les N premiers extraits")
    analyseur.add_argument("--echantillon", type=int, default=0,
                           help="traiter un echantillon aleatoire de N extraits")
    options = analyseur.parse_args()

    utils.preparer_dossiers()
    if not any(config.WAV.glob("*.wav")):
        print("Aucun WAV : lancer d'abord 01_convertir_audio.py")
        return

    print(f"Chargement de htdemucs sur {config.DISPOSITIF}...")
    separateur = separation.charger_separateur()

    liste = utils.extraits()
    if options.limite:
        liste = liste[:options.limite]
    if options.echantillon:
        liste = utils.echantillon(liste, options.echantillon)

    depart = time.time()
    faits = 0
    for extrait in liste:
        entree = config.WAV / f"{extrait['video_id']}.wav"
        sortie = config.VOIX / f"{extrait['video_id']}.wav"
        if sortie.exists() or not entree.exists():
            continue

        separation.separer(separateur, entree, sortie)

        faits += 1
        if faits % 50 == 0:
            ecoule = time.time() - depart
            print(f"  {faits} extraits | {ecoule / 60:.1f} min | {ecoule / faits:.2f} s/extrait")

    print(f"{faits} pistes voix ->", config.VOIX)


if __name__ == "__main__":
    main()
