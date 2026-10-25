"""03 - Transcrire les extraits avec Whisper.

Whisper est un modele pre-entraine : on ne l'entraine pas, on le fait tourner
sur nos extraits et on jette les sorties peu fiables, car sur des extraits
sans dialogue il invente du texte (la logique est dans transcription.py).

Un fichier JSON est ecrit par extrait. Un extrait deja traite est saute :
on peut arreter et relancer.

Usage :
    python 03_transcrire.py                 # piste voix de Demucs si elle existe
    python 03_transcrire.py --source wav    # audio brut converti (comparaison)
    python 03_transcrire.py --sans-vad      # sans detection de parole (comparaison)
    python 03_transcrire.py --limite 20     # essai sur 20 extraits
"""

import argparse
import time

import config
import transcription
import utils


def choisir_source(source):
    """Dossier contenant les fichiers a transcrire."""
    if source == "voix" and any(config.VOIX.glob("*.wav")):
        return config.VOIX
    return config.WAV


def main():
    analyseur = argparse.ArgumentParser(description="Transcription des extraits audio")
    analyseur.add_argument("--source", choices=["voix", "wav"], default="voix")
    analyseur.add_argument("--sans-vad", action="store_true",
                           help="desactive la detection de parole")
    analyseur.add_argument("--limite", type=int, default=0,
                           help="ne traiter que les N premiers extraits")
    analyseur.add_argument("--echantillon", type=int, default=0,
                           help="traiter un echantillon aleatoire de N extraits")
    options = analyseur.parse_args()

    utils.preparer_dossiers()
    dossier = choisir_source(options.source)
    avec_vad = not options.sans_vad

    # Une sortie par configuration, pour pouvoir les comparer sans les melanger.
    etiquette = dossier.name if avec_vad else dossier.name + "_sans_vad"
    dossier_sortie = config.TRANSCRIPTIONS / etiquette
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    print(f"Modele {config.CHEMIN_MODELE} | source {dossier.name} | VAD {avec_vad}")
    modele = transcription.charger_modele()

    liste = utils.extraits()
    if options.limite:
        liste = liste[:options.limite]
    if options.echantillon:
        liste = utils.echantillon(liste, options.echantillon)

    depart = time.time()
    faits = 0
    for extrait in liste:
        sortie = dossier_sortie / f"{extrait['video_id']}.json"
        if sortie.exists():
            continue

        wav = dossier / f"{extrait['video_id']}.wav"
        if not wav.exists():
            print("fichier manquant :", wav)
            continue

        resultat = {"video_id": extrait["video_id"], "film_id": extrait["film_id"]}
        resultat.update(transcription.transcrire(modele, wav, avec_vad))
        utils.ecrire_json(sortie, resultat)

        faits += 1
        if faits % 100 == 0:
            print(f"  {faits} extraits | {time.time() - depart:.0f} s")

    print(f"{faits} extraits transcrits ->", dossier_sortie)


if __name__ == "__main__":
    main()
