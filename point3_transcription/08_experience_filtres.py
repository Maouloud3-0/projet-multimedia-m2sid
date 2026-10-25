"""08 - Comparer trois pre-traitements sur les memes extraits.

Question posee : un filtre frequentiel manuel (passe-bande 300-3400 Hz, la
bande du telephone) suffit-il a isoler la voix, ou faut-il un modele de
separation comme Demucs ?

On prend N extraits contenant des dialogues, on produit trois versions,
on transcrit les trois et on affiche les textes cote a cote.

Usage :
    python 08_experience_filtres.py
    python 08_experience_filtres.py --nombre 10
"""

import argparse
import csv
import subprocess

import config
import separation
import transcription


def extraits_avec_parole(nombre):
    """Les N premiers extraits ou Whisper a entendu des dialogues."""
    chemin = config.RESULTATS / "transcriptions_wav.csv"
    if not chemin.exists():
        raise SystemExit("Lancer d'abord : python 03_transcrire.py --source wav")

    with chemin.open(encoding="utf-8") as flux:
        lignes = [ligne for ligne in csv.DictReader(flux) if ligne["a_parole"] == "1"]
    return lignes[:nombre]


def passe_bande(entree, sortie):
    """Garde seulement la bande telephone (ffmpeg)."""
    sortie.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(entree),
        "-af", f"highpass=f={config.BANDE_BASSE},lowpass=f={config.BANDE_HAUTE}",
        str(sortie),
    ], check=True)


def main():
    analyseur = argparse.ArgumentParser(description="Experience sur les filtres")
    analyseur.add_argument("--nombre", type=int, default=20)
    options = analyseur.parse_args()

    dossier = config.DATA / "experience"
    extraits = extraits_avec_parole(options.nombre)
    print(f"{len(extraits)} extraits | preparation des versions...")

    separateur = separation.charger_separateur()
    modele = transcription.charger_modele()

    lignes = []
    for numero, extrait in enumerate(extraits, start=1):
        identifiant = extrait["video_id"]
        brut = config.WAV / f"{identifiant}.wav"
        bande = dossier / "bande" / f"{identifiant}.wav"
        voix = dossier / "voix" / f"{identifiant}.wav"

        if not bande.exists():
            passe_bande(brut, bande)
        if not voix.exists():
            separation.separer(separateur, brut, voix)

        resultats = {}
        for nom, chemin in (("brut", brut), ("bande", bande), ("voix", voix)):
            resultats[nom] = transcription.transcrire(modele, chemin)["texte"]

        lignes.append(f"{identifiant}  [{extrait['emotion']}]")
        for nom in ("brut", "bande", "voix"):
            texte = resultats[nom] or "(rien)"
            lignes.append(f"   {nom:<6}: {texte[:150]}")
        lignes.append("")
        print(f"  {numero}/{len(extraits)}")

    texte = "\n".join(lignes)
    print()
    print(texte)

    config.RESULTATS.mkdir(parents=True, exist_ok=True)
    sortie = config.RESULTATS / "experience_filtres.txt"
    sortie.write_text(texte + "\n", encoding="utf-8")
    print("Termine ->", sortie)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
