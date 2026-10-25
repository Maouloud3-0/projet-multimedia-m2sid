"""02 - Isoler la voix de la musique et des effets sonores (Demucs).

Dans un film, la musique et les bruits couvrent souvent les dialogues.
Demucs separe chaque extrait en deux pistes : "vocals" et le reste.
On ne garde que la piste voix, qui sert ensuite a la transcription.

Etape facultative (voir la comparaison A/B/C/D du README).

Usage :
    python 02_separer_voix.py
"""

import shutil
import subprocess

import config
import utils


def main():
    utils.preparer_dossiers()
    sortie = config.DATA / "demucs"

    commande = [
        "demucs",
        "--two-stems", "vocals",   # separer voix / reste
        "-n", "htdemucs",          # modele pre-entraine de Meta
        "-o", str(sortie),
        str(config.WAV),
    ]
    print(" ".join(commande))
    subprocess.run(commande, check=True)

    # Demucs ecrit data/demucs/htdemucs/<extrait>/vocals.wav
    # On copie ces fichiers a plat dans data/voix/<extrait>.wav
    copies = 0
    for fichier in sorted(sortie.glob("htdemucs/*/vocals.wav")):
        shutil.copyfile(fichier, config.VOIX / f"{fichier.parent.name}.wav")
        copies += 1

    print(f"{copies} pistes voix ->", config.VOIX)


if __name__ == "__main__":
    main()
