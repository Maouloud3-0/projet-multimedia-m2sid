"""01 - Convertir les MP3 en WAV 16 kHz mono.

C'est le format attendu par Whisper et par Demucs. Les fichiers deja
convertis sont sautes, donc on peut relancer le script sans risque.

Usage :
    python 01_convertir_audio.py
"""

import subprocess

import config
import utils


def convertir(mp3, wav):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp3),
        "-ac", "1",                      # mono
        "-ar", str(config.FREQUENCE),    # 16 kHz
        str(wav),
    ], check=True)


def main():
    utils.preparer_dossiers()
    liste = utils.extraits()
    print(f"{len(liste)} extraits a convertir")

    for numero, extrait in enumerate(liste, start=1):
        wav = config.WAV / f"{extrait['video_id']}.wav"
        if wav.exists():
            continue
        convertir(extrait["mp3"], wav)
        if numero % 250 == 0:
            print(f"  {numero}/{len(liste)}")

    print("Termine ->", config.WAV)


if __name__ == "__main__":
    main()
