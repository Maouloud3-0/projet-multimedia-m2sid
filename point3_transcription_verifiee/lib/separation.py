"""Séparation de la voix avec BandIt pour les extraits où la musique gêne la transcription.

Le modèle travaille sur des fenêtres de 8 s (pas de 4 s) à 48 kHz. Les fenêtres sont
recollées avec un fondu linéaire. Le réseau tourne dans serveur_bandit.py (GPU, DirectML) ;
les transformées de Fourier se font ici, sur le processeur.
"""
import http.client
import io
import math
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import en_mono


def debuts_fenetres(longueur, taille, pas):
    nombre = max(1, math.ceil((longueur - taille) / pas) + 1)
    return list(range(0, nombre * pas, pas))


def fondu(debut, longueur, total, pas):
    """Poids de recollage : montée au début et descente à la fin quand une autre fenêtre chevauche."""
    poids = np.ones(longueur, dtype=np.float64)
    duree = min(pas, longueur)
    if debut > 0:
        poids[:duree] *= np.linspace(1 / duree, 1, duree)
    if debut + longueur < total:
        poids[-duree:] *= np.linspace(1, 1 / duree, duree)
    return poids


class SeparateurVoix:
    def __init__(self, racine, modele, reglages, dossier_logs):
        self.racine, self.modele, self.reglages = Path(racine), Path(modele), reglages
        self.dossier_logs = Path(dossier_logs)
        self.processus = self.journal = None

    def __enter__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.dossier_logs.mkdir(parents=True, exist_ok=True)
        self.journal = (self.dossier_logs / "bandit.log").open("wb")
        python_dml = self.racine / ".venv-dml" / "Scripts" / "python.exe"
        serveur = Path(__file__).with_name("serveur_bandit.py")
        self.processus = subprocess.Popen(
            [str(python_dml), str(serveur), "--model", str(self.modele), "--port", str(self.port)],
            cwd=self.racine, stdout=self.journal, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        fin = time.time() + 180
        while time.time() < fin:
            if self.processus.poll() is not None:
                self.__exit__()
                raise RuntimeError("Le serveur BandIt s'est arrêté, voir bandit.log")
            try:
                connexion = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                connexion.request("GET", "/health")
                if connexion.getresponse().status == 200:
                    print("BandIt : prêt", flush=True)
                    return self
            except OSError:
                pass
            time.sleep(0.2)
        self.__exit__()
        raise TimeoutError("Le serveur BandIt ne répond pas")

    def masque(self, spectre):
        flux = io.BytesIO()
        np.save(flux, np.asarray(spectre, dtype=np.float32), allow_pickle=False)
        connexion = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        try:
            connexion.request("POST", "/infer", flux.getvalue(), {"Content-Type": "application/octet-stream"})
            reponse = connexion.getresponse()
            return np.load(io.BytesIO(reponse.read()), allow_pickle=False)
        finally:
            connexion.close()

    def separer(self, source, destination):
        import torch
        import torchaudio.functional as F
        torch.set_num_threads(2)
        audio, frequence = sf.read(source, dtype="float32", always_2d=True)
        diviseur = math.gcd(frequence, 48000)
        melange = resample_poly(en_mono(audio), 48000 // diviseur, frequence // diviseur).astype(np.float32)

        taille = int(self.reglages["chunk_seconds"] * 48000)
        pas = int(self.reglages["hop_seconds"] * 48000)
        debuts = debuts_fenetres(len(melange), taille, pas)
        somme = np.zeros(len(melange))
        poids_total = np.zeros(len(melange))
        fenetre = torch.hann_window(2048)
        stft = dict(pad=0, window=fenetre, n_fft=2048, hop_length=512, win_length=2048,
                    normalized=True, center=True, pad_mode="reflect", onesided=True)
        with torch.inference_mode():
            for debut in debuts:
                bloc = melange[debut:debut + taille]
                n = len(bloc)
                x = torch.from_numpy(np.pad(bloc, (0, taille - n)))[None, None, :]
                spectre = F.spectrogram(x, power=None, **stft)
                m = torch.from_numpy(self.masque(torch.view_as_real(spectre).numpy()))
                m = torch.view_as_complex(m.permute(0, 2, 1, 3).unsqueeze(1).contiguous())
                voix = F.inverse_spectrogram(spectre * m, length=taille, **stft)[0, 0, :n].numpy()
                poids = fondu(debut, n, len(melange), pas)
                somme[debut:debut + n] += voix * poids
                poids_total[debut:debut + n] += poids
        voix = (somme / poids_total).astype(np.float32)

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        chemin_voix = destination / "speech_48k.wav"
        sf.write(chemin_voix, voix, 48000, subtype="FLOAT")
        chemin_asr = destination / "speech_asr_16k.wav"
        sf.write(chemin_asr, resample_poly(voix, 1, 3).astype(np.float32), 16000, subtype="PCM_16")
        niveau_melange = float(np.sqrt(np.mean(melange.astype(np.float64) ** 2)))
        niveau_voix = float(np.sqrt(np.mean(voix.astype(np.float64) ** 2)))
        return dict(speech_path=str(chemin_voix), recognition_path=str(chemin_asr),
                    audio_seconds=len(voix) / 48000, window_count=len(debuts),
                    speech_to_mixture_rms_ratio=niveau_voix / max(niveau_melange, 1e-12))

    def __exit__(self, *_):
        if self.processus is not None and self.processus.poll() is None:
            self.processus.terminate()
            try:
                self.processus.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.processus.kill()
        if self.journal is not None:
            self.journal.close()
            self.journal = None
