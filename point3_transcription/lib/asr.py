"""Clients des deux serveurs de transcription locaux (whisper.cpp et llama.cpp, sur GPU Vulkan).

Chaque modèle tourne comme un petit serveur HTTP local ; on lui envoie le WAV
complet de l'extrait et on récupère le texte et la langue détectée.
"""
import base64
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid


def requete_json(port, methode, adresse, contenu=None, delai=240):
    connexion = http.client.HTTPConnection("127.0.0.1", port, timeout=delai)
    try:
        corps = None if contenu is None else json.dumps(contenu).encode("utf-8")
        connexion.request(methode, adresse, corps, {"Content-Type": "application/json"})
        reponse = connexion.getresponse()
        donnees = reponse.read()
        if reponse.status != 200:
            raise RuntimeError(f"Erreur HTTP {reponse.status} : {donnees[:300]!r}")
        return json.loads(donnees)
    finally:
        connexion.close()


def lire_reponse_qwen(reponse, codes):
    """Qwen répond « language English<asr_text>texte… » : on sépare la langue du texte."""
    choix = reponse["choices"][0]
    contenu = choix["message"].get("content") or ""
    debut, separateur, texte = contenu.partition("<asr_text>")
    trouve = re.fullmatch(r"\s*language\s+(.+?)\s*", debut, flags=re.I)
    nom = trouve.group(1).strip() if trouve else None
    return dict(text=texte.strip() if separateur else contenu.strip(),
                language=codes.get(nom.casefold()) if nom else None,
                language_name=nom,
                finish_reason=choix["finish_reason"],
                format_valid=bool(separateur) or not contenu.strip(),
                segments=[])


def lire_reponse_whisper(reponse, codes):
    nom = str(reponse.get("language", "")).strip()
    code = codes.get(nom.casefold(), nom.casefold() if nom.casefold() in codes.values() else None)
    segments = [dict(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip())
                for s in reponse["segments"]]
    return dict(text=" ".join(s["text"] for s in segments).strip(),
                language=code, language_name=nom,
                finish_reason=None, format_valid=True, segments=segments)


class ServeurASR:
    """Lance le serveur du modèle, attend qu'il soit prêt, puis transcrit des fichiers WAV."""

    def __init__(self, modele, chemins, reglages, codes, dossier_logs):
        self.modele, self.chemins, self.reglages, self.codes = modele, chemins, reglages, codes
        self.dossier_logs = Path(dossier_logs)
        self.processus = None
        self.journal = None

    def commande(self):
        r = self.reglages
        if self.modele == "whisper":
            w = r["whisper"]
            return [str(self.chemins["whisper_server"]), "-m", str(self.chemins["whisper_model"]),
                    "-t", str(r["threads"]), "-l", "auto",
                    "-bo", str(w["best_of"]), "-bs", str(w["beam_size"]), "-ml", "-1",
                    "--no-language-probabilities", "--host", "127.0.0.1", "--port", str(self.port)]
        q = r["qwen"]
        return [str(self.chemins["qwen_server"]), "-m", str(self.chemins["qwen_decoder"]),
                "--mmproj", str(self.chemins["qwen_encoder"]),
                "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(q["context_size"]), "-ngl", "99", "-t", str(r["threads"]),
                "-np", "1", "-n", str(q["max_tokens"]), "--temp", "0", "--seed", str(q["seed"]),
                "--no-cache-prompt", "--no-webui"]

    def __enter__(self):
        # Port libre choisi par le système.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.dossier_logs.mkdir(parents=True, exist_ok=True)
        self.journal = (self.dossier_logs / f"{self.modele}.log").open("wb")
        env = os.environ.copy()
        env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
        serveur = self.chemins[self.modele + "_server"]
        self.processus = subprocess.Popen(self.commande(), cwd=serveur.parent, env=env,
                                          stdout=self.journal, stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
        # Le chargement du modèle sur la carte graphique prend un peu de temps.
        fin = time.time() + 150
        while time.time() < fin:
            if self.processus.poll() is not None:
                self.__exit__()
                raise RuntimeError(f"Le serveur {self.modele} s'est arrêté, voir {self.journal.name}")
            try:
                if requete_json(self.port, "GET", "/health", delai=1).get("status") == "ok":
                    print(f"{self.modele} : prêt", flush=True)
                    return self
            except (OSError, http.client.HTTPException, RuntimeError, ValueError):
                pass
            time.sleep(0.2)
        self.__exit__()
        raise TimeoutError(f"Le serveur {self.modele} ne répond pas")

    def transcrire(self, wav):
        delai = self.reglages["request_timeout_seconds"]
        audio = Path(wav).read_bytes()
        if self.modele == "qwen":
            q = self.reglages["qwen"]
            message = dict(role="user", content=[dict(
                type="input_audio",
                input_audio=dict(data=base64.b64encode(audio).decode("ascii"), format="wav"))])
            contenu = dict(messages=[dict(role="system", content=""), message],
                           temperature=q["temperature"], seed=q["seed"], max_tokens=q["max_tokens"],
                           stream=False, cache_prompt=False)
            reponse = requete_json(self.port, "POST", "/v1/chat/completions", contenu, delai)
            return lire_reponse_qwen(reponse, self.codes)

        # Whisper attend un formulaire multipart avec le fichier WAV.
        w = self.reglages["whisper"]
        champs = dict(response_format="verbose_json", language="auto", translate="false",
                      beam_size=str(w["beam_size"]), best_of=str(w["best_of"]),
                      temperature=str(w["temperature"]), temperature_inc=str(w["temperature_increment"]),
                      max_len="-1", token_timestamps="true", no_language_probabilities="true",
                      entropy_thold="2.4", logprob_thold="-1.0",
                      no_speech_thold=str(w["no_speech_threshold"]),
                      suppress_non_speech="false", vad="false")
        limite = "asr" + uuid.uuid4().hex
        morceaux = [f'--{limite}\r\nContent-Disposition: form-data; name="{cle}"\r\n\r\n{valeur}\r\n'.encode()
                    for cle, valeur in champs.items()]
        morceaux += [f'--{limite}\r\nContent-Disposition: form-data; name="file"; '
                     f'filename="{Path(wav).name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
                     audio, f"\r\n--{limite}--\r\n".encode()]
        connexion = http.client.HTTPConnection("127.0.0.1", self.port, timeout=delai)
        try:
            connexion.request("POST", "/inference", b"".join(morceaux),
                              {"Content-Type": "multipart/form-data; boundary=" + limite})
            reponse = connexion.getresponse()
            donnees = reponse.read()
            if reponse.status != 200:
                raise RuntimeError(f"Erreur Whisper {reponse.status} : {donnees[:300]!r}")
            return lire_reponse_whisper(json.loads(donnees), self.codes)
        finally:
            connexion.close()

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
