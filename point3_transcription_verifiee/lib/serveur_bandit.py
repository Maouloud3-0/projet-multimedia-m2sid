"""Petit serveur HTTP qui exécute le modèle BandIt (ONNX) sur la carte graphique via DirectML.

Il tourne dans l'environnement séparé .venv-dml (onnxruntime-directml), lancé par
lib/separation.py. Il reçoit un spectrogramme et renvoie le masque de la voix.
"""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import time

import numpy as np
import onnxruntime as ort


def main():
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--model", required=True)
    parseur.add_argument("--port", type=int, required=True)
    args = parseur.parse_args()

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Avec la fusion de graphe activée, DirectML donnait des résultats faux sur notre carte.
    options.add_session_config_entry("ep.dml.disable_graph_fusion", "1")
    session = ort.InferenceSession(args.model, sess_options=options,
                                   providers=[("DmlExecutionProvider", {"device_id": 0})])

    class Requetes(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def repondre(self, contenu, type_contenu, entetes=None):
            self.send_response(200)
            self.send_header("Content-Type", type_contenu)
            self.send_header("Content-Length", str(len(contenu)))
            for cle, valeur in (entetes or {}).items():
                self.send_header(cle, valeur)
            self.end_headers()
            self.wfile.write(contenu)

        def do_GET(self):
            self.repondre(json.dumps({"status": "ok"}).encode(), "application/json")

        def do_POST(self):
            longueur = int(self.headers.get("Content-Length", 0))
            spectre = np.load(io.BytesIO(self.rfile.read(longueur)), allow_pickle=False)
            debut = time.perf_counter()
            masque = session.run(["speech_mask"], {"spectrum": spectre})[0]
            sortie = io.BytesIO()
            np.save(sortie, masque, allow_pickle=False)
            self.repondre(sortie.getvalue(), "application/octet-stream",
                          {"X-Inference-Seconds": str(time.perf_counter() - debut)})

    print(f"BandIt prêt sur le port {args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Requetes).serve_forever()


if __name__ == "__main__":
    main()
