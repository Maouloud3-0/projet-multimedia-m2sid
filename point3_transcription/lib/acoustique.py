"""Indices acoustiques sur l'extrait complet (CPU) : Silero VAD pour la parole,
YAMNet pour la parole, la musique et le chant. Ces indices servent à l'aiguillage ;
ils ne suppriment jamais un texte à eux seuls."""
import csv
import math

import numpy as np


def intervalles_parole(probabilites, nb_echantillons, regles):
    """Intervalles de parole avec hystérésis (seuil d'entrée / de sortie) et une petite marge."""
    trame = regles["frame_samples"]
    minimum = regles["min_speech_ms"] * 16
    silence = regles["min_silence_ms"] * 16
    marge = regles["padding_ms"] * 16
    segments = []
    debut = calme = None
    for i, score in enumerate(probabilites):
        position = min(i * trame, nb_echantillons)
        if score >= regles["speech_threshold"]:
            calme = None
            if debut is None:
                debut = position
        elif debut is not None and score < regles["exit_threshold"]:
            if calme is None:
                calme = position
            if position - calme >= silence:
                if calme - debut >= minimum:
                    segments.append((debut, calme))
                debut = calme = None
    if debut is not None and nb_echantillons - debut >= minimum:
        segments.append((debut, nb_echantillons))
    fusion = []
    for debut, fin in segments:
        debut, fin = max(0, debut - marge), min(nb_echantillons, fin + marge)
        if fusion and debut <= fusion[-1][1]:
            fusion[-1][1] = max(fin, fusion[-1][1])
        else:
            fusion.append([debut, fin])
    return [dict(start=d / 16000, end=f / 16000) for d, f in fusion]


class ModelesAcoustiques:
    def __init__(self, chemins, reglages):
        import onnxruntime as ort
        from ai_edge_litert.interpreter import Interpreter
        self.reglages = reglages
        options = ort.SessionOptions()
        options.intra_op_num_threads = reglages["threads_per_model"]
        options.inter_op_num_threads = 1
        self.vad = ort.InferenceSession(str(chemins["silero"]), sess_options=options,
                                        providers=["CPUExecutionProvider"])
        self.yamnet = Interpreter(model_path=str(chemins["yamnet"]), num_threads=reglages["threads_per_model"])
        self.yamnet.allocate_tensors()
        self.entree = self.yamnet.get_input_details()[0]["index"]
        self.sortie = self.yamnet.get_output_details()[0]["index"]
        with open(chemins["class_map"], encoding="utf-8", newline="") as flux:
            self.classes = [ligne["display_name"] for ligne in csv.DictReader(flux)]
        # Groupes de classes YAMNet : parole, musique, chant, cris / rires / pleurs.
        self.groupes = {nom: [self.classes.index(c) for c in reglages["yamnet"][nom + "_classes"]]
                        for nom in ["speech", "music", "singing", "vocalization"]}

    def silero(self, audio):
        r = self.reglages["silero"]
        etat = np.zeros((2, 1, 128), dtype=np.float32)
        contexte = np.zeros((1, r["context_samples"]), dtype=np.float32)
        probabilites = []
        for debut in range(0, len(audio), r["frame_samples"]):
            morceau = audio[debut:debut + r["frame_samples"]]
            morceau = np.pad(morceau, (0, r["frame_samples"] - len(morceau)))[None, :]
            courant = np.concatenate([contexte, morceau], axis=1)
            score, etat = self.vad.run(None, {"input": courant, "state": etat,
                                              "sr": np.array(16000, dtype=np.int64)})
            probabilites.append(float(score[0, 0]))
            contexte = courant[:, -r["context_samples"]:]
        return np.asarray(probabilites, dtype=np.float32)

    def scores_yamnet(self, audio):
        r = self.reglages["yamnet"]
        fenetre, pas = r["window_samples"], r["hop_samples"]
        nombre = max(1, math.ceil((len(audio) - fenetre) / pas) + 1)
        scores, longueurs = [], []
        for debut in range(0, nombre * pas, pas):
            morceau = audio[debut:debut + fenetre]
            longueurs.append(len(morceau))
            self.yamnet.set_tensor(self.entree, np.pad(morceau, (0, fenetre - len(morceau))))
            self.yamnet.invoke()
            scores.append(self.yamnet.get_tensor(self.sortie)[0].copy())
        # La dernière fenêtre est complétée par des zéros : elle compte moins dans les moyennes.
        return np.asarray(scores, dtype=np.float32), np.asarray(longueurs, dtype=np.float64)

    def analyser(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        rs, ry = self.reglages["silero"], self.reglages["yamnet"]
        probabilites = self.silero(audio)
        durees = np.minimum(rs["frame_samples"], len(audio) - np.arange(len(probabilites)) * rs["frame_samples"])
        scores, poids = self.scores_yamnet(audio)
        seuil = ry["evidence_threshold"]
        moyennes = np.average(scores, axis=0, weights=poids)
        pics = scores.max(axis=0)

        groupes, par_fenetre = {}, {}
        for nom, indices in self.groupes.items():
            valeur = scores[:, indices].max(axis=1)
            par_fenetre[nom] = valeur
            groupes[nom] = dict(classes=[self.classes[i] for i in indices],
                                peak=float(valeur.max()),
                                mean=float(np.average(valeur, weights=poids)),
                                frame_fraction_at_threshold=float(np.average(valeur >= seuil, weights=poids)))
        intervalles = intervalles_parole(probabilites, len(audio), rs)
        parole_et_musique = (par_fenetre["speech"] >= seuil) & (par_fenetre["music"] >= seuil)
        return dict(
            silero=dict(frame_count=len(probabilites),
                        mean=float(np.average(probabilites, weights=durees)),
                        peak=float(probabilites.max()),
                        speech_frame_fraction=float(np.average(probabilites >= rs["speech_threshold"], weights=durees)),
                        candidate_speech_intervals=intervalles,
                        candidate_speech_seconds=sum(i["end"] - i["start"] for i in intervalles)),
            yamnet=dict(frame_count=len(scores), groups=groupes,
                        speech_music_cooccurrence_fraction=float(np.average(parole_et_musique, weights=poids)),
                        top_mean_classes=[dict(name=self.classes[i], mean=float(moyennes[i]), peak=float(pics[i]))
                                          for i in np.argsort(moyennes)[-10:][::-1]]))
