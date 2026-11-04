"""Répartition des émotions dans la partition commune (train / dev / test).

Lit les listes de films (train.txt, dev.txt, test.txt) et les annotations movie/annotation/<film>/<extrait>.json.
Le chemin vers movie/ : variable EMOVID_DIR, sinon l'archive du cours décompressée à côté du dépôt.

python statistiques.py
"""

import json
import os
from collections import Counter
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
MOVIE = Path(os.environ.get("EMOVID_DIR", DOSSIER.parents[1] / "PROJET_SID_2026" / "PROJET_SID_2026" / "movie"))
EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
ENSEMBLES = ["train", "dev", "test"]


def main():
    films = {e: (DOSSIER / f"{e}.txt").read_text(encoding="utf-8").split() for e in ENSEMBLES}
    comptes = {}
    for ensemble, liste in films.items():
        comptes[ensemble] = Counter(json.loads(f.read_text(encoding="utf-8"))["emotion"]
                                    for film in liste for f in (MOVIE / "annotation" / film).glob("*.json"))
    comptes["total"] = sum(comptes.values(), Counter())

    print(f"{'':12s}" + "".join(f"{e:>9s}" for e in ENSEMBLES + ["total"]))
    print(f"{'films':12s}" + "".join(f"{len(films[e]):9d}" for e in ENSEMBLES) + f"{sum(map(len, films.values())):9d}")
    print(f"{'extraits':12s}" + "".join(f"{sum(comptes[e].values()):9d}" for e in ENSEMBLES + ["total"]))
    for emotion in EMOTIONS:
        print(f"{emotion:12s}" + "".join(f"{comptes[e][emotion] / sum(comptes[e].values()):9.1%}"
                                         for e in ENSEMBLES + ["total"]))
    communs = set(films["train"]) & set(films["dev"]) | set(films["train"]) & set(films["test"]) \
        | set(films["dev"]) & set(films["test"])
    print(f"\nfilms présents dans deux ensembles : {len(communs)}")


if __name__ == "__main__":
    main()
