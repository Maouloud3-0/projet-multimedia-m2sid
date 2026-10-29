"""09 - Preparer l'ecoute de reference pour mesurer le WER.

Le taux d'erreur sur les mots demande une transcription faite a la main.
Ce script selectionne des extraits, ecrit un tableau a remplir et une page
HTML qui permet de les ecouter dans le navigateur puis d'exporter le CSV.

Usage :
    python 09_preparer_reference.py
    python 09_preparer_reference.py --nombre 40
"""

import argparse
import collections
import csv
import html

import config

PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Ecoute de reference - point 3</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 900px; }}
  .extrait {{ border-bottom: 1px solid #ddd; padding: 10px 0; }}
  .titre {{ font-weight: 600; font-size: 14px; }}
  audio {{ width: 100%; margin: 6px 0; }}
  input {{ width: 100%; padding: 6px; font-size: 14px; }}
  button {{ padding: 10px 16px; font-size: 15px; margin-top: 16px; }}
  textarea {{ width: 100%; height: 160px; font-family: monospace; }}
</style>
</head>
<body>
<h1>Ecoute de reference</h1>
<p>Ecoutez chaque extrait et tapez ce qui est dit. Laissez vide s'il n'y a
aucun dialogue, et cochez la case dans ce cas. A la fin, cliquez sur
"Exporter" et enregistrez le resultat dans
<code>reference/transcriptions_manuelles.csv</code>.</p>
{extraits}
<button onclick="exporter()">Exporter le CSV</button>
<textarea id="sortie" readonly></textarea>
<script>
function exporter() {{
  const lignes = ["video_id,texte,a_parole"];
  document.querySelectorAll(".extrait").forEach(function (bloc) {{
    const id = bloc.dataset.id;
    const texte = bloc.querySelector("input").value.trim().replace(/"/g, '""');
    const muet = bloc.querySelector("input[type=checkbox]").checked ? "0" : "1";
    lignes.push(id + ',"' + texte + '",' + muet);
  }});
  document.getElementById("sortie").value = lignes.join("\n");
}}
</script>
</body>
</html>
"""

BLOC = """<div class="extrait" data-id="{identifiant}">
  <div class="titre">{identifiant} - {emotion}</div>
  <audio controls preload="metadata" src="{source}"></audio>
  <input type="text" placeholder="Ce qui est dit...">
  <label><input type="checkbox"> aucun dialogue</label>
</div>"""


def choisir(lignes, nombre):
    """Echantillon equilibre entre les emotions, pris dans le developpement."""
    par_emotion = collections.defaultdict(list)
    for ligne in lignes:
        if ligne["partition"] == "dev" and ligne["a_parole"] == "1":
            par_emotion[ligne["emotion"]].append(ligne)

    choix = []
    emotions = sorted(par_emotion)
    while len(choix) < nombre and any(par_emotion.values()):
        for emotion in emotions:
            if par_emotion[emotion] and len(choix) < nombre:
                choix.append(par_emotion[emotion].pop(0))
    return choix


def main():
    analyseur = argparse.ArgumentParser(description="Preparation de l'ecoute")
    analyseur.add_argument("--nombre", type=int, default=32)
    options = analyseur.parse_args()

    chemin = config.RESULTATS / "transcriptions_wav.csv"
    if not chemin.exists():
        raise SystemExit("Lancer d'abord : python 03_transcrire.py --source wav")
    with chemin.open(encoding="utf-8") as flux:
        lignes = list(csv.DictReader(flux))

    choix = choisir(lignes, options.nombre)
    if not choix:
        raise SystemExit("Aucun extrait de developpement avec de la parole.")

    reference = config.DOSSIER / "reference"
    reference.mkdir(parents=True, exist_ok=True)

    # Tableau a remplir
    with (reference / "transcriptions_manuelles.csv").open("w", newline="", encoding="utf-8") as flux:
        ecrivain = csv.writer(flux)
        ecrivain.writerow(["video_id", "texte", "a_parole"])
        for ligne in choix:
            ecrivain.writerow([ligne["video_id"], "", ""])

    # Page d'ecoute
    blocs = []
    for ligne in choix:
        source = (config.WAV / f"{ligne['video_id']}.wav").as_uri()
        blocs.append(BLOC.format(identifiant=ligne["video_id"],
                                 emotion=html.escape(ligne["emotion"]),
                                 source=source))
    (reference / "ecoute.html").write_text(PAGE.format(extraits="\n".join(blocs)),
                                           encoding="utf-8")

    print(f"{len(choix)} extraits a ecouter")
    print("  tableau :", reference / "transcriptions_manuelles.csv")
    print("  page    :", reference / "ecoute.html")


if __name__ == "__main__":
    main()
