"""05 - Comparer deux configurations de transcription.

On ne peut pas calculer un taux d'erreur sans transcription de reference
(c'est le role de 06_evaluer_wer.py). Ce script mesure donc ce qui est
observable sans reference :

  - la part d'extraits ou de la parole est detectee ;
  - la longueur des textes produits ;
  - la part d'extraits dont le texte change entre les deux configurations ;
  - la similarite des textes (0 = tout different, 1 = identique) ;
  - quelques exemples concrets de differences.

Usage :
    python 05_comparer_configurations.py
    python 05_comparer_configurations.py --a transcriptions_wav --b transcriptions_voix
"""

import argparse
import csv
import difflib

import config


def lire(nom):
    """Charge un CSV de resultats : video_id -> ligne."""
    chemin = config.RESULTATS / f"{nom}.csv"
    if not chemin.exists():
        raise SystemExit(f"Fichier absent : {chemin}")
    with chemin.open(encoding="utf-8") as flux:
        return {ligne["video_id"]: ligne for ligne in csv.DictReader(flux)}


def statistiques(nom, lignes):
    paroles = [ligne for ligne in lignes.values() if ligne["a_parole"] == "1"]
    textes = [ligne["texte"] for ligne in paroles if ligne["texte"]]
    moyenne_mots = sum(len(t.split()) for t in textes) / len(textes) if textes else 0
    return {
        "configuration": nom,
        "extraits": len(lignes),
        "avec_parole": len(paroles),
        "part_parole": len(paroles) / len(lignes) if lignes else 0,
        "mots_moyens": moyenne_mots,
    }


def main():
    analyseur = argparse.ArgumentParser(description="Comparaison de configurations")
    analyseur.add_argument("--a", default="transcriptions_wav")
    analyseur.add_argument("--b", default="transcriptions_voix")
    options = analyseur.parse_args()

    a = lire(options.a)
    b = lire(options.b)
    communs = sorted(set(a) & set(b))

    if not communs:
        raise SystemExit("Aucun extrait commun entre les deux fichiers.")

    differents = [identifiant for identifiant in communs
                  if a[identifiant]["texte"].strip() != b[identifiant]["texte"].strip()]
    parles = [identifiant for identifiant in differents
              if a[identifiant]["texte"].strip() and b[identifiant]["texte"].strip()]
    similarites = [
        difflib.SequenceMatcher(None, a[i]["texte"].lower(), b[i]["texte"].lower()).ratio()
        for i in parles
    ]

    # On compare exactement le meme sous-ensemble d'extraits.
    a_communs = {identifiant: a[identifiant] for identifiant in communs}
    b_communs = {identifiant: b[identifiant] for identifiant in communs}

    lignes = []
    lignes.append(f"Comparaison de {len(communs)} extraits communs")
    lignes.append("")
    lignes.append(f"{'configuration':<24}{'extraits':>10}{'parole':>10}{'part':>8}{'mots/extrait':>14}")
    for nom, dico in ((options.a, a_communs), (options.b, b_communs)):
        stats = statistiques(nom, dico)
        lignes.append(f"{stats['configuration']:<24}{stats['extraits']:>10}"
                      f"{stats['avec_parole']:>10}{stats['part_parole']:>7.1%}"
                      f"{stats['mots_moyens']:>14.1f}")
    lignes.append("")
    lignes.append(f"Extraits dont le texte change : {len(differents)} "
                  f"({len(differents) / len(communs):.1%})")

    # Desaccords sur le fait qu'il y ait, ou non, des dialogues.
    desaccords = sum(a[i]["a_parole"] != b[i]["a_parole"] for i in communs)
    lignes.append(f"Desaccords sur la presence de parole : {desaccords} "
                  f"({desaccords / len(communs):.1%})")
    if similarites:
        lignes.append(f"Similarite moyenne des textes : {sum(similarites) / len(similarites):.2f}")
    lignes.append("")
    lignes.append("Exemples de differences")
    for identifiant in parles[:3]:
        lignes.append(f"  {identifiant}")
        lignes.append(f"    {options.a} : {a[identifiant]['texte'][:130]}")
        lignes.append(f"    {options.b} : {b[identifiant]['texte'][:130]}")

    texte = "\n".join(lignes)
    print(texte)

    config.RESULTATS.mkdir(parents=True, exist_ok=True)
    sortie = config.RESULTATS / f"comparaison_{options.a}_{options.b}.txt"
    sortie.write_text(texte + "\n", encoding="utf-8")
    print("\nTermine ->", sortie)


if __name__ == "__main__":
    main()
