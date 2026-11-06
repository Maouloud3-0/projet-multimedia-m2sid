# Rapport

- `../RAPPORT.md` : texte du rapport ; `../RAPPORT.pdf` : version mise en page (54 pages, 38 figures, 41 tableaux).
- `dessiner_figures.py` : redessine figures et schémas dans `figures/`.
- `construire_rapport.py` : construit le PDF (couverture, sommaire, vue d'ensemble, mise en page).
- `verifier_resultats.py` : recalcule les scores principaux depuis les probabilités et les compare au rapport.
- `regenerer.ps1` : enchaîne les trois scripts ; dépendances dans `requirements.txt`.

Non versionnés : `donnees/` (copie des fichiers `resultats/` des points 3 à 5 et des CSV d'importance des
descripteurs) et `polices/` (Constantia et Calibri, fournies avec Windows).
