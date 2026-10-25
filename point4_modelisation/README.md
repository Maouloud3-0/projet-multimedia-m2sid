# Point 4 — Modélisation par modalité

À venir. Trois sous-dossiers prévus : `texte/`, `audio/`, `video/`.

Principe retenu : modèles pré-entraînés de Hugging Face utilisés comme
extracteurs de vecteurs, puis un classifieur simple entraîné sur les 8 émotions.
Trois niveaux par modalité (caractéristiques simples, vecteurs pré-entraînés,
ré-entraînement léger) et évaluation au F1 macro sur l'ensemble de développement.

L'entrée du modèle texte sera le CSV produit par le
[point 3](../point3_transcription/).
