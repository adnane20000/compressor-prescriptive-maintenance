# Règles du projet MetroPT — maintenance prescriptive

## Objectif et périmètre

Prototype académique réalisé par deux étudiants en ingénierie pour analyser
les données MetroPT-3 d'un compresseur et, à terme, aider à préparer des
recommandations de maintenance soumises à validation humaine.

Périmètre autorisé : socle, acquisition CSV MetroPT-3, exploration, rapport et
rejeu chronologique historique. Ne pas implémenter le ML, le RAG, le LLM,
LangGraph, l'historique métier SQLite ou un plan de maintenance avant les étapes
correspondantes. SQLite temporaire est autorisé pour vérifier les doublons sur
disque sans conserver toutes les lignes en mémoire.

## Règles métier obligatoires

- Aucun contrôle physique du compresseur, aucune commande vers un équipement.
- Ne jamais présenter une anomalie comme un diagnostic confirmé.
- Distinguer observations, hypothèses, diagnostic préliminaire et validation humaine.
- Ne jamais inventer de procédure technique. Les recommandations futures devront
  citer les documents techniques utilisés (document, page ou section).
- Signaler explicitement une information manquante ou une preuve insuffisante.
- Un futur plan de maintenance doit rester une proposition jusqu'à validation
  explicite par un humain ; cette validation ne déclenche aucune action physique.
- Ne jamais inventer de performances de modèle. Publier uniquement des mesures
  réellement calculées avec leur protocole, leurs données et leurs limites.
- Conserver un mode démonstration sans clé API ni appel à un service LLM.
  Identifier clairement toute donnée synthétique ou sortie simulée.

## Architecture et lisibilité

- Utiliser Python 3.11 ou 3.12 et une architecture modulaire simple.
- Garder `app.py` comme point d'entrée court ; placer l'interface dans
  `metropt_maintenance/ui.py` et les futurs traitements dans des modules séparés.
- `ingestion` contient l'acquisition et l'exploration. Les dossiers `ml`, `rag`,
  `orchestration`, `maintenance` et `storage` restent réservés.
- Utiliser les noms exacts du CSV officiel (notamment `DV_eletric` et `Towers`).
  L'index exporté n'est pas un capteur. Ne créer aucune étiquette de défaillance.
- Lire les fichiers volumineux par blocs, limiter les points des graphiques et
  indiquer explicitement toute limite d'analyse ou agrégation.
- Le rejeu simule des données historiques, jamais une acquisition en temps réel.
  Refuser un rejeu désordonné plutôt que trier silencieusement en mémoire.
- Écrire la documentation et les explications en français, avec des noms de
  fonctions explicites et des docstrings utiles. Expliquer les choix sans
  commenter chaque ligne évidente.
- Ne pas ajouter de dépendance ou d'abstraction avant qu'elle soit nécessaire.
- Stack cible : Pandas, NumPy, Scikit-learn, Plotly, Sentence Transformers,
  ChromaDB, PyMuPDF, LangGraph, LLM configurable par API, Streamlit et SQLite.

## Données, secrets et tests

- Ne pas versionner de clés API, de fichiers `.env`, de données brutes volumineuses,
  de modèles entraînés, de bases locales ou de documents confidentiels.
- Préserver les données brutes ; documenter les futures transformations.
- Chaque module Python implémenté doit être couvert par des tests Pytest.
  Les fichiers `__init__.py` purement déclaratifs et les dossiers réservés
  n'exigent pas de test isolé.
- Les tests courants doivent fonctionner sans clé API et sans service distant.
- Après chaque changement, exécuter `python -m pytest` depuis la racine.
- Rapporter les tests réellement exécutés et les erreurs observées ; ne jamais
  annoncer un test réussi sans l'avoir exécuté.
- Mettre à jour le README quand les commandes ou le périmètre changent.
