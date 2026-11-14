# Interface Streamlit — guide de démonstration

L'application conserve les traitements Python existants et propose cinq vues
dans la navigation latérale. `app.py` reste le point d'entrée ; `ui.py` gère les
pages et `presentation.py` les lectures bornées et graphiques Plotly.

## Lancer

Depuis la racine, avec Python 3.11 ou 3.12 :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false
```

Ouvrir <http://localhost:8501>. Le thème commun est dans
`.streamlit/config.toml`. Aucun entraînement, téléchargement, indexation ou
appel à un LLM n'est déclenché par la simple ouverture d'une page.

Le mode **Démonstration sans API**, sélectionné par défaut, impose le fournisseur
mock pour les générations de l'interface. Une réponse simulée est signalée et
ne transforme pas les mesures historiques en données synthétiques. Le mode
**Fournisseur configuré** réutilise les variables décrites dans `.env.example` ;
il ne remplace pas silencieusement une erreur API par une simulation.

## Parcours des cinq pages

1. **Dashboard** : vérifier le chemin et l'origine déclarée dans **Source des
   mesures**. L'aperçu montre les 360 premières mesures disponibles. Initialiser
   le rejeu puis avancer par 1, 10 ou 100 mesures. Les courbes conservent les
   valeurs observées et matérialisent les trous de plus de 30 secondes. Le mode
   de fonctionnement est une interprétation descriptive, sans diagnostic.
2. **Détection d'anomalies** : consulter les artefacts de la baseline existante,
   leur provenance, le seuil et les scores. Les marqueurs indiquent des fenêtres
   contenant des alertes, sans conclure à une défaillance continue. L'historique
   est paginé et exporte uniquement les lignes affichées. L'entraînement reste
   une action explicite dans la section de préparation.
3. **Assistant RAG** : choisir les dossiers du corpus et de l'index, puis poser
   une question. Exemple vérifié : « Selon UCI, que mesure Oil_temperature et
   dans quelle unité ? ». La définition se trouve à la page physique 2 de
   `Data_Description_Metro.pdf`. Consulter le texte original, la page, le statut
   de provenance et l'applicabilité de chaque source. Une description de dataset
   n'est pas un manuel de maintenance.
4. **Diagnostic agentique** : démarrer une analyse sur une fenêtre chronologique
   du CSV, puis consulter le graphe, les faits, les hypothèses non confirmées,
   la proposition et le journal. Sur les données locales utilisées pour les
   essais, la fenêtre du 1er mars 2020 à 04:08–04:15:30 contient 45 mesures.
   La question DOE sur les effets des fuites permet d'exercer la recherche ;
   elle ne constitue pas une preuve de fuite ni une identification de cause.
   En mock, aucune hypothèse de panne n'est inventée.
5. **Gestion des interventions** : rouvrir un dossier par son identifiant et
   lire la fiche complète. Les actions **Approuver**, **Refuser** et **Demander
   une révision** exigent un identifiant de validateur, un commentaire et une
   confirmation explicite. Une proposition insuffisamment étayée ne peut pas
   être approuvée. La décision documentaire ne commande aucun équipement.

Le workflow s'interrompt réellement avant la revue. Les checkpoints et décisions
restent dans SQLite après fermeture de Streamlit ; les choix de navigation,
paramètres de rejeu et dernières recherches appartiennent à la session de
l'interface. La page de gestion permet de retrouver les dossiers persistés.
Une nouvelle session réinitialise les commandes de rejeu, sans perdre l'audit.

## Données absentes et limites

Le CSV, le modèle entraîné, les PDF, le registre de provenance, les poids E5 et
l'index Chroma restent des artefacts locaux ignorés par Git. Les pages affichent
des états vides ou une erreur contextualisée lorsqu'ils manquent. Les commandes
de préparation sont décrites dans le README. Le générateur de démonstration crée
721 mesures explicitement synthétiques, sans étiquette de panne ; il ne fournit
ni modèle entraîné, ni performance simulée, ni faux corpus documentaire.

Les traitements volumineux utilisent les modules de lecture par blocs. Le rejeu
conserve au plus 1 000 mesures, les scores sont réduits à 1 600 points avec maxima
et comptages préservés, l'historique affiche 50 alertes par page et les vues de
gestion au plus 100 dossiers ou décisions récents. Ces limites sont indiquées
dans l'interface. Un score d'anomalie n'est pas une probabilité de panne.

## Réponses documentaires

`generate_document_answer()` réutilise le fournisseur structuré existant pour
sélectionner des citations. Son entrée ne contient aucun événement ML factice
(`event=None`, `facts=[]`). Le schéma `DiagnosticDraft` valide la sélection ;
les hypothèses sont interdites sans mesures. Chaque citation doit être un
extrait littéral d'une page revérifiée dans le PDF dont l'empreinte est conservée.

La réponse fournit des passages, sans procédure libre ni certification de
suffisance. Sans passage admissible, elle indique « information insuffisante »
et n'appelle pas le fournisseur. Une source modifiée, une citation inventée ou
une réponse invalide est refusée. Les sources vérifiées restent visibles si
l'appel au fournisseur échoue. Les textes PDF demeurent des données non fiables.

La recherche reconnaît les variantes typographiques des noms de signaux :
`Oil_temperature` dans le CSV correspond à « Oil Temperature » dans le PDF UCI.
Une recherche explicite d'un signal auprès d'UCI peut retrouver sa définition
par correspondance exacte du signal et de l'éditeur vérifié ; cette base de
recherche est annoncée et ne valide aucune procédure propre à l'équipement.

## Vérifications effectuées

- `python -m pytest` : **245 tests réussis**, sans clé API ni service distant.
  Ils couvrent notamment les cinq pages, les états vides, les lectures bornées,
  les citations, les erreurs API simulées et l'interdiction d'approbation implicite.
- `python -m pip check` : aucune incompatibilité détectée.
- Chromium piloté par Playwright : navigation des cinq pages, avance du rejeu
  réel, affichage des artefacts ML existants, recherche dans le PDF UCI avec
  réponse mock, lancement d'un workflow réel et interruption avant revue.
- Affichage vérifié à 1 440 × 1 100 et 390 × 844 pixels, sans débordement
  horizontal sur mobile ni erreur JavaScript relevée pendant ce parcours.
- Application démarrée sur le port 8501 ; contrôle HTTP de santé réussi.

Aucune API LLM réelle n'a été appelée. Aucune revue n'a été soumise pour les
dossiers réels ouverts pendant ces essais ; ils restent en attente. Les
approbations des tests unitaires concernent uniquement des fixtures synthétiques.
Ces vérifications d'interface ne mesurent pas les performances générales du ML
ou du RAG. Résumé exploitable : `docs/interface_validation.json`.
