# Diagnostic préliminaire structuré

Le module `metropt_maintenance.diagnostic` relie un événement d'anomalie, des
mesures disponibles et des passages du RAG. Il propose un dossier préliminaire
à examiner, sans confirmer de cause ni définir de procédure physique. Aucun
outil autonome ni commande physique n'est ajouté par ce module. Le workflow
distinct et ses décisions locales sont décrits dans `docs/workflow_langgraph.md`.

## Contrat public

```python
from metropt_maintenance.diagnostic import generate_diagnostic

result = generate_diagnostic(
    anomaly_event=event,              # événement de prediction_to_record ou JSON équivalent
    measurements=event["observed_signals"],
    passages=retrieval,                # SearchResult, liste de Passage ou dictionnaires
    documents_directory="documents",
)
print(result.model_dump_json(indent=2))
```

Les trois entrées sont nécessaires pour appeler le fournisseur. Un événement
non scoré, sans alerte, sans mesure exploitable ou sans passage renvoie
`information_insuffisante`. Les entrées ML et mesures structurellement invalides
lèvent `ValueError`/`pydantic.ValidationError` ; l'interface les affiche comme
erreurs de saisie. Aucun capteur absent du schéma MetroPT n'est accepté.

`AnomalyEvent` conserve date, modèle, score, seuil, statut, contexte historique
ou synthétique et indicateurs. La cohérence `is_anomaly == (score > threshold)`
est vérifiée. Un score n'est pas une probabilité de panne.
Les métadonnées ML disponibles (mode interprété, mode mixte, effectif, fraction
manquante, début de fenêtre et empreinte du CSV) sont conservées sans les deviner.
`Measurement` accepte
une valeur numérique finie ou absente et une agrégation déclarée. Les unités
proviennent du dictionnaire UCI ; une unité différente est rejetée, sans conversion
implicite. Aucune mesure n'est imputée pour produire ce diagnostic. Les valeurs
reçues peuvent déjà être des moyennes par minute produites par le ML : leur
agrégation est conservée, sans les présenter comme des mesures instantanées.

Le résultat Pydantic `DiagnosticResult` contient :

| Champ | Contenu |
| --- | --- |
| `faits_observes` | Score, seuil, indicateur ML et valeurs réellement fournies, identifiants F |
| `hypotheses_de_defaillance` | Libellés documentaires, statut « non confirmée », liens vers faits et arguments |
| `arguments_documentaires` | Citations littérales et références S, sans affirmation de causalité |
| `sources` | Nom exact, chemin, SHA-256, page physique, passage, provenance et applicabilité |
| `informations_manquantes` | Éléments à compléter, issus d'un catalogue explicite |
| `controles_recommandes` | Revues documentaires et analyses de données uniquement |
| `niveau_incertitude` | « élevé » ou « très élevé », appréciation qualitative non calibrée |
| `statut` | Toujours « diagnostic préliminaire » |
| `evenement` | Contexte de l'événement ML et modèle ayant produit le score |
| `etat_generation` | Succès technique, insuffisance ou catégorie d'erreur |

`diagnostic_confirme`, `autorisation_intervention` et
`intervention_physique_declenchee` restent faux ; `validation_humaine_requise`
reste vrai. Un succès technique `ok` ne valide ni une hypothèse ni une intervention.

## Ce que le LLM peut produire

Le LLM reçoit des faits construits par le code, au plus huit passages vérifiés,
et un schéma JSON strict. `DiagnosticDraft` interdit les champs supplémentaires.
Il permet uniquement de sélectionner des citations, un libellé d'hypothèse
extrait d'une citation, des identifiants de faits/arguments et des codes connus.
Ce choix réduit la liberté rédactionnelle pour rendre les inventions détectables.
Il ne s'agit pas d'un texte de diagnostic libre.

Les faits observés ne viennent jamais de la réponse LLM. Les citations sont
comparées au passage fourni, puis restituées avec les vrais espaces du PDF.
Avant l'appel, fichier, empreinte, page et présence du passage dans le PDF sont
vérifiés avec PyMuPDF. Les attributs de provenance viennent du registre lié à
l'empreinte. Un changement de source ou une référence inexistante empêche
l'appel ou invalide la réponse.

Une hypothèse doit citer un manuel, guide de maintenance ou AMDEC déclaré
officiel et être liée à au moins une vraie mesure fournie. Une description de
dataset, une note pédagogique ou un guide de sécurité ne peut pas devenir une
cause de défaillance. Les libellés confirmant une cause ou détectés comme des
instructions d'intervention sont refusés. Un extrait général conserve une
applicabilité à l'équipement « non établie ». Le corpus actuel ne contient pas
de notice identifiée du compresseur MetroPT-3.

Les contrôles sont choisis parmi six revues : cohérence des données, contexte
de fonctionnement, historique, demande de notice, examen de sécurité par une
personne compétente et comparaison de fenêtres historiques. Le LLM ne peut
ajouter aucune opération physique. Identification de l'équipement, état de
sécurité et historique sont systématiquement signalés comme non établis.

Les PDF et les mesures sont des données non fiables. Le prompt système le
rappelle ; aucun outil, script, lien ou pièce jointe n'est exécuté. Le contrôle
de citations et le catalogue restent appliqués même si le LLM ignore le prompt.
La conformité formelle et une citation exacte ne prouvent pas la pertinence
causale, l'exhaustivité ou la validité d'une norme pour cet équipement. Ces points
nécessitent une lecture et une validation humaines. Aucun taux d'hallucination,
précision diagnostique ou niveau de sécurité certifié n'est annoncé.

## Fournisseurs et erreurs

Le fournisseur par défaut est `mock`, sans réseau ni clé. Il renvoie un argument
littéral pour tester la liaison des sources et **aucune hypothèse de panne**.
Le résultat indique `simulation=true` et « SIMULATION ». Ce format simulé ne
rend pas synthétiques les entrées historiques réelles ; `contexte_donnees`
conserve leur nature.

`LLM_PROVIDER=openai_compatible` utilise `httpx` et Chat Completions :

| Variable | Rôle |
| --- | --- |
| `LLM_BASE_URL` | Base HTTPS, défaut `https://api.openai.com/v1` ; HTTP autorisé seulement en boucle locale |
| `LLM_MODEL` | Modèle à choisir, compatible JSON Schema strict et `max_completion_tokens` |
| `LLM_API_KEY` | Secret fourni dans l'environnement, jamais dans Git |
| `LLM_TIMEOUT_SECONDS` | Délai de 1 à 60 secondes, défaut 30 |

`.env.example` est indicatif ; aucun chargement `.env` automatique. Le mode API
envoie les faits et les extraits sélectionnés au fournisseur. Les réponses ne
sont pas sauvegardées dans un historique métier ; Streamlit conserve la sortie
dans la session. Aucun appel API n'a lieu au démarrage ou lors de la seule recherche
RAG. Aucun remplacement silencieux par le mock, nouvelle tentative automatique
ou redirection HTTP n'est effectué.

| État | Comportement |
| --- | --- |
| `information_insuffisante` | Entrées exploitables insuffisantes, aucun appel LLM |
| `sources_invalides` | PDF, provenance, empreinte, page ou passage non vérifiable, aucun appel LLM |
| `erreur_configuration` | Fournisseur, variables ou endpoint invalides |
| `erreur_api` | Échec réseau, expiration du délai ou statut HTTP non 200 |
| `reponse_invalide` | JSON, schéma, référence, citation ou contrôle non conforme ; réponse refusée/incomplète |

Les erreurs ne recopient ni clé API ni corps d'erreur distant. Les faits vérifiés
restent disponibles, sans hypothèses publiées en cas d'erreur. Les limites de
taille de réponse et les références sont contrôlées après l'appel.

## Vérifications réellement effectuées

Lors de la livraison de ce module, `python -m pytest` : **167 tests réussis**,
dont **36** pour le diagnostic. Les vérifications plus récentes du workflow
sont documentées dans `docs/workflow_langgraph.md`.
Les tests couvrent le mock sans réseau, le protocole HTTP avec `MockTransport`,
les erreurs 401/429/500, timeouts, connexion, configuration, JSON invalide,
refus/troncature, inventions de mesures/normes/contrôles, références falsifiées,
absence de sources, PDF illisible et génération depuis Streamlit. Les petits PDF,
événements et statuts officiels des fixtures sont explicitement synthétiques.
`python -m pip check` ne signale aucune dépendance incompatible.

Essai séparé sur données réelles : première alerte locale du test ML,
disponible le **1er mars 2020 à 04:15:00**, modèle `2435e546bca6e2323ff0`, et
quatre passages retrouvés dans les vrais PDF INRS/DOE. Le mock a produit sept
faits conservant les valeurs, quatre références vérifiées, un argument littéral
et zéro hypothèse. La sortie complète reste locale dans
`data/local/diagnostic_mock_real_inputs.json`. Le rapport sans texte PDF est
publié dans `docs/diagnostic_integration_metrics.json`.

**Aucune API LLM réelle n'a été appelée** : aucune clé n'est configurée dans cet
environnement. Le fonctionnement d'un modèle distant et la qualité de ses
sélections restent à évaluer après configuration. La prochaine validation
consiste à faire relire quelques dossiers avec leur source par l'encadrant,
avant toute orchestration ou proposition de maintenance.
