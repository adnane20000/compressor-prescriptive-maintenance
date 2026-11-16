# Campagne d'évaluation du prototype complet

La campagne sépare les vérifications fonctionnelles simulées, la recherche dans
les vrais PDF et l'exploitation d'artefacts ML réels. Elle ne contacte jamais une
API LLM : le mock est imposé, même si un fournisseur API est configuré dans
l'environnement. Les scénarios n'agissent sur aucun équipement.

## Reproduire

Avec l'environnement du projet installé, depuis la racine :

```bash
python -m pytest --junitxml=data/local/evaluation_tests.xml
python -m metropt_maintenance.evaluation --repeats 3 \
  --junit data/local/evaluation_tests.xml
```

Pour ajouter le volet réel, préparer le CSV MetroPT, la baseline sauvegardée,
`predictions.csv`, le corpus PDF avec son registre et l'index E5, selon le README.
Les poids E5 doivent être présents localement ; la campagne ne les télécharge pas.

```bash
python -m metropt_maintenance.evaluation --real --repeats 3 \
  --junit data/local/evaluation_tests.xml
```

Les options `--output` et `--work-directory` permettent de changer les dossiers
par défaut. Le rapport HTML autonome, les tableaux Markdown et le JSON sont écrits
sous `data/local/evaluation/report/`. Le HTML inclut Plotly pour être consultable
sans réseau. Les PDF synthétiques et bases de test restent dans des sous-dossiers
uniques de `data/local/evaluation/runs/`, séparés des décisions d'usage courant.
Une section dont les artefacts manquent reste non mesurée, avec son code d'absence.
Une incohérence de prédiction, d'index ou de référence bloque l'évaluation concernée.

## Scénarios et oracles

| Scénario | Critère fonctionnel fixé avant l'exécution de la campagne |
| --- | --- |
| Normal simulé | Clôture sans RAG ni LLM, sans validation humaine créée |
| Anomalie simulée | Réponse documentée, incertitude explicite, faits conservés |
| Suspicion de fuite | Même critère ; aucune cause confirmée ni procédure physique |
| Documentation absente | Information insuffisante ; B/C n'appellent pas le LLM |
| Documents contradictoires | Deux arguments opposés exposés et insuffisance documentaire signalée |
| Échec API | Erreur visible, sans repli silencieux ni citation générée |
| Diagnostic incertain | Incertitude élevée/très élevée et informations manquantes |
| Refus humain | Reçu SQLite de refus, sans approbation |
| Approbation humaine | Interruption préalable, soumission explicite liée à la proposition, reçu enregistré |
| Reprise | Réouverture SQLite, état conservé, aucune répétition ML/RAG/LLM, décision de révision enregistrée |

Trois répétitions de dix scénarios pour trois configurations donnent 90
exécutions. Chaque fenêtre synthétique contient 43 mesures sur sept minutes.
Le cas de suspicion contient une baisse progressive de TP3 avec courant
maintenu ; il ne valide aucune signature de fuite. Le cas incertain comporte
une température d'huile manquante. L'ordre A/B/C tourne entre les cas ; aucun résultat ML injecté n'est
présenté comme une prédiction de la baseline. Les PDF pédagogiques contradictoires
sont exclusivement des données de test et ne sont pas ajoutés au corpus réel.

A et B sont des adaptateurs d'ablation : ils réutilisent le même fournisseur mock,
la validation Pydantic et l'assemblage des citations du module diagnostic. Ils
reçoivent les mêmes mesures, sans événement ou score ML fabriqué. A reçoit zéro
source ; B appelle le vrai module de recherche. C utilise les nœuds LangGraph,
les validateurs et SQLite de production, avec un moniteur injecté pour contrôler
le scénario. Les vecteurs constants synthétiques ne testent pas la qualité E5.

Une capacité absente est notée `not_supported`. Un oracle non satisfait est noté
`failed`. Le taux de scénarios traités garde les deux dans son dénominateur ;
il mesure une couverture fonctionnelle, pas l'intelligence ou la qualité clinique.
Les échecs du benchmark ne sont pas des échecs de Pytest : les tests vérifient
également que le rapport ne masque pas ces lacunes.

## Volet réel

Les statistiques ML sont recalculées par blocs sur les prédictions figées :
partition, alertes, couverture des quatre intervalles UCI documentés. Aucun
label nouveau n'est créé. La source de ces quatre périodes a été relue dans
`Data Description_Metro.pdf`, **page physique 3**, tableau « Failure Information » ;
son empreinte est conservée dans l'inventaire RAG du rapport.
Aucun
apprentissage ni ajustement de seuil n'est relancé. Precision, Recall de défaillance
et F1 restent NA : hors intervalle documenté, le statut réel est inconnu. La
fraction de périodes couvertes n'est pas un rappel diagnostique de fuite.

Le volet documentaire réutilise les 17 questions annotées des deux fichiers
`tests/fixtures/rag_*questions.json` retenus par le programme (UCI et références
générales). Le Recall@k est une moyenne par question de la fraction des pages
annotées retrouvées ; le Hit@k exige seulement une page. Ces mesures diffèrent
lorsqu'une question comporte plusieurs pages attendues. Les annotations sont
petites, parfois alternatives et déjà utilisées pendant le développement.
L'annotation d'abstention d'une question générale sur la consignation provient
du corpus UCI initial : les sources INRS ajoutées depuis peuvent être pertinentes.
Le désaccord est conservé et commenté, sans réétiquetage après observation.

L'intégration choisit la première minute test scorée sans alerte et la première
avec alerte dans les prédictions sauvegardées. Les valeurs moyennes réelles sont
communes aux trois configurations. A/B n'utilisent pas l'événement ML ; C
réexécute le vrai prétraitement causal et la forêt sur la fenêtre historique,
puis vérifie la réouverture du checkpoint. Les dossiers réels ne reçoivent aucune
soumission humaine, même simulée. Les cas sans alerte ne sont pas des états sains
confirmés.

## Indicateurs et limites

- Fidélité littérale : citations exactes / citations émises. NA si aucune citation.
  Ni la pertinence sémantique ni la causalité ne sont prouvées par cette mesure.
- Qualité diagnostique : conservation des faits, références vérifiées et
  incertitude sont auditables ; la note clinique reste NA sans experts et vérité
  terrain. Le mock n'émet pas d'hypothèse causale.
- Validation humaine : occasions de revue, interruptions, soumissions et reçus
  sont comptés. A/B n'ont pas de mécanisme de validation, indiqué NA.
- Latence : durée locale mesurée par `perf_counter`, médiane et p95 descriptifs.
  L'initialisation synthétique est exclue ; le chargement E5 et la préparation
  commune des fenêtres réelles sont publiés séparément. Les opérations diffèrent
  entre configurations ; aucune inférence générative distante n'est mesurée.

Les résultats réellement exécutés sont dans `rapport.md` et `metrics.json`.
Le rapport HTML interactif reste un artefact local reproductible. Une étude de
LLM réel demanderait le même modèle, des réglages fixés, des répétitions, un jeu
indépendant plus large et une annotation experte en aveugle. Ce travail n'est
pas remplacé par des scores mock.
