# MetroPT — anomalies ML et recherche documentaire

Prototype académique de maintenance prescriptive du compresseur MetroPT-3.
Le projet implémente l'acquisition CSV, l'exploration, les rapports avec graphiques,
le rejeu chronologique, une baseline ML Isolation Forest et la recherche documentaire
RAG sans clé API. Le LLM,
LangGraph et les propositions de maintenance restent à implémenter.

## Installer et lancer

Python **3.11 ou 3.12**. Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --browser.gatherUsageStats=false
```

Sous Windows PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --browser.gatherUsageStats=false
```

Remplacer `-3.12` par `-3.11` si nécessaire. Si l'activation est bloquée,
utiliser `.\.venv\Scripts\python.exe` à la place de `python`.
Ouvrir l'adresse affichée, généralement <http://localhost:8501>.

Pour une installation CPU sous Linux/Windows, installer PyTorch CPU **avant** les
dépendances permet d'éviter les composants CUDA inutiles :

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Sous macOS, conserver l'installation standard via `requirements.txt`.

## Acquérir les données

Source : [UCI — MetroPT-3, dataset 791](https://archive.ics.uci.edu/dataset/791/metropt+3+),
Davari et al., DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R), CC BY 4.0.

Téléchargement explicite de l'archive officielle, sans la charger en mémoire :

```bash
python -m metropt_maintenance.ingestion download --directory data/raw
```

Le script extrait uniquement `MetroPT3(AirCompressor).csv`, vérifie son schéma et
écrit un fichier `.source.json` avec l'URL, la date UTC et l'empreinte de l'archive.
Un fichier existant n'est remplacé qu'avec `--overwrite`, après validation du
nouveau téléchargement. Les fichiers temporaires sont nettoyés en cas d'échec.
Prévoir de l'espace disque pour l'archive et le CSV pendant l'extraction.

**Téléchargement manuel pris en charge :** utiliser « Download » sur UCI,
décompresser l'archive, puis placer le CSV dans `data/raw/` ou conserver son chemin
local. Aucun renommage de capteur n'est nécessaire. L'index exporté est facultatif.
L'application demande un chemin local plutôt qu'un téléversement du gros fichier
en mémoire. Aucun téléchargement n'a lieu au démarrage du dashboard.

Un CSV ayant le bon schéma n'est pas automatiquement certifié comme provenant
d'UCI : conserver sa provenance. Les données brutes restent locales et ignorées
par Git ; aucune modification ni suppression automatique de lignes.

## Explorer et générer le rapport

```bash
python -m metropt_maintenance.ingestion explore "data/raw/MetroPT3(AirCompressor).csv" --output data/local/metropt3_official
```

Pour un autre fichier, remplacer le chemin. Options utiles :

```bash
python -m metropt_maintenance.ingestion explore "data/raw/MetroPT3(AirCompressor).csv" --chunksize 10000 --max-bins 1200 --max-rows 100000 --output data/local/aperçu
```

Sans `--max-rows`, l'analyse porte sur le fichier complet. Avec cette option,
le rapport affiche **PRÉFIXE LIMITÉ** : ses résultats ne décrivent que les premières
lignes. L'empreinte complète du fichier est alors omise pour éviter de lire le
reste inutilement. `--chunksize` règle le nombre de lignes résidentes par bloc.

Trois sorties sont produites :

- `report.html` : rapport autonome avec Plotly intégré, ouvrable hors ligne ;
- `profile.json` : mesures, qualité, schéma, périmètre et règles vérifiables ;
- `chart_aggregates.csv` : min/max/moyenne/effectif par fenêtre des quatre signaux.

Le dashboard propose aussi **Analyser le CSV**, un mode limité, les graphiques,
les statistiques, les manquants et un téléchargement du rapport HTML.

### Ce que mesure le profilage

Les colonnes sont validées avant lecture. Chaque bloc est lu comme texte pour
séparer les marqueurs de valeurs manquantes des chaînes non convertibles. Les
signaux sont ensuite convertis en `float64`, les dates en `datetime64[ns]` sans
inventer de fuseau. Les valeurs infinies sont comptées comme invalides. Aucune
imputation, aucun lissage des données brutes, aucune déduplication automatique.

Les moyennes et écarts-types sont combinés progressivement à partir des blocs,
sans garder toutes les mesures en RAM. L'écart-type utilise `ddof=1`. Les doublons
sont vérifiés dans une base SQLite temporaire sur disque, supprimée à la fin :
ce n'est pas un historique métier. Prévoir plusieurs centaines de Mo temporaires
pour le CSV officiel ; leur taille dépend du contenu du fichier.

Deux indicateurs sont distincts : répétitions des 16 champs documentés, index
exporté exclu, et répétitions d'horodatages valides. La comparaison des mesures
porte sur les champs textuels après reconnaissance des marqueurs NA ; `1` et
`1.0` sont donc distincts. Les collisions de hachage n'affectent pas ce comptage,
car les clés entières sont stockées sur disque.

Les intervalles sont calculés entre lignes adjacentes, y compris entre blocs,
sans relier artificiellement les lignes séparées par une date invalide.
Un écart supérieur à 30 s est un indicateur descriptif, jamais une panne.
L'histogramme des intervalles est limité à 4096 valeurs distinctes ; en cas de
dépassement, la médiane est omise et le débordement est signalé dans le JSON.

Les graphiques `Oil_temperature`, `Motor_current`, `TP2` et `TP3` utilisent au plus
1200 fenêtres par défaut. Une deuxième passe conserve min/max/moyenne par fenêtre,
ce qui garde les extrema sans envoyer 1,5 million de points au navigateur. Les
fenêtres vides restent vides. Cette vue globale peut masquer le détail des cycles.

## Vraies colonnes et unités

Le CSV officiel contient un index exporté sans nom (`Unnamed: 0` à la lecture),
`timestamp` et les **15 signaux** suivants :

| Colonnes exactes | Nature | Unité documentée |
| --- | --- | --- |
| `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs` | Analogique | bar |
| `Oil_temperature` | Analogique | °C |
| `Motor_current` | Analogique | A |
| `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses` | Numérique, valeurs attendues 0/1 | Sans unité physique |

Conserver notamment l'orthographe `DV_eletric` et la casse de `Towers`.
`Caudal_impulses` est un signal d'impulsions, pas une mesure de débit en m³/h.
L'index n'est pas un capteur. Les noms et descriptions complets figurent dans
`ingestion/schema.py` et dans le rapport.

La documentation UCI mentionne 1 Hz et 0,1 Hz dans différentes sections ; le PDF
de l'archive mentionne aussi un nombre de lignes différent. Le code mesure les
dates et la cadence du CSV réellement analysé. Aucun fuseau n'étant documenté,
aucune conversion locale ou UTC n'est appliquée aux mesures.

## Modes interprétés

Selon les descriptions UCI, `COMP=1` correspond à l'absence d'admission et
`DV_eletric=1` à une commande de fonctionnement en charge. Les règles utilisent
aussi `Motor_current` :

| Conditions | Interprétation |
| --- | --- |
| 0 ≤ courant ≤ 1 A, COMP=1, DV_eletric=0 | Arrêt probable |
| Courant > 1 A, COMP=1, DV_eletric=0 | Marche à vide probable |
| Courant > 1 A, COMP=0, DV_eletric=1 | Charge probable |
| Valeur absente, incohérente ou autre combinaison | Indéterminé |

Le seuil de 1 A est pédagogique et configurable via `infer_modes`, sans validation
constructeur. Ces règles instantanées ne séparent pas les phases de démarrage.
Les résultats comptent des **lignes**, pas des durées : la cadence est irrégulière.
Aucun mode n'est une étiquette de défaillance ou un diagnostic confirmé.

## Rejeu chronologique

Rejeu immédiat de cinq événements, sans attente, avec leurs délais calculés :

```bash
python -m metropt_maintenance.ingestion replay "data/raw/MetroPT3(AirCompressor).csv" --limit 5 --speed 60
```

Pour un pilote temporel qui attend effectivement les délais :

```bash
python -m metropt_maintenance.ingestion replay "data/raw/MetroPT3(AirCompressor).csv" --limit 20 --speed 60 --wait --max-delay 1
```

Options `--start` et `--end` : dates inclusives sans fuseau, par exemple
`--start "2020-02-01 12:00:00"`. Le délai vaut l'écart source divisé par la vitesse.
`--max-delay` est un plafond explicite du délai d'attente, utile pour sauter les
grands écarts ; le JSON conserve l'écart d'origine et indique `gap_capped`.
Sans plafond, les grands écarts historiques sont conservés. Sans `--wait`, la
CLI n'attend pas. Le premier événement a un délai nul.

Le dashboard utilise le même générateur, en **pas à pas** : initialiser le rejeu,
puis avancer de 1, 10 ou 100 mesures. La vitesse y calcule un délai proposé,
sans attente automatique. Seules les 1000 dernières mesures sont affichées.

Une date invalide ou un retour chronologique arrête le lecteur avant émission
du bloc concerné ; aucun tri implicite en mémoire. Deux dates égales sont
conservées dans leur ordre source, avec délai nul. Le contrôle porte sur la
portion effectivement lue : analyser le fichier entier avant de conclure sur
son ordre global. `max_events` permet un arrêt anticipé. Les filtres de dates
ne dispensent pas de vérifier les blocs lus ; ils ne constituent pas un index
permettant un accès direct à la fin du CSV.

**Il s'agit d'une simulation de données historiques, jamais de temps réel.**

## Baseline Machine Learning

Après installation des nouvelles dépendances :

```bash
python -m pip install -r requirements.txt
python -m metropt_maintenance.ml "data/raw/MetroPT3(AirCompressor).csv"
```

Le dashboard propose aussi **Entraîner et évaluer Isolation Forest**. Cette
opération traite le CSV complet, indépendamment du préfixe d'exploration.

Le protocole par défaut est fixé avant l'analyse du test : entraînement sur les
mesures antérieures au 22 février 2020, calibration du 22 février au 1er mars,
test à partir du 1er mars. Les fenêtres représentent des minutes closes,
horodatées à leur fin. Le seuil est le 99e percentile des scores de calibration ;
ni le test ni les rapports de défaillance ne servent à le choisir.

Les sorties locales sont :

- `models/isolation_forest/baseline.joblib` : forêt, imputation, seuil et contrat ;
- `models/isolation_forest/manifest.json` : paramètres, variables, versions et empreinte ;
- `data/local/ml_baseline/report.html` : scores et comparaison descriptive aux périodes UCI ;
- `data/local/ml_baseline/metrics.json` : métriques mesurées et provenance ;
- `data/local/ml_baseline/predictions.csv` : toutes les fenêtres, y compris non scorées ;
- `data/local/ml_baseline/agent_alerts.jsonl` : alertes du test et observations pour futurs agents ;
- `data/local/ml_baseline/score_plot_data.csv` : maxima horaires pour affichage réduit.

Le prétraitement utilise les 15 vrais signaux, des moyennes et variations par
minute, des fractions de modes, des retards et les cinq minutes précédentes.
Les trous temporels réinitialisent l'historique. Les fenêtres sans historique,
trop pauvres ou trop incomplètes ne sont pas scorées ; les autres valeurs absentes
sont imputées avec les seules médianes d'entraînement. Les fenêtres réduites
restent en mémoire ; les 1,5 million de lignes brutes sont lues par blocs.

API réutilisable après entraînement, avec contexte historique suffisant :

```python
from metropt_maintenance.ml import load_model, predict_anomaly
from metropt_maintenance.ml.features import build_features
from metropt_maintenance.ml.workflow import prediction_to_record

model = load_model("models/isolation_forest/baseline.joblib")
features = build_features("data/raw/MetroPT3(AirCompressor).csv")
predictions = predict_anomaly(model, features)
record = prediction_to_record(predictions.iloc[-1].to_dict(), model_id=model.model_id)
```

`predict_anomaly()` attend les variables causales, pas une ligne brute isolée.
Pour une séquence courte, utiliser `aggregate_minutes()` puis `engineer_features()`
sur au moins cinq minutes précédentes et la minute à scorer, en conservant aussi
une mesure ultérieure pour finaliser le dernier intervalle. Ne charger que des
artefacts locaux de confiance ; Joblib utilise la sérialisation Python. Une
version scikit-learn différente nécessite de recréer le modèle.

**Résultat réel de cette baseline** : seuil 0,659023 ; 33 491 alertes sur 215 611
fenêtres de test scorées, soit 15,53 %. Des alertes recouvrent les quatre périodes
documentées, mais la charge est très élevée en mode charge (81,38 % des fenêtres
scorées de ce mode). Cette baseline globale reste à améliorer ; ces nombres ne
mesurent ni précision, ni probabilité de panne, ni identification de fuite.

Le protocole, les limites, les métriques par mode et la comparaison aux rapports
UCI sont détaillés dans `docs/ml_baseline.md`. Un instantané vérifiable des
résultats réels figure dans `docs/ml_baseline_metrics.json`.

## Recherche documentaire RAG

Cette étape retrouve des extraits de PDF, avec leur fichier et leur page.
Elle ne génère pas de réponse LLM et n'ajoute aucun agent autonome.

1. Déposer les PDF autorisés dans `documents/`.
2. Facultativement créer `documents/sources.json` à partir de l'exemple fourni,
   en renseignant la provenance réelle et l'empreinte de chaque fichier.
3. Préparer les poids publics du modèle d'embeddings, une seule fois avec Internet.
4. Indexer les PDF puis rechercher une question.

```bash
python -m metropt_maintenance.rag prepare-model
python -m metropt_maintenance.rag index --documents documents
python -m metropt_maintenance.rag search "Dans quelle unité est mesurée la température de l'huile ?"
```

Le modèle local multilingue est `intfloat/multilingual-e5-small`, avec une révision
fixée et des embeddings normalisés de 384 dimensions. Ses poids et caches restent
dans `data/local/embedding_models/`. Après préparation, indexation et recherche
fonctionnent sans clé API et hors ligne. Les textes des PDF ne sont pas envoyés à
un service d'embeddings. Les commandes `index`, `search` et `evaluate` acceptent
`--cache` si un autre dossier de poids a été utilisé pour `prepare-model`.

Le dashboard propose **Indexer les PDF**, un filtre officiel/pédagogique/non
vérifié et un champ **Question documentaire**. Aucun index ni modèle lourd n'est
chargé au simple démarrage. Sans index ou sans résultat pertinent, la réponse est
**« information insuffisante »**.

```python
from metropt_maintenance.rag import search_documents

result = search_documents("Que mesure TP2 et dans quelle unité ?")
print(result.message)
for hit in result.passages:
    print(hit.reference, hit.authority, hit.similarity)
    print(hit.text)
```

Les références conservent le nom exact, le chemin relatif, la page physique
(première page = 1), l'éventuel label PDF et son SHA-256. Les chunks de 900
caractères avec chevauchement ne mélangent jamais deux pages. ChromaDB persiste
localement dans `data/local/rag/`. L'index actif est remplacé après succès ; un
échec d'encodage conserve l'ancien index. Réindexer après modification du corpus
ou du registre : le moteur refuse de retourner les anciennes références.

Un document n'est pas officiel parce que son texte l'affirme. Le statut vient
d'un registre humain lié à son empreinte ; sans déclaration valide, il reste
non vérifié. « Officiel déclaré » n'est pas une certification automatique.
Une description de dataset et une note pédagogique ne valident aucune procédure
de maintenance, de sécurité ou limite constructeur.

La recherche combine similarité cosinus (seuil initial 0,80), présence d'ancres
lexicales et garde-fous de type/provenance pour les demandes opérationnelles.
Ce seuil est heuristique, sans calibration sur vos futurs manuels. La similarité
n'est pas une probabilité de réponse correcte. `passages_found` signifie passages
retrouvés, pas réponse complète certifiée ; `generated_answer` reste nul et
`sufficiency_verified` reste faux. Certains refus peuvent être trop conservateurs.
Les instructions, liens et scripts des PDF ne sont jamais exécutés.

**Corpus local étendu :** description UCI, guide NASA RCM, sourcebook DOE,
guide KAESER en français et Compressed Air Manual Atlas Copco, 9e édition.
Les quatre nouveaux ouvrages sont des références générales ; aucune notice
spécifique au compresseur MetroPT-3 n'a été identifiée. Le fichier KAESER reste
non vérifié faute de copie officielle exacte confirmée. Pour Atlas Copco, une
copie de la même édition a été obtenue sur le site de l'éditeur, la pièce jointe
dépassant la limite de transfert ; son identité binaire avec l'original n'est
pas vérifiable. Les détails figurent dans `docs/rag_corpus_import.md`.

La provenance est distincte de l'applicabilité : `general_reference` ne signifie
pas procédure validée pour MetroPT-3. Une demande opérationnelle explicitement
liée à MetroPT exige une source spécifique liée à cet équipement ; sinon le
message reste « information insuffisante », avec éventuellement du contexte cité.

Les tests ordinaires utilisent de petits PDF et un encodeur **synthétiques**,
sans téléchargement. L'évaluation initiale avec les vrais embeddings E5 sur le
seul PDF UCI a retrouvé la page attendue au premier rang pour 4 questions sur 4,
avec 3 abstentions sur 3 demandes non couvertes. Ces résultats historiques ne
sont pas extrapolés au corpus étendu.

```bash
python -m metropt_maintenance.rag evaluate tests/fixtures/rag_reference_questions.json
```

Protocole et limites : `docs/rag_retrieval.md`. Résultats mesurés :
`docs/rag_quality_metrics.json`. Les PDF, registres locaux, poids et bases Chroma
restent ignorés par Git. Les pages scannées exigent un OCR externe non implémenté.

## Organisation du code

```text
app.py                          point d'entrée Streamlit
metropt_maintenance/
  ui.py                         exploration locale et commandes de rejeu
  ingestion/
    schema.py                   dictionnaire des 15 signaux et unités
    acquisition.py              téléchargement officiel en flux
    csv_reader.py               validation, blocs typés et aperçu limité
    modes.py                    règles descriptives des modes
    exploration.py              qualité, doublons sur disque, statistiques
    report.py                   agrégats, graphiques et rapport autonome
    replay.py                   événements historiques et délais calculés
    __main__.py                 commandes download / explore / replay
  ml/
    features.py                 fenêtres et variables temporelles causales
    model.py                    entraînement, seuil, predict_anomaly, persistance
    evaluation.py               périodes documentées et comparaison descriptive
    workflow.py                 exécution et observations JSON pour agents
    report.py                   courbes des scores et rapport ML
    __main__.py                 commande de baseline complète
  rag/
    documents.py                extraction PDF et provenance déclarée
    chunking.py                 passages limités à une page et offsets
    embeddings.py               Sentence Transformers local, révision fixée
    index.py                    ChromaDB et remplacement de l'index actif
    search.py                   search_documents et abstention
    evaluation.py               qualité sur questions et pages annotées
    __main__.py                 préparation, indexation, recherche, évaluation
  orchestration/,
  maintenance/, storage/        dossiers réservés, aucune implémentation
 documents/                     PDF locaux et exemple de registre de provenance
 data/raw/                      CSV locaux ignorés par Git
 data/local/                    rapports et futurs artefacts locaux
 docs/exploration_metropt3.md    résultats réels et limites de l'analyse
 tests/                         tests synthétiques et intégration Streamlit
```

`requirements.txt` contient maintenant Streamlit, Pytest, Pandas, NumPy, Plotly,
Requests, Scikit-learn, Joblib, PyMuPDF, Sentence Transformers et ChromaDB.
Les autres bibliothèques de la stack seront ajoutées lors de leur
utilisation. SQLite temporaire utilise `sqlite3` fourni avec Python. Les bornes
de versions ne remplacent pas un verrouillage exact de l'environnement.

## Tests et résultats réels

```bash
python -m pytest
python -m pip check
```

Les tests sont indépendants du réseau, sans clé API et sans CSV industriel.
Leurs petits CSV sont **synthétiques**, avec uniquement les noms officiels. Ils
couvrent chaque module, les frontières de blocs, les données invalides, les
doublons, les agrégats, les erreurs réseau simulées, le rejeu et les interactions
Streamlit via AppTest. Le CSV réel n'est jamais traité automatiquement par Pytest.

L'analyse complète du CSV officiel a été exécutée séparément ; ses résultats
sont expliqués dans `docs/exploration_metropt3.md`. Le rapport généré est dans
`data/local/metropt3_official/report.html`.

## Suite du projet

Examiner la charge d'alertes par mode avec l'encadrant et définir une prochaine
expérience, par exemple une référence conditionnée par le mode. Préserver un
nouveau jeu de test pour cette comparaison : ne pas optimiser rétrospectivement
le seuil sur les quatre périodes déjà examinées. Ne pas transformer les modes de
fonctionnement ou les périodes de rapports de maintenance en fausses étiquettes.
Les règles métier complètes restent dans `AGENTS.md` : aucun contrôle physique,
aucune procédure technique inventée, aucune performance ML sans mesure et
validation humaine obligatoire pour tout futur plan de maintenance.
