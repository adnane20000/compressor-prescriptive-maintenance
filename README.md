# MetroPT — acquisition, exploration et rejeu historique

Prototype académique de maintenance prescriptive du compresseur MetroPT-3.
Cette étape implémente l'acquisition CSV, l'exploration, un rapport avec graphiques
et un rejeu chronologique. Elle fonctionne sans clé API. Le ML, le RAG, le LLM,
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
  ml/, rag/, orchestration/,
  maintenance/, storage/        dossiers réservés, aucune implémentation
 data/raw/                      CSV locaux ignorés par Git
 data/local/                    rapports et futurs artefacts locaux
 docs/exploration_metropt3.md    résultats réels et limites de l'analyse
 tests/                         tests synthétiques et intégration Streamlit
```

`requirements.txt` contient maintenant Streamlit, Pytest, Pandas, NumPy, Plotly
et Requests. Les autres bibliothèques de la stack seront ajoutées lors de leur
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

Avant le ML : examiner des fenêtres temporelles plus courtes, discuter les
transitions et les règles de mode avec l'encadrant, puis définir un découpage
chronologique et un protocole d'évaluation. Ne pas transformer les modes de
fonctionnement ou les périodes de rapports de maintenance en fausses étiquettes.
Les règles métier complètes restent dans `AGENTS.md` : aucun contrôle physique,
aucune procédure technique inventée, aucune performance ML sans mesure et
validation humaine obligatoire pour tout futur plan de maintenance.
