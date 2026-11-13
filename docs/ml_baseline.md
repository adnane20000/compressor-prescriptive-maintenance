# Baseline d'anomalies MetroPT-3 : protocole et résultats mesurés

## Données et périmètre

Cette expérience utilise le CSV réel téléchargé depuis
[UCI, MetroPT-3 dataset 791](https://archive.ics.uci.edu/dataset/791/metropt+3+),
DOI 10.24432/C5VW3R, CC BY 4.0. Aucun jeu synthétique n'est utilisé pour entraîner
le modèle présenté ici. Les données synthétiques servent exclusivement aux tests.

Le fichier compte 1 516 948 lignes, du 1er février au 1er septembre 2020.
Son SHA-256 est `db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24`.
Les dates sont sans fuseau documenté. L'index d'export n'entre jamais dans le ML.

```bash
python -m metropt_maintenance.ml "data/raw/MetroPT3(AirCompressor).csv"
python -m pytest
```

Les métriques exactes, versions et paramètres de l'exécution réelle sont dans
`docs/ml_baseline_metrics.json`. Les artefacts volumineux sont locaux, ignorés
par Git, et se recréent avec la commande précédente. Aucune performance de modèle
n'a été extrapolée à un autre compresseur ou à une utilisation industrielle.

## Prétraitement et variables temporelles

Les 15 signaux documentés sont lus par blocs de 50 000 lignes. Les dates invalides,
dupliquées ou désordonnées bloquent le ML : aucun tri implicite n'est effectué.
Les nombres non finis/non convertibles deviennent absents ; un signal numérique
hors {0, 1} devient absent. Les pressions légèrement négatives ne sont pas
écrêtées arbitrairement. Aucune interpolation, aucun remplissage à rebours.

Chaque ligne ML résume les mesures de la minute `[t-1 min, t[` et porte la date
`t`. La valeur agrégée n'utilise aucune observation de la minute suivante.
L'agrégateur garde le dernier intervalle brut entre blocs et finalise les autres
lorsque le temps source avance. À EOF, le dernier intervalle non confirmé est
omis, même s'il paraît contenir les six mesures attendues à cadence nominale.

Les **43 variables** fixées avant lecture du test sont :

- moyennes des 15 signaux ; pour les signaux numériques, proportion d'échantillons actifs ;
- écarts-types intra-minute des 7 signaux analogiques, avec `ddof=0` ;
- fractions d'échantillons dans les 4 modes interprétés ;
- différence dérivée `TP3_mean - TP2_mean` ;
- pour chacun des 4 signaux demandés : valeur de la minute précédente, différence
  avec cette valeur, moyenne et écart-type des 5 minutes précédentes.

Les noms dérivés désignent des calculs, jamais des capteurs supplémentaires.
Les fenêtres temporelles sont décalées avant de calculer les statistiques des
cinq minutes précédentes ; elles ne sont jamais centrées. Toute minute absente
ouvre un nouveau segment et remet l'historique à zéro. Cinq fenêtres de contexte
doivent alors être observées avant de scorer à nouveau.

Une fenêtre exige au moins **3 mesures**, au plus **20 % de cellules manquantes**
et un historique suffisant. Ces règles sont des choix pédagogiques explicites,
pas des exigences constructeur. Une première minute partielle n'est pas scorée.
Pour une fenêtre non éligible, `is_anomaly` et le score sont absents, pas « normal ».
Les autres valeurs absentes sont imputées par les médianes d'entraînement ; les
indicateurs de manque sont eux aussi déterminés exclusivement sur l'entraînement.

Il n'y a pas de normalisation affine : Isolation Forest effectue des coupures
par variable, qui ne nécessitent pas de scaler. Le prétraitement ajusté est
`SimpleImputer`, sauvegardé avec la forêt. Le contrat des calculs temporels est
versionné `metropt-minute-causal-v1` et figure dans le manifeste du modèle.

Le traitement réduit le CSV à **252 719 minutes observées**. Ces fenêtres réduites
restent en mémoire ; il ne s'agit pas d'un apprentissage `partial_fit` en flux.
**1 776 fenêtres ne sont pas scorées**. Les **6 dernières mesures brutes** sont
omises à EOF, faute de confirmation de fermeture de leur minute.

## Séparation chronologique et absence de fuite

Les bornes ont été choisies avant le calcul des résultats du test :

| Partition | Dates des mesures brutes | Fenêtres observées | Fenêtres scorées |
| --- | --- | --- | --- |
| Entraînement | Avant le 22/02/2020 00:00 | 26 461 | 26 301 |
| Calibration | Du 22/02 au 01/03/2020, fin exclue | 9 064 | 9 031 |
| Test | À partir du 01/03/2020 | 217 194 | 215 611 |

Une fenêtre finissant exactement à une borne appartient à la partition précédente,
car ses mesures précèdent cette borne. Les retards du début de calibration/test
peuvent utiliser le passé déjà observé de la partition précédente : c'est le
contexte disponible dans un rejeu causal, sans réajuster les paramètres.

La recommandation UCI réserve le premier mois à la référence ; ici février est
partagé entre entraînement et calibration afin de sélectionner le seuil sans
utiliser le test. L'absence de panne documentée en février ne prouve pas que
cette référence est entièrement saine ou représentative des mois suivants.

L'imputation et la forêt sont ajustées seulement sur l'entraînement. Le plafond
de 50 000 fenêtres d'entraînement n'est pas atteint dans cette expérience : les
26 301 fenêtres éligibles sont toutes utilisées. Pour un historique plus grand,
le sous-échantillonnage déterministe reste limité à cette partition.

Les tests modifient les valeurs futures et vérifient que ni les variables passées,
ni les médianes d'entraînement, ni les arbres, ni le seuil ne changent. Les périodes
externes documentées sont introduites uniquement après entraînement et calibration.
Le test n'a pas servi à ajuster le seuil ou les hyperparamètres de cette expérience.

## Modèle et seuil

Isolation Forest utilise **200 arbres**, au plus **256 fenêtres par arbre**,
`random_state=42`, `n_jobs=1`. Le score exposé vaut
`-IsolationForest.score_samples(X)` : une valeur plus élevée signifie plus
inhabituel relativement à la référence apprise. Ce score n'est pas calibré
comme probabilité de panne. La documentation de
[Scikit-learn, IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
précise la convention du score natif ; son signe est inversé ici pour la lecture.

Le seuil est le **99e percentile des scores de calibration**, avec le quantile
empirique `method="higher"` et une alerte si `score > seuil` strictement. Il vise
une charge d'environ 1 % sur cette référence, pas une prévalence de panne ni un
taux de faux positifs connu. Les égalités au seuil ne déclenchent pas d'alerte.
Le seuil interne correspondant à `contamination="auto"` n'est pas utilisé.

**Seuil mesuré : 0,6590230212367194.** Aucun ajustement sur le test n'est réalisé.

| Partition | Fenêtres en alerte | Part des fenêtres scorées |
| --- | --- | --- |
| Entraînement | 273 | 1,0380 % |
| Calibration | 90 | 0,9966 % |
| Test | 33 491 | 15,5331 % |

Le taux du test est bien supérieur à la cible de calibration : cette cible n'est
pas une garantie sur une distribution future différente.

## Modes de fonctionnement et limite principale

Les règles des modes restent celles de `ingestion/modes.py`, avec un seuil
pédagogique de 1 A croisé avec `COMP` et `DV_eletric`. La minute reçoit son mode
majoritaire ; ses fractions de modes sont aussi conservées. **34 388 fenêtres**
contiennent plusieurs modes. Le démarrage n'est pas identifié séparément.

| Mode majoritaire sur le test | Fenêtres scorées | Alertes | Taux d'alertes |
| --- | --- | --- | --- |
| Arrêt probable | 113 493 | 41 | 0,0361 % |
| Marche à vide probable | 64 172 | 4 148 | 6,4639 % |
| Charge probable | 35 952 | 29 257 | 81,3779 % |
| Indéterminé | 1 994 | 45 | 2,2568 % |

La baseline globale est très sensible aux fenêtres en charge, peu représentées
dans la référence de février. Elle ne distingue pas à elle seule changement de
régime, dérive de distribution, contexte d'exploitation et anomalie physique.
Les modes indéterminés sont absents des fenêtres de référence dans cette
expérience et ne disposent pas d'une référence propre apprise.

Cette charge d'alertes empêche de présenter la baseline comme un détecteur de
pannes validé. Les métriques ne permettent pas d'attribuer sa hausse à la seule
température, aux saisons ou aux fuites : aucune cause n'a été identifiée.

## Comparaison aux périodes documentées

Les quatre périodes ci-dessous proviennent de la section **Failure Information**
de la fiche UCI et du document `Data Description_Metro.pdf` distribué dans son
archive. Elles rapportent des fuites d'air. La note de maintenance ambiguë de la
deuxième ligne n'est pas corrigée ni utilisée pour inventer une date d'intervention.

Les bornes source ont une résolution d'une minute. La comparaison inclut donc la
minute de fin, sous forme d'un intervalle `[début, fin + 1 minute[`. Elle associe
les fenêtres de test recouvrant cet intervalle. Ce rapprochement reste externe
au modèle ; aucun label sain/panne exhaustif n'est construit.

| Début documenté | Fin documentée | Fenêtres scorées | Alertes | Première alerte disponible |
| --- | --- | --- | --- | --- |
| 18/04/2020 00:00 | 18/04/2020 23:59 | 1 430 | 1 412 | 18/04/2020 00:29 |
| 29/05/2020 23:30 | 30/05/2020 06:00 | 391 | 387 | 29/05/2020 23:31 |
| 05/06/2020 10:00 | 07/06/2020 14:30 | 2 852 | 2 852 | 05/06/2020 10:01 |
| 15/07/2020 14:30 | 15/07/2020 19:00 | 265 | 233 | 15/07/2020 14:31 |

Au moins une alerte apparaît dans **4 périodes observables sur 4**. Ce résultat
est une couverture descriptive de quatre rapports, pas un rappel généralisable
ni une preuve de prédiction de fuite. Les premières alertes ci-dessus sont après
les débuts documentés : aucune anticipation n'est revendiquée.

La couverture des observations est incomplète : la troisième période attendrait
3 151 minutes au calendrier, mais seulement 2 863 fenêtres sont observées et
2 852 scorées. Les minutes absentes ne sont ni prédites ni inventées comme saines.

Le test comporte **9 615 épisodes d'alerte**, définis comme des suites de minutes
consécutives en alerte sans relier un trou de données. **28 607 alertes** sont hors
des quatre périodes, sur **210 673 fenêtres scorées non étiquetées**. Elles ne
peuvent pas être déclarées automatiquement fausses ou vraies.

Sans vérité terrain négative exhaustive et validée, **précision, F1, ROC-AUC et
taux de faux positifs ne sont pas calculés**. La fréquence d'alertes, les comptes
par mode, le recouvrement temporel et les délais après début sont disponibles.

## Graphiques, artefacts et API

Le rapport HTML autonome dans `data/local/ml_baseline/report.html` montre les
maxima de score par heure, le seuil, les bornes des partitions et les quatre
périodes UCI. Une seconde figure montre le nombre de fenêtres en alerte par jour.
L'agrégation d'affichage ne modifie pas les prédictions par minute ; le CSV complet
reste dans `predictions.csv`. Les heures sans score restent vides.

`models/isolation_forest/baseline.joblib` contient la forêt, l'imputation ajustée,
le seuil, la configuration et la version des variables. `manifest.json` conserve
les paramètres et l'empreinte du fichier. Les artefacts Joblib doivent provenir
d'une source de confiance ; les versions scikit-learn sont contrôlées au chargement.

```python
from metropt_maintenance.ml import load_model, predict_anomaly
from metropt_maintenance.ml.features import build_features

model = load_model("models/isolation_forest/baseline.joblib")
features = build_features("data/raw/MetroPT3(AirCompressor).csv")
predictions = predict_anomaly(model, features)
```

L'API ne réentraîne rien. Elle accepte les fenêtres construites avec le même
contrat et un contexte passé suffisant. Un appel avec une seule ligne brute ne
suffit pas à créer les variables temporelles. Les états de contexte de rejeu ne
sont pas stockés dans l'artefact : il faut les reconstruire à partir du passé.

`agent_alerts.jsonl` exporte uniquement les alertes du test. Chaque observation
contient les dates, le modèle, l'empreinte source, le mode interprété, le score,
le seuil et quatre moyennes de vrais signaux avec leurs unités. Elle déclare
explicitement `score_is_failure_probability=false`, `cause=null` et
`diagnosis_confirmed=false`. La fonction `prediction_to_record()` expose le même
contrat, y compris les scores absents si les données sont insuffisantes.

## Vérification et prochaine expérience

Les tests couvrent causalité, frontières de blocs, réinitialisation après trous,
imputation sur entraînement seul, seuil sur calibration seule, séparation des
partitions, sérialisation, comparaison aux périodes documentées, sorties agents,
rapports et interaction Streamlit. Tous les modules implémentés sont couverts.
Ils n'exigent aucun réseau, aucune clé API ni données réelles.

Prochaine expérience proposée : une référence ou des seuils conditionnés par le
mode, définis avant évaluation sur un nouveau jeu de test. Ne pas optimiser sur
les quatre périodes déjà examinées et présenter ensuite ce même test comme
indépendant. Cette étape n'ajoute ni RAG, ni LLM, ni diagnostic de cause, ni plan
de maintenance, ni commande physique du compresseur.
