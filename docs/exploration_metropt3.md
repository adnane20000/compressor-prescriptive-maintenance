# Exploration réelle de MetroPT-3

## Source et périmètre

Analyse de l'intégralité de `MetroPT3(AirCompressor).csv`, extrait de
[l'archive officielle UCI, dataset 791](https://archive.ics.uci.edu/dataset/791/metropt+3+).
Référence : Davari et al., DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R),
CC BY 4.0. La documentation PDF incluse dans cette archive a également été lue
pour vérifier les unités et la signification des signaux électriques.

Fichier analysé : **218 300 507 octets**, SHA-256 :

```text
db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24
```

Commande de reproduction après installation :

```bash
python -m metropt_maintenance.ingestion explore "data/raw/MetroPT3(AirCompressor).csv" --output data/local/metropt3_official --chunksize 50000 --max-bins 1200
```

Le rapport HTML autonome avec graphiques est dans
`data/local/metropt3_official/report.html`. Les valeurs complètes sont dans
`profile.json`, accompagnées des agrégats de tracé dans `chart_aggregates.csv`.
Les données brutes et ces artefacts locaux sont ignorés par Git.

## Schéma et dates observés

Le CSV contient **17 colonnes** : un index exporté sans nom, `timestamp`,
7 signaux analogiques et 8 signaux numériques. Les noms exacts sont :

```text
timestamp
TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current
COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses
```

Les cinq pressions sont documentées en bar, la température en °C et le courant
en A. Les autres signaux n'ont pas d'unité physique ; leurs valeurs observées
sont toutes dans {0, 1}. Aucune quantité de débit ou de volume n'est déduite du
signal `Caudal_impulses`. L'index est une métadonnée, jamais un capteur.

| Indicateur calculé | Valeur |
| --- | --- |
| Nombre de mesures | 1 516 948 |
| Première date | 2020-02-01 00:00:00 |
| Dernière date | 2020-09-01 03:59:50 |
| Dates absentes ou invalides | 0 |
| Horodatages répétés | 0 |
| Retours chronologiques | 0 |
| Intervalle médian | 10 s |
| Intervalle moyen | 12,1412 s |
| Intervalle minimal | 8 s |
| Intervalle maximal | 172 918 s, soit 2 jours et 1 min 58 s |
| Intervalles supérieurs à 30 s | 331 |

Le CSV est ordonné chronologiquement. Sa cadence typique est proche de 0,1 Hz,
avec des intervalles variables et de grandes interruptions. Il ne faut donc pas
supposer une grille régulière de 1 seconde, ni assimiler le nombre de lignes à
une durée de fonctionnement. Le fuseau horaire n'est pas documenté ; les dates
restent sans fuseau, sans conversion ou correction de calendrier.

La fiche UCI décrit une période février–août et mentionne selon les sections
1 Hz ou 0,1 Hz. Le PDF décrit un volume supérieur à celui du CSV distribué.
Les chiffres présentés ici proviennent du fichier effectivement analysé, dont
la dernière date dépasse août. Les grands écarts temporels sont des constats :
ils ne prouvent ni perte de données, ni arrêt du train, ni panne du compresseur.

## Qualité des valeurs

Toutes les colonnes du fichier, y compris l'index, ont **0 valeur manquante**
selon les marqueurs NA reconnus par Pandas. Les 15 signaux ont **0 valeur non
convertible ou infinie**, et les 8 signaux numériques ont **0 valeur hors {0, 1}**.

Le comptage exact sur disque trouve **0 doublon des 16 champs de mesure**,
horodatage inclus et index exporté exclu. Ce comptage se fait sur les champs
textuels après reconnaissance des marqueurs NA, pas sur une équivalence
numérique approchée. Aucune ligne n'a été supprimée ou imputée.

L'absence de valeurs manquantes ne signifie pas absence de défaut physique,
de dérive capteur ou de données absentes entre deux horodatages.

## Les quatre signaux demandés

Statistiques calculées sur toutes les valeurs valides, doublons conservés :

| Signal | Unité | Minimum | Moyenne | Maximum | Écart-type (ddof=1) |
| --- | --- | --- | --- | --- | --- |
| Oil_temperature | °C | 15,4000 | 62,6442 | 89,0500 | 6,5163 |
| Motor_current | A | 0,0200 | 2,0502 | 9,2950 | 2,3021 |
| TP2 | bar | -0,0320 | 1,3678 | 10,6760 | 3,2509 |
| TP3 | bar | 0,7300 | 8,9846 | 10,3020 | 0,6391 |

Ces plages montrent des amplitudes différentes entre la pression au compresseur
et celle du panneau pneumatique. La valeur légèrement négative de TP2 reste
dans les données : sans information d'étalonnage, elle ne permet pas de déclarer
une panne. De même, 89,05 °C n'est pas automatiquement une surchauffe : aucune
limite constructeur validée n'a été fournie. Les moyennes globales mélangent
plusieurs modes et ne constituent pas des seuils d'anomalie.

Les graphiques du rapport montrent min/max/moyenne dans **1200 fenêtres** d'environ
**4 h 16 min**. Ils conservent les extrema par fenêtre mais ne permettent pas de
mesurer précisément la durée d'un pic ni de distinguer tous les cycles courts.
Il faudra examiner des fenêtres plus fines pour étudier les transitions.

## Modes de fonctionnement : hypothèses descriptives

Les descriptions UCI relient l'admission, la vanne de sortie et le courant aux
états de marche. Les règles détaillées dans le README croisent `Motor_current`,
`COMP` et `DV_eletric`, avec un seuil pédagogique de 1 A pour séparer un courant
proche de zéro d'une marche. Ce seuil n'est ni appris, ni validé par le constructeur.

| Mode interprété | Lignes | Part des lignes |
| --- | --- | --- |
| Arrêt probable | 822 113 | 54,20 % |
| Marche à vide probable | 440 971 | 29,07 % |
| Charge probable | 235 038 | 15,49 % |
| Indéterminé | 18 826 | 1,24 % |

Ces pourcentages portent sur les échantillons et **ne sont pas des taux de durée**.
Le mode indéterminé conserve les combinaisons qui ne satisfont pas les règles ;
il ne signifie pas anomalie. Le démarrage n'est pas isolé automatiquement :
un pic de courant seul manque de contexte temporel. Aucun état ne devient une
étiquette « normal », « panne » ou « défaillance ».

La documentation comporte des rapports de défaillance externes, mais le CSV est
non étiqueté. Ces périodes n'ont pas été converties en labels, ni utilisées pour
produire un diagnostic ou une performance de modèle.

## Rejeu et limites

Le générateur relit les mesures en flux, conserve leurs dates et calcule les
délais d'attente selon la vitesse demandée. Le dashboard avance manuellement
de 1, 10 ou 100 mesures. La CLI peut attendre effectivement ces délais avec
`--wait`, éventuellement avec un plafond explicite. Les écarts historiques
d'origine sont toujours conservés dans les événements.

Les tests utilisent de petits fichiers synthétiques clairement séparés de cette
analyse réelle. Ils vérifient notamment les doublons et retours temporels entre
blocs, les dates invalides, les trous temporels, le maintien des extrema et les
interactions du dashboard. Aucune donnée en temps réel, aucune détection ML,
aucune procédure technique et aucun diagnostic confirmé ne sont produits.

Prochaine étape : examiner les cycles sur de courtes périodes et valider les
règles d'interprétation avec l'encadrant avant de définir le protocole ML et
le découpage chronologique d'entraînement et d'évaluation.
