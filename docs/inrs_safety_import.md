# Import des guides INRS et garde-fous de prévention

Les pièces jointes `ed6109.pdf` et `ed6129.pdf` ont été copiées sans modification
dans `documents/`. Leurs SHA-256 correspondent aux PDF obtenus sur le site INRS.
La provenance est déclarée dans le registre local lié aux empreintes, pas déduite
d'une affirmation contenue dans le texte. Les PDF et le registre restent ignorés
par Git ; les références publiques et résultats ci-dessous sont versionnés.

| Fichier exact | Titre et édition | Pages physiques | Copie officielle |
| --- | --- | --- | --- |
| `ed6109.pdf` | Consignations et déconsignations, 3e édition, octobre 2020 | 35 | [INRS ED 6109](https://www.inrs.fr/dam/inrs/CataloguePapier/ED/TI-ED-6109.pdf) |
| `ed6129.pdf` | Sécurité des machines. Modes de fonctionnement protections neutralisées, 2e édition 2015, réimpression novembre 2017 | 32 | [INRS ED 6129](https://www.inrs.fr/dam/inrs/CataloguePapier/ED/TI-ED-6129.pdf) |

Empreintes réelles :

```text
ed6109.pdf  77e164b01354b66e1f6cba4735e450dc928594373647a1dba734ba0bab992574
ed6129.pdf  b418e494af71506d65983e6e9ceee1e422144364c879660ea8f89ff102e360d2
```

Le registre utilise `authority=official`, `document_type=safety_procedure`,
`applicability=general_reference`, `equipment_reference=""`. Ce type interne
permet d'identifier la documentation de prévention ; il ne signifie pas qu'une
procédure applicable au compresseur MetroPT est disponible. Aucun équipement
précis, état de consignation ou compétence d'un intervenant n'est établi par ces
guides. Ils ne sont ni une notice constructeur MetroPT ni une autorisation.

## Extraction et indexation réellement exécutées

```bash
python -m metropt_maintenance.rag index --documents documents
```

La réindexation locale a obtenu **7 PDF, 851 pages physiques, 4 337 chunks**.
**4 062 vecteurs** inchangés ont été réutilisés et **275** nouveaux chunks INRS
encodés avec le même E5 local. Aucun service d'embeddings distant n'a été appelé.
Les deux fichiers sont extractibles sans mot de passe utilisateur. Les chunks
restent limités à une page et conservent nom, page, empreinte et portée.

Quatorze pages du corpus complet n'ont pas de texte exploitable. Parmi les
nouveaux PDF : ED 6109 pages 6, 8 et 32 ; ED 6129 pages 6 et 26. Elles ne sont
pas complétées par du texte inventé ; aucun OCR n'est implémenté. Les tableaux et
schémas exigent une lecture du PDF original.

## Règles du prototype et portée des documents

`assess_safety_request()` refuse certaines formulations explicites demandant
de shunter une protection, d'ouvrir un circuit sous pression, d'intervenir sous
tension ou d'omettre la mise en sécurité. Le refus intervient avant accès à
l'index ou au modèle d'embeddings. `search_documents()` renvoie
« information insuffisante », `safety_blocked=true`, sans mode opératoire.
`intervention_authorized` reste faux pour tous les résultats.

Les questions de prévention restent recherchables. **ED 6129 étudie des modes
encadrés avec protections neutralisées**, notamment pour observer un processus.
Il ne doit pas être présenté comme une interdiction universelle de ces modes.
La politique du prototype refuse de fournir des instructions pour les réaliser ;
leur analyse réelle exige des compétences et mesures propres au site et à la
machine. Ce choix logiciel ne réécrit pas le contenu de la brochure.

Le filtre est lexical, avec des cas explicitement testés, et peut manquer des
paraphrases ou refuser excessivement certaines questions. Il ne certifie pas la
sécurité d'une intervention. L'application ne commande aucun équipement et le
module de diagnostic limite ses contrôles à des revues documentaires ou analyses
de données soumises à validation humaine.

## Essais de recherche et limites observées

Les essais réels figurent dans `docs/diagnostic_integration_metrics.json`, avec
chaque référence et similarité calculée. Exemples, pages physiques :

- définition de la consignation : ED 6129 p. 11, ED 6109 p. 13 et p. 10 ;
- énergies résiduelles : ED 6109 p. 26 ;
- conditions des modes protections neutralisées : ED 6129 p. 3, p. 14 et p. 1 ;
- demandes de shunt du pressostat et d'ouverture sous pression : refus explicite.

Les couvertures p. 1 et p. 3 remontent encore sur la troisième question : leur
titre est pertinent pour le thème, mais elles ne détaillent pas les conditions.
La page 14 fournit du contexte de prévention. Une forte similarité ne garantit
donc toujours pas une réponse complète. L'ancre « résiduel » est exigée pour
une question sur les énergies résiduelles, afin d'écarter les présentations
génériques de l'organisme. Ces essais ont servi au développement et ne constituent
pas un benchmark indépendant ni une mesure de sécurité industrielle.

Les tests hors réseau couvrent neuf demandes explicites refusées, cinq questions
de prévention non bloquées, la présence d'ancres spécifiques et la réutilisation
des vecteurs à l'ajout d'un document. Toutes les entrées de ces tests sont
synthétiques. Aucun taux de protection ou de fiabilité générale n'est annoncé.
