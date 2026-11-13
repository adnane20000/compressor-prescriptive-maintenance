# Import des ouvrages de référence

Les quatre ouvrages demandés sont disponibles dans `documents/`, avec la description
UCI déjà présente. L'index RAG actif a été reconstruit localement avec les poids E5
en cache. Les noms des PDF et leurs pages physiques sont conservés.

| Fichier | Pages physiques | Passages | Provenance déclarée |
| --- | ---: | ---: | --- |
| `Data Description_Metro.pdf` | 3 | 9 | official |
| `P-2010-FR-7-23_23-4888.pdf` | 33 | 601 | unverified |
| `The Compressed Air Manual - Ed. 9.pdf` | 148 | 571 | official |
| `compressed_air_sourcebook.pdf` | 128 | 766 | official |
| `nasa-rcmguide.pdf` | 472 | 2115 | official |

Total : **5 PDF, 784 pages lues et 4 062 passages indexés**. Neuf pages n'ont pas
fourni de texte exploitable et sont signalées ; aucun OCR n'a été inventé ou
exécuté. Les PDF originaux, le registre et la base Chroma restent locaux et
ignorés par Git. Ce rapport et les correctifs de code peuvent être versionnés.

## Acquisition et provenance

- `nasa-rcmguide.pdf` : guide NASA de septembre 2008 ; la copie fournie correspond
  octet pour octet à [la copie officielle NASA](https://www.nasa.gov/wp-content/uploads/2023/06/nasa-rcmguide.pdf).
- `compressed_air_sourcebook.pdf` : sourcebook DOE / Compressed Air Challenge,
  deuxième édition, novembre 2003. La copie fournie correspond octet pour octet
  à [la copie officielle DOE](https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/compressed_air_sourcebook.pdf).
- `P-2010-FR-7-23_23-4888.pdf` : « Technique d'air comprimé — Notions fondamentales
  et conseils », KAESER, référence P-2010FR 7/23, éditeur identifié dans le PDF.
  La copie officielle exacte n'a pas été confirmée : statut conservé **non vérifié**.
- `The Compressed Air Manual - Ed. 9.pdf` : le transfert de la pièce jointe était
  bloqué par la limite de 32 Mio de l'outil. Une [copie de la neuvième édition sur
  le site Atlas Copco](https://www.atlascopco.com/content/dam/atlas-copco/local-countries/south-africa/compressor-technique/Compressed%20Air%20Manual%209th%20Edition.pdf)
  a été obtenue : 148 pages physiques, édition 2019, ISBN 9789081535809.
  Il s'agit d'une copie de remplacement de la même édition ; son identité binaire
  avec la pièce jointe trop volumineuse **ne peut pas être vérifiée**.

Les empreintes effectives figurent dans `documents/sources.json` et dans
`docs/rag_reference_quality_metrics.json`. Les textes, liens et instructions des
PDF restent des données non fiables. Aucune instruction contenue dans un ouvrage
n'a été exécutée comme consigne du projet.

## Portée des ouvrages

Tous sont classés `general_reference`. Ce sont des sources de principes et de
méthodes générales, pas des notices identifiées du compresseur MetroPT-3.
L'authenticité d'une publication et son applicabilité à un équipement sont deux
questions distinctes. Le moteur conserve `applicability` et `equipment_reference`
dans les résultats. Une demande opérationnelle explicitement liée à MetroPT
reste « information insuffisante » en l'absence de source spécifique, tout en
pouvant présenter un contexte général cité.

## Corrections nécessaires à l'import

Le guide KAESER contient de grands espaces de mise en page. Le chevauchement des
fenêtres puis la suppression des blancs pouvaient produire deux fois exactement
le même intervalle source et donc des identifiants dupliqués dans Chroma.
Le découpage déduplique maintenant les intervalles identiques d'une même page,
sans supprimer les textes répétés à des positions différentes. Le test de
régression et l'indexation des quatre ouvrages valident ce cas. L'index actif
précédent avait été conservé lors de l'échec initial.

Les premiers essais montraient aussi des couvertures et sommaires avant les
paragraphes utiles. La recherche respecte désormais l'éditeur explicitement
nommé, écarte les sommaires à pointillés pour une question de contenu, demande
une ancre de sujet et diversifie les pages retournées. Les mots génériques et
les prépositions anglaises ne suffisent plus à prouver la pertinence. Le seuil
cosinus reste à 0,80 ; il n'a pas été abaissé pour obtenir de meilleurs chiffres.

## Validation mesurée et limites

Dix questions ont été annotées à partir des pages des ouvrages : sept demandes
documentaires et trois abstentions attendues. Ce jeu a ensuite servi à améliorer
le filtrage : **il s'agit d'une validation de développement, pas d'un test
indépendant**. Les résultats avant et après figurent dans le JSON associé.

- Page annotée au premier rang : **3/7** après correction (2/7 initialement).
- Page annotée dans les trois premiers passages : **5/7** (3/7 initialement).
- Abstention sur les demandes non couvertes : **3/3**.

Les deux pages attendues manquées dans le top 3 concernent l'analyse des
lubrifiants NASA et le principe du sécheur frigorifique Atlas Copco. Des pages
voisines ou connexes sont retournées, mais les annotations n'ont pas été étendues
rétrospectivement pour augmenter les scores. Les couvertures et doublons ne
masquent plus les résultats KAESER testés. Les annotations de pertinence ne sont
pas exhaustives et les résultats ne prouvent pas une réponse complète.

Le corpus est utilisable pour retrouver et consulter des extraits ; la précision
du classement reste perfectible sur les ouvrages longs. Aucune qualité générale,
probabilité de réponse correcte ou procédure industrielle validée n'est annoncée.

```bash
python -m metropt_maintenance.rag index
python -m metropt_maintenance.rag evaluate tests/fixtures/rag_reference_questions.json
python -m pytest
```

Les tests automatiques couvrent également la provenance, les pages, les données
hostiles inertes, la portée générale, les demandes propres à MetroPT, les sources
absentes et le cas de mise en page. Aucun LLM, agent autonome ou contrôle physique
n'est ajouté.
