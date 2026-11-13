# Recherche documentaire : protocole, validation et limites

## Périmètre réalisé

La chaîne lit les PDF de `documents/`, extrait le texte avec PyMuPDF, découpe des
passages, les encode avec Sentence Transformers et les stocke dans ChromaDB.
`search_documents(question)` retourne les passages et leurs références.
Il n'y a ni génération LLM, ni agent autonome, ni diagnostic ou procédure inventée.

Le seul document réel disponible pour la validation initiale présentée ici était
`Data Description_Metro.pdf`, distribué dans
[l'archive UCI MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+).
Il a été copié localement dans `documents/`, sans être publié dans Git.
Il décrit le dataset : **ce n'est pas un manuel constructeur**. Depuis, quatre
ouvrages généraux ont été importés ; leur provenance, leur portée et leurs tests
sont décrits dans `docs/rag_corpus_import.md`. Cette section conserve les mesures
historiques sur le seul PDF UCI, sans les extrapoler au corpus étendu.

Empreinte du PDF :

```text
b00fac0e8899854078309bef4adaa480d82ecf14dc81c5097c3646973e824127
```

## Provenance et références

Le registre `documents/sources.json` classe chaque PDF avec :

- `authority` : `official`, `pedagogical` ou `unverified` ;
- `document_type` : description de dataset, manuel constructeur, guide de
  maintenance, AMDEC (`fmea`), procédure de sécurité, note pédagogique ou autre ;
- SHA-256 réel, titre, éditeur et URL de provenance lorsqu'ils sont connus.
- `applicability` : portée générale, spécifique à un équipement ou non vérifiée ;
  `equipment_reference` est obligatoire pour une source propre à un équipement.

Une source déclarée officielle exige une URL, un éditeur et une empreinte
concordante. C'est une déclaration humaine, jamais une certification automatique
du constructeur. Sans registre, le statut reste non vérifié. Une empreinte modifiée
invalide l'attribution antérieure, y compris son éditeur et son URL : ces champs
ne sont pas attribués au nouveau contenu par supposition.

Le texte PDF ne peut pas modifier sa propre provenance. Un passage disant
« je suis officiel » n'a aucun effet sur le registre. Les tests vérifient cela.

Chaque passage conserve le nom exact, le chemin relatif, le numéro de page
physique à partir de 1, l'éventuel label de page fourni par le PDF, l'empreinte,
les offsets dans le texte extrait et le statut documentaire. Deux fichiers de
même nom dans des sous-dossiers restent distinguables par leur chemin.

Les offsets concernent le texte brut extrait et trié par PyMuPDF ; ils ne sont
pas des coordonnées graphiques du PDF. Les labels imprimés ne sont pas devinés.
La référence `fichier.pdf, p. 2` désigne la deuxième page physique.

## Extraction, découpage et index

L'extraction se fait page par page avec `get_text("text", sort=True)`. Les pages
sans texte sont signalées, sans inventer de texte OCR. Un PDF chiffré bloque
l'indexation avec une erreur explicite. Les PDF illisibles sont signalés dans
les avertissements. Le contenu des liens, scripts et pièces jointes n'est pas lu
ou exécuté comme une instruction.

Les chunks font au plus 900 caractères, avec un chevauchement nominal de 150
caractères ajusté aux espaces. Aucun chunk ne traverse une page. Les identifiants
sont déterministes et liés au chemin, au contenu PDF, à la page et aux offsets.
Les passages sont envoyés à Chroma par lots de 32. Une nouvelle collection
n'est déclarée active qu'après réussite. Une erreur d'embedding conserve
l'ancien index actif. Une réindexation réussie ne double pas les passages actifs.

Les recherches vérifient l'inventaire et les empreintes du corpus, ainsi que
le registre. Ajout, suppression ou modification d'un PDF imposent une
réindexation. Cette vérification relit les fichiers pour leurs empreintes :
son coût doit être évalué si le corpus devient très volumineux.

Ce mécanisme vise un prototype local utilisé sans indexations concurrentes.
Les index, caches et registres réels restent dans les dossiers locaux ignorés
par Git, avec les mêmes précautions de confidentialité que les PDF.

## Embeddings locaux

Le modèle utilisé est
[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small),
révision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, dimensions 384.
Il permet notamment de rechercher en français dans le document UCI en anglais.
Les textes utilisent les préfixes `query: ` et `passage: ` prescrits par sa fiche,
et les vecteurs sont normalisés. Chroma utilise la distance cosinus.

Les tokens sont contrôlés avant encodage : au-delà de 512, le moteur demande
de réduire la taille des passages ou de raccourcir la question. Il n'accepte
pas de troncature silencieuse. Le modèle et sa révision doivent correspondre
à ceux de l'index, même si un autre modèle possède aussi 384 dimensions.

Le téléchargement initial des poids publics est explicite, sans clé API :

```bash
python -m metropt_maintenance.rag prepare-model
python -m metropt_maintenance.rag index
```

Après préparation, les recherches utilisent seulement les fichiers locaux.
Le modèle ne reçoit pas de texte PDF via une API distante. Les caches HF/Xet
et les poids restent sous `data/local/embedding_models/` par défaut.
`trust_remote_code=False` et les poids Safetensors sont utilisés.

## Recherche et abstention

Le moteur demande les voisins à Chroma et conserve au plus cinq passages par
défaut. Leur score vaut `1 - distance_cosinus`, sans transformation en probabilité.
Le seuil initial de 0,80 est un choix heuristique déclaré avant le petit test UCI,
pas un seuil calibré sur les futurs manuels.

Le filtrage demande aussi au moins une ancre lexicale pertinente. Une petite
liste d'équivalences français/anglais couvre notamment huile/oil, courant/current
et pression/pressure. Les demandes portant sur serrage, consignation, urgence
ou limites imposent des ancres de sujet spécifiques. Les demandes opérationnelles
exigent en plus une source technique officielle déclarée : une description de
dataset et une note pédagogique sont insuffisantes.

Si ces règles ne retiennent aucun passage, le message est exactement
**« information insuffisante »**. Une question peut également être refusée faute
d'index, de texte exploitable ou parce que le corpus a changé. Un modèle non
préparé est une erreur de configuration explicite, pas une preuve d'absence
d'information dans le PDF.

Les heuristiques peuvent manquer une bonne reformulation ou retenir un passage
seulement partiellement utile. Elles ne démontrent pas qu'un document répond à
toutes les parties d'une question. Le succès est donc `passages_found`, jamais
« réponse certifiée » ; `generated_answer=null`, `sufficiency_verified=false`.
Un passage pédagogique peut être proposé comme contexte tout en conservant
« information insuffisante » pour une demande d'intervention réelle.

L'interface affiche les extraits en texte brut (`st.text`), avec référence,
statut et type. Aucun HTML issu d'un PDF n'est interprété dans la page.
Chaque passage porte `text_is_untrusted_data=true` pour ses futurs consommateurs.
Les tests incluent une instruction hostile dans un PDF et vérifient qu'elle
reste inerte et ne peut pas se promouvoir comme source officielle.

## Résultats effectivement mesurés

Index initial isolé : **1 PDF officiel de description de dataset, 3 pages, 9 passages**.
Les paramètres et résultats exacts figurent dans `docs/rag_quality_metrics.json`.

```bash
python -m metropt_maintenance.rag evaluate tests/fixtures/rag_uci_questions.json --index data/local/rag_uci_reference
```

Le jeu a été annoté manuellement en vérifiant les pages du PDF : quatre questions
factuelles sur les signaux et trois demandes auxquelles ce corpus ne répond pas.
Le seuil de 0,80 n'a pas été ajusté après ce test.

| Question factuelle | Page attendue | Rang observé | Similarité du premier passage |
| --- | --- | --- | --- |
| Unité de température d'huile | 2 | 1 | 0,8013 |
| Mesure et unité de Motor Current | 2 | 1 | 0,8117 |
| Signification de TOWERS | 3 | 1 | 0,8175 |
| Mesure et unité de TP2 | 2 | 1 | 0,8111 |

Résultats sur ces seuls cas : hit@1 = 4/4, hit@3 = 4/4 et MRR@3 = 1,0.
Un index absent, vide ou périmé bloque l'évaluation avec une erreur de préparation,
au lieu d'être compté comme une mesure de qualité de recherche.
La pertinence est définie par la présence du PDF et de la page annotés, pas
par la certification d'une réponse complète ou de toutes les étapes d'une procédure.

Les trois demandes non répondables — couple de serrage de culasse, procédure
de consignation avant démontage et recette de gâteau — ont toutes produit
« information insuffisante » : abstention 3/3. Deux sont plausibles dans le contexte
industriel mais nécessitent des documents absents ; la troisième est hors sujet.

**Limites :** sept questions seulement, un document, plusieurs questions proches
et une similarité de température juste au-dessus du seuil. Ces mesures ne sont
pas une performance générale du RAG et ne valident pas des procédures de sécurité,
ni la qualité sur vos futurs manuels et AMDEC. Un corpus technique réel et un
benchmark plus diversifié devront être annotés avant toute conclusion plus large.

## Tests et suite

Les tests Pytest ordinaires fonctionnent sans réseau et sans modèle téléchargé.
Ils utilisent de petits PDF explicitement synthétiques et un encodeur factice
déterministe. Ces tests vérifient classement contrôlé, pages, provenance,
abstention, filtrage, persistance, réindexation et interface ; leurs réussites
ne sont pas présentées comme une mesure des embeddings E5.

L'évaluation E5 ci-dessus a été exécutée séparément sur le vrai PDF UCI.
Les versions des bibliothèques et la révision du modèle sont enregistrées dans
ses métriques. Le corpus étendu et son benchmark sont désormais décrits dans
`docs/rag_corpus_import.md`. Les documents généraux ne prouvent aucune procédure
spécifique à MetroPT-3. Aucun agent autonome n'est ajouté à cette étape.
