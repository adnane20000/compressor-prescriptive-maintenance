# Corpus documentaire local

Déposer ici les PDF autorisés : manuels, guides, AMDEC et procédures de sécurité.
Les PDF, leurs index et `sources.json` ne sont pas publiés dans Git. Les documents
peuvent être organisés dans des sous-dossiers. Une recherche conserve le chemin
relatif, le nom exact du PDF, son empreinte et sa page physique (première page = 1).

Créer `sources.json` à partir de `sources.example.json`. Remplacer chaque champ
par la provenance réelle de vos documents. Une attribution officielle exige un
éditeur, une URL source et une empreinte SHA-256 concordante. Cette déclaration
humaine n'est pas une certification automatique d'authenticité. Le texte du PDF
ne peut pas se déclarer lui-même officiel. Sans déclaration valide, le PDF reste
« non vérifié ». Une empreinte modifiée invalide l'attribution précédente.

Le PDF UCI `Data Description_Metro.pdf`, lorsqu'il est disponible localement,
décrit le dataset. Il n'est ni un manuel constructeur, ni une AMDEC, ni une
procédure de sécurité. Le corpus local contient aussi les guides généraux NASA,
DOE, KAESER, Atlas Copco et INRS fournis ou acquis pour le projet. Aucun n'est une
notice identifiée du compresseur MetroPT-3. Voir `docs/rag_corpus_import.md`
et `docs/inrs_safety_import.md`. Les PDF INRS `ed6109.pdf` et `ed6129.pdf` sont
déclarés `official`, `safety_procedure`, `general_reference` dans le registre
local, après comparaison de leurs empreintes avec les copies du site INRS.

ED 6109 traite des consignations et déconsignations ; ED 6129 étudie des modes
de fonctionnement encadrés avec protections neutralisées. Ces guides ne sont
pas des procédures propres à MetroPT et ne valent pas autorisation d'intervenir.

Le champ `applicability` distingue `general_reference`, `equipment_specific`
et `not_verified`. Une source spécifique doit renseigner `equipment_reference`.
Le statut officiel d'un éditeur ne prouve pas l'applicabilité à votre équipement.

Les pages scannées sans texte nécessitent un OCR externe et une vérification
humaine ; ce module n'effectue pas d'OCR. Les tableaux, dessins, colonnes et textes
tronqués peuvent limiter la recherche. Aucun passage ne valide à lui seul une
procédure complète ou une action sur le compresseur.
