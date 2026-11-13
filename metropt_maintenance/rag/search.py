"""Recherche de passages sourcés, abstention prudente ; aucune génération de réponse."""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .documents import AUTHORITIES, OPERATIONAL_TYPES, pdf_inventory
from .embeddings import default_encoder
from .index import INDEX_VERSION, chroma_client, read_manifest, registry_hash

STOPWORDS = set("a au aux avec ce ces cet cette comme comment dans de des du en est et etre font for from how il ils is la le les leur leurs lui ne nous on ou par pas pour que quel quelle quelles quels qui se selon sont sur the to un une units what when where which pourquoi pouvez donner donne veuillez please does do are can could would expliquer explique signifie definition document documents pdf page pages".split())
SYNONYMS = {"huile": "oil", "courant": "current", "moteur": "motor", "pression": "pressure",
            "compresseur": "compressor", "unite": "unit", "unites": "unit", "units": "unit",
            "temperatures": "temperature", "oil_temperature": "temperature", "capteur": "sensor",
            "capteurs": "sensor", "reservoir": "reservoirs", "donnees": "data", "collecte": "collected",
            "collectees": "collected"}
OPERATIONAL_TERMS = set("procedure procedures securite safety urgence emergency consignation lockout serrage torque couple demonter demontage reparer reparation repair replacement replace disassemble disconnect debrancher isolate consigner intervention intervenir remplacement remplacer constructeur manufacturer maximale maximal maximum limite autorisee admissible".split())
CRITICAL_GROUPS = (
    {"serrage", "torque"}, {"consignation", "lockout"}, {"urgence", "emergency"},
    {"maximale", "maximal", "maximum", "limite", "admissible", "autorisee"},
)


def query_terms(text: str) -> set[str]:
    """Ancres lexicales conservatrices, avec quelques équivalences bilingues métier."""
    normalized = "".join(character for character in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(character))
    words = re.findall(r"[a-z0-9]+", normalized)
    return {SYNONYMS.get(word, word) for word in words if len(word) >= 2 and word not in STOPWORDS}


@dataclass(frozen=True)
class Passage:
    text: str
    reference: str
    file_name: str
    relative_path: str
    page: int
    page_label: str
    pdf_sha256: str
    authority: str
    document_type: str
    publisher: str
    source_url: str
    authority_basis: str
    similarity: float
    matched_query_terms: list[str]
    text_is_untrusted_data: bool = True


@dataclass
class SearchResult:
    question: str
    status: str
    message: str
    reason: str
    passages: list[Passage] = field(default_factory=list)
    related_passages: list[Passage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_answer: None = None
    similarity_is_probability: bool = False
    sufficiency_verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def insufficient(question: str, reason: str, *, warnings=None, related=None) -> SearchResult:
    return SearchResult(question, "information_insuffisante", "information insuffisante", reason,
                        warnings=warnings or [], related_passages=related or [])


def search_documents(
    question: str, *, index_directory: str | Path = "data/local/rag", encoder=None,
    authority: str = "all", top_k: int = 5, min_similarity: float = 0.80,
) -> SearchResult:
    """Retourner seulement des extraits, jamais exécuter les instructions d'un PDF.

    Le seuil et les ancres lexicales sont des garde-fous heuristiques. Leur succès
    ne prouve ni la complétude d'une réponse, ni l'applicabilité d'une procédure.
    Les demandes opérationnelles exigent une source technique officielle déclarée.
    """
    question = question.strip()
    if not question or len(question) > 4000:
        raise ValueError("La question doit contenir entre 1 et 4 000 caractères.")
    if authority not in (*AUTHORITIES, "all") or not 1 <= top_k <= 20 or not -1 <= min_similarity <= 1:
        raise ValueError("Filtre, top_k ou seuil de similarité invalide.")
    manifest = read_manifest(index_directory)
    if manifest is None:
        return insufficient(question, "index_missing")
    corpus = manifest["documents_directory"]
    if pdf_inventory(corpus) != manifest["inventory"] or registry_hash(corpus) != manifest["registry_sha256"]:
        return insufficient(question, "corpus_changed_reindex_required")
    if not manifest["chunk_count"]:
        return insufficient(question, "no_extractable_text", warnings=manifest["warnings"])
    encoder = encoder or default_encoder()
    if encoder.signature != manifest["encoder"]:
        raise ValueError("Modèle d'embeddings incompatible avec l'index : réindexer avec le même modèle.")
    collection = chroma_client(index_directory).get_collection(manifest["collection"], embedding_function=None)
    if collection.metadata.get("index_version") != INDEX_VERSION or collection.metadata.get("encoder_signature") != json.dumps(encoder.signature, sort_keys=True):
        raise ValueError("Contrat de la collection Chroma incompatible.")
    terms = query_terms(question)
    if not terms:
        return insufficient(question, "question_without_searchable_terms")
    vector = np.asarray(encoder.encode_query(question), dtype="float32")
    if vector.shape != (encoder.signature["dimension"],) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Embedding de question invalide.")
    vector = vector / np.linalg.norm(vector)
    kwargs = {"where": {"authority": authority}} if authority != "all" else {}
    candidates = collection.query(query_embeddings=[vector.tolist()], n_results=min(collection.count(), max(20, top_k * 4)),
                                  include=["documents", "metadatas", "distances"], **kwargs)
    operational = bool(terms & OPERATIONAL_TERMS)
    accepted, related = [], []
    # Seuls les documents de l'inventaire actif peuvent fournir une citation.
    known = {item["relative_path"]: item["sha256"] for item in manifest["inventory"]}
    for text, metadata, distance in zip(candidates["documents"][0], candidates["metadatas"][0], candidates["distances"][0]):
        similarity = float(np.clip(1 - distance, -1, 1))
        passage_terms = query_terms(text)
        matched = terms & passage_terms
        critical_covered = all(not (terms & group) or bool(passage_terms & group) for group in CRITICAL_GROUPS)
        if similarity < min_similarity or not matched or not critical_covered:
            continue
        if known.get(metadata["relative_path"]) != metadata["pdf_sha256"] or metadata["page"] < 1:
            raise ValueError("Référence hors de l'inventaire actif.")
        hit = Passage(
            text=text, reference=f"{metadata['relative_path']}, p. {metadata['page']}",
            file_name=metadata["file_name"], relative_path=metadata["relative_path"], page=metadata["page"],
            page_label=metadata["page_label"], pdf_sha256=metadata["pdf_sha256"], authority=metadata["authority"],
            document_type=metadata["document_type"], publisher=metadata["publisher"], source_url=metadata["source_url"],
            authority_basis=metadata["authority_basis"], similarity=similarity, matched_query_terms=sorted(matched),
        )
        if operational and not (hit.authority == "official" and hit.document_type in OPERATIONAL_TYPES):
            related.append(hit)
        else:
            accepted.append(hit)
    if not accepted:
        return insufficient(question, "operational_source_missing" if related else "no_relevant_passage",
                            warnings=manifest["warnings"], related=related[:top_k])
    return SearchResult(question, "passages_found", "Passages retrouvés ; vérifier leur portée et leur applicabilité.",
                        "retrieval_only", passages=accepted[:top_k], warnings=manifest["warnings"])
