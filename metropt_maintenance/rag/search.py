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
from .safety import SAFETY_NOTICE, assess_safety_request
from ..ingestion.schema import SIGNAL_NAMES

STOPWORDS = set("a au aux avec ce ces cet cette comme comment dans de des du en est et etre font for from how il ils is la le les leur leurs lui ne nous on ou par pas pour que quel quelle quelles quels qui se selon sont sur the to un une units what when where which pourquoi pouvez donner donne veuillez please does do are can could would expliquer explique signifie definition document documents pdf page pages".split())
STOPWORDS.update("in of by at an it its as about into with be been being this that these those their them our your we you they than then through".split())
SYNONYMS = {"huile": "oil", "courant": "current", "moteur": "motor", "pression": "pressure",
            "compresseur": "compressor", "unite": "unit", "unites": "unit", "units": "unit",
            "temperatures": "temperature", "oil_temperature": "temperature", "capteur": "sensor",
            "capteurs": "sensor", "reservoir": "reservoirs", "donnees": "data", "collecte": "collected",
            "collectees": "collected"}
SYNONYMS.update({"secher": "dry", "sechage": "dry", "drying": "dry", "dryer": "dry", "dryers": "dry",
                 "condensats": "condensate", "condensates": "condensate", "fuites": "leak", "leaks": "leak"})
TOPIC_STOPWORDS = set("guide manuel manual sourcebook air compressed comprime compressor compresseurs system systeme nasa doe atlas copco kaeser according".split())
SOURCE_ALIASES = {"nasa": ("nasa", "aeronautics"), "doe": ("department of energy",),
                  "kaeser": ("kaeser",), "atlas": ("atlas copco",), "copco": ("atlas copco",), "inrs": ("inrs",), "uci": ("uci",)}
TOPIC_STOPWORDS.update(("inrs", "uci"))
OPERATIONAL_TERMS = set("procedure procedures securite safety urgence emergency consignation lockout serrage torque couple demonter demontage reparer reparation repair replacement replace disassemble disconnect debrancher isolate consigner intervention intervenir remplacement remplacer constructeur manufacturer maximale maximal maximum limite autorisee admissible".split())
CRITICAL_GROUPS = (
    {"serrage", "torque"}, {"consignation", "lockout"}, {"urgence", "emergency"},
    {"maximale", "maximal", "maximum", "limite", "admissible", "autorisee"},
    {"residuel", "residuels", "residuelle", "residuelles", "residual", "residuals"},
    {"temperature", "temperatures"},
)


def query_terms(text: str) -> set[str]:
    """Ancres lexicales conservatrices, avec quelques équivalences bilingues métier."""
    normalized = "".join(character for character in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(character))
    words = re.findall(r"[a-z0-9]+", normalized)
    return {SYNONYMS.get(word, word) for word in words if len(word) >= 2 and word not in STOPWORDS}


def signal_pattern(name: str) -> str:
    """Relier le nom CSV à son libellé PDF espacé, sans changer les lettres du signal."""
    return r"(?<!\w)" + r"[\s_]+".join(re.escape(part) for part in name.split("_")) + r"(?!\w)"


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
    applicability: str = "not_verified"
    equipment_reference: str = ""
    text_is_untrusted_data: bool = True
    retrieval_basis: str = "semantic_and_lexical_guards"


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
    safety_blocked: bool = False
    intervention_authorized: bool = False

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
    safety = assess_safety_request(question)
    if safety.blocked:
        result = insufficient(question, "unsafe_intervention_request", warnings=[SAFETY_NOTICE, *safety.reasons])
        result.safety_blocked = True
        return result
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
    conditions = [{"authority": authority}] if authority != "all" else []
    requested_sources = terms & SOURCE_ALIASES.keys()
    explicit_signals = [name for name in SIGNAL_NAMES if re.search(signal_pattern(name), question, re.I)]
    if requested_sources:
        paths = [document["relative_path"] for document in manifest["documents"]
                 if any(alias in (document["title"] + " " + document["publisher"] + " " + document["relative_path"]).lower()
                        for source in requested_sources for alias in SOURCE_ALIASES[source])]
        if not paths:
            return insufficient(question, "requested_source_missing")
        conditions.append({"relative_path": {"$in": paths}})
    kwargs = {"where": conditions[0] if len(conditions) == 1 else {"$and": conditions}} if conditions else {}
    candidates = collection.query(query_embeddings=[vector.tolist()], n_results=min(collection.count(), max(100, top_k * 20)),
                                  include=["documents", "metadatas", "distances"], **kwargs)
    operational = bool(terms & OPERATIONAL_TERMS)
    project_specific = operational and any(term.startswith("metropt") for term in terms)
    accepted, related = [], []
    topics = terms - TOPIC_STOPWORDS
    seen_pages = set()
    # Seuls les documents de l'inventaire actif peuvent fournir une citation.
    known = {item["relative_path"]: item["sha256"] for item in manifest["inventory"]}
    for text, metadata, distance in zip(candidates["documents"][0], candidates["metadatas"][0], candidates["distances"][0]):
        similarity = float(np.clip(1 - distance, -1, 1))
        passage_terms = query_terms(text)
        matched = terms & passage_terms
        critical_covered = all(not (terms & group) or bool(passage_terms & group) for group in CRITICAL_GROUPS)
        navigation = bool(re.search(r"\.{5,}", text)) and not (terms & {"sommaire", "contents"})
        exact_signal = any(re.search(signal_pattern(name), text, re.I) for name in explicit_signals)
        if explicit_signals and not exact_signal:
            continue
        # Une définition de variable explicitement demandée à UCI peut être
        # retrouvée par son identifiant littéral. Ce n'est pas un seuil ML abaissé
        # ni une validation de procédure ; les garde-fous de type restent appliqués.
        exact_dataset_lookup = bool("uci" in requested_sources and exact_signal and metadata["authority"] == "official"
                                    and metadata["document_type"] == "dataset_description")
        if (similarity < min_similarity and not exact_dataset_lookup) or not matched or not critical_covered or (navigation and not exact_dataset_lookup) or (topics and not topics.intersection(passage_terms)):
            continue
        if known.get(metadata["relative_path"]) != metadata["pdf_sha256"] or metadata["page"] < 1:
            raise ValueError("Référence hors de l'inventaire actif.")
        page_key = (metadata["relative_path"], metadata["page"])
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        hit = Passage(
            text=text, reference=f"{metadata['relative_path']}, p. {metadata['page']}",
            file_name=metadata["file_name"], relative_path=metadata["relative_path"], page=metadata["page"],
            page_label=metadata["page_label"], pdf_sha256=metadata["pdf_sha256"], authority=metadata["authority"],
            document_type=metadata["document_type"], publisher=metadata["publisher"], source_url=metadata["source_url"],
            authority_basis=metadata["authority_basis"], similarity=similarity, matched_query_terms=sorted(matched),
            applicability=metadata.get("applicability", "not_verified"),
            equipment_reference=metadata.get("equipment_reference", ""),
            retrieval_basis="exact_dataset_signal_and_requested_publisher" if exact_dataset_lookup else "semantic_and_lexical_guards",
        )
        project_matched = (hit.applicability == "equipment_specific" and
                           any(term.startswith("metropt") for term in query_terms(hit.equipment_reference)))
        if (operational and not (hit.authority == "official" and hit.document_type in OPERATIONAL_TYPES)) or (project_specific and not project_matched):
            related.append(hit)
        else:
            accepted.append(hit)
    warnings = list(manifest["warnings"])
    if any(hit.retrieval_basis == "exact_dataset_signal_and_requested_publisher" for hit in accepted + related):
        warnings.append("Définition de dataset retrouvée par nom exact du signal et éditeur UCI demandé ; aucune procédure de maintenance n'est validée.")
    if any(hit.applicability != "equipment_specific" for hit in accepted + related):
        warnings.append("Référence générale ou portée non vérifiée : aucune applicabilité au compresseur MetroPT-3 n'est établie.")
    if not accepted:
        reason = "equipment_specific_source_missing" if project_specific and related else "operational_source_missing" if related else "no_relevant_passage"
        return insufficient(question, reason, warnings=warnings, related=related[:top_k])
    return SearchResult(question, "passages_found", "Passages retrouvés ; vérifier leur portée et leur applicabilité.",
                        "retrieval_only", passages=accepted[:top_k], warnings=warnings)
