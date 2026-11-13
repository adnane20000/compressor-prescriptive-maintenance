"""Index ChromaDB persistant ; publication atomique d'une nouvelle collection."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymupdf

from .chunking import CHUNK_VERSION, chunk_page
from .documents import document_metadata, extract_pages, file_sha256, load_registry, pdf_inventory
from .embeddings import default_encoder

INDEX_VERSION = "metropt-retrieval-v1"


def chroma_client(directory: str | Path):
    """Aucune fonction d'embeddings distante ou automatique attachée à Chroma."""
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(Path(directory).resolve() / "chroma"),
                                     settings=Settings(anonymized_telemetry=False, allow_reset=False))


def read_manifest(directory: str | Path) -> dict | None:
    path = Path(directory) / "index.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("index_version") != INDEX_VERSION:
        raise ValueError("Version d'index incompatible : réindexer les PDF.")
    return manifest


def registry_hash(directory: str | Path) -> str:
    path = Path(directory) / "sources.json"
    return file_sha256(path) if path.exists() else "absent"


def index_documents(
    documents_directory: str | Path = "documents", *, index_directory: str | Path = "data/local/rag",
    encoder=None, chunk_chars: int = 900, overlap_chars: int = 150, batch_size: int = 32,
) -> dict:
    """Indexer en lots ; conserver l'ancien index actif si le nouvel encodage échoue."""
    if batch_size < 1:
        raise ValueError("batch_size doit être strictement positif.")
    # Valider les paramètres même si le dossier est vide.
    from .documents import PDFPage
    chunk_page(PDFPage("", 1, "", {}), chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    documents_directory = Path(documents_directory).resolve()
    index_directory = Path(index_directory)
    encoder = encoder or default_encoder()
    inventory = pdf_inventory(documents_directory)
    original_registry_hash = registry_hash(documents_directory)
    registry = load_registry(documents_directory)
    client = chroma_client(index_directory)
    previous = read_manifest(index_directory)
    name = "metropt_" + uuid.uuid4().hex
    collection = client.create_collection(name, embedding_function=None,
                                          metadata={"index_version": INDEX_VERSION,
                                                    "encoder_signature": json.dumps(encoder.signature, sort_keys=True)},
                                          configuration={"hnsw": {"space": "cosine", "num_threads": 2}})
    warnings, documents, batch = [], [], []
    pages = empty_pages = count = 0

    def flush():
        nonlocal count
        if not batch:
            return
        vectors = np.asarray(encoder.encode_documents([chunk.text for chunk in batch]), dtype="float32")
        if vectors.shape != (len(batch), encoder.signature["dimension"]) or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
            raise ValueError("Embeddings invalides : dimensions, valeurs non finies ou vecteur nul.")
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        collection.add(ids=[chunk.identifier for chunk in batch], embeddings=vectors.tolist(),
                       documents=[chunk.text for chunk in batch], metadatas=[chunk.metadata for chunk in batch])
        count += len(batch)
        batch.clear()

    try:
        for item in inventory:
            metadata = document_metadata(item, registry)
            documents.append(metadata)
            if metadata["authority_basis"] == "checksum_mismatch":
                warnings.append(f"{item['relative_path']} : empreinte modifiée, provenance rétrogradée à non vérifiée.")
            try:
                for page in extract_pages(documents_directory / item["relative_path"], metadata):
                    pages += 1
                    if not page.text.strip():
                        empty_pages += 1
                        warnings.append(f"{item['relative_path']}, p. {page.page} : page sans texte exploitable ; OCR éventuellement nécessaire.")
                    for chunk in chunk_page(page, chunk_chars=chunk_chars, overlap_chars=overlap_chars):
                        batch.append(chunk)
                        if len(batch) >= batch_size:
                            flush()
            except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
                warnings.append(f"{item['relative_path']} : PDF illisible ({type(error).__name__}).")
        flush()
        # Le corpus/registre ne doit pas avoir changé pendant le travail.
        if pdf_inventory(documents_directory) != inventory or registry_hash(documents_directory) != original_registry_hash:
            raise ValueError("Le corpus a changé pendant l'indexation : relancer.")
        manifest = {"index_version": INDEX_VERSION, "collection": name,
                    "documents_directory": str(documents_directory), "inventory": inventory,
                    "registry_sha256": original_registry_hash, "encoder": encoder.signature,
                    "chunking": {"version": CHUNK_VERSION, "chunk_chars": chunk_chars, "overlap_chars": overlap_chars},
                    "pdf_count": len(inventory), "page_count": pages, "empty_pages": empty_pages,
                    "chunk_count": count, "documents": documents, "warnings": warnings,
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
        index_directory.mkdir(parents=True, exist_ok=True)
        temporary = index_directory / "index.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(index_directory / "index.json")
    except BaseException:
        client.delete_collection(name)
        raise
    if previous is not None:
        client.delete_collection(previous["collection"])
    return manifest
