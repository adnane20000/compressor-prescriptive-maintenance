"""Provenance humaine et extraction textuelle des PDF, page physique par page."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pymupdf

AUTHORITIES = ("official", "pedagogical", "unverified")
DOCUMENT_TYPES = ("dataset_description", "manufacturer_manual", "maintenance_guide", "fmea", "safety_procedure", "educational_note", "other")
OPERATIONAL_TYPES = ("manufacturer_manual", "maintenance_guide", "fmea", "safety_procedure")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_inventory(directory: str | Path) -> list[dict]:
    """Lister uniquement les PDF sous le dossier demandé, sans liens vers l'extérieur."""
    directory = Path(directory).resolve()
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("Le chemin des documents doit être un dossier.")
    inventory = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".pdf":
            if not path.resolve().is_relative_to(directory):
                raise ValueError(f"PDF hors du dossier autorisé : {path.name}")
            inventory.append({"relative_path": path.relative_to(directory).as_posix(), "sha256": file_sha256(path)})
    return inventory


def load_registry(directory: str | Path) -> dict:
    """Charger sources.json ; ce registre est une déclaration, pas une certification."""
    path = Path(directory) / "sources.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("documents"), dict):
        raise ValueError("Registre sources.json invalide : version 1 et documents attendus.")
    for relative, entry in data["documents"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not isinstance(entry, dict):
            raise ValueError("Chemin ou entrée invalide dans le registre.")
        if entry.get("authority") not in AUTHORITIES or entry.get("document_type") not in DOCUMENT_TYPES:
            raise ValueError(f"Statut ou type documentaire invalide : {relative}")
        digest = entry.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Empreinte SHA-256 invalide : {relative}")
        if entry["authority"] == "official":
            url = urlparse(entry.get("source_url", ""))
            if not entry.get("publisher") or url.scheme not in ("http", "https") or not url.netloc:
                raise ValueError(f"Une source officielle déclarée exige un éditeur et une URL : {relative}")
    return data["documents"]


def document_metadata(item: dict, registry: dict) -> dict:
    """Lier la provenance au contenu ; une empreinte modifiée invalide l'attribution."""
    relative = item["relative_path"]
    metadata = {"file_name": Path(relative).name, "relative_path": relative, "pdf_sha256": item["sha256"],
                "authority": "unverified", "document_type": "other", "publisher": "", "source_url": "",
                "title": Path(relative).name, "authority_basis": "not_declared"}
    declaration = registry.get(relative)
    if declaration is not None:
        if declaration["sha256"] != item["sha256"]:
            metadata["authority_basis"] = "checksum_mismatch"
        else:
            metadata.update({name: str(declaration.get(name, metadata[name])) for name in
                             ("authority", "document_type", "publisher", "source_url", "title")})
            metadata["authority_basis"] = "human_registry_matched_sha256"
    return metadata


@dataclass(frozen=True)
class PDFPage:
    text: str
    page: int
    page_label: str
    metadata: dict


def extract_pages(path: str | Path, metadata: dict):
    """Extraire du texte brut ; ne lire ni exécuter scripts, liens ou pièces jointes.

    Les pages sans texte sont conservées comme telles pour signaler les scans et
    l'absence d'OCR. La page physique est indépendante du numéro imprimé.
    """
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"PDF chiffré : {Path(path).name}")
        for index in range(document.page_count):
            page = document.load_page(index)
            yield PDFPage(page.get_text("text", sort=True).replace("\x00", ""),
                          index + 1, page.get_label() or "", metadata)
