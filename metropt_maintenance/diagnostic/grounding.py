"""Vérifier les données et citations avant et après appel au fournisseur."""

from dataclasses import asdict, is_dataclass
from pathlib import Path

import pymupdf

from ..ingestion.schema import SIGNALS
from ..rag.documents import document_metadata, file_sha256, load_registry
from .schemas import Measurement, SourceReference


class GroundingError(ValueError):
    """Source ou citation non vérifiable, sans publication de contenu non validé."""


def normalize_measurements(measurements: dict) -> dict[str, Measurement]:
    if not isinstance(measurements, dict):
        raise ValueError("Les mesures doivent être un dictionnaire de vrais signaux MetroPT-3.")
    units = {signal.name: signal.unit for signal in SIGNALS}
    normalized = {}
    for name, value in measurements.items():
        if name not in units:
            raise ValueError(f"Capteur non documenté : {name}")
        measure = value if isinstance(value, Measurement) else Measurement.model_validate(value if isinstance(value, dict) else {"value": value})
        if measure.unit is not None and measure.unit != units[name]:
            raise ValueError(f"Unité non conforme au schéma MetroPT-3 : {name}")
        normalized[name] = measure.model_copy(update={"unit": units[name]})
    return normalized


def verify_sources(passages, *, documents_directory: str | Path = "documents") -> list[SourceReference]:
    """Vérifier empreinte, page et texte littéral ; le PDF ne décide pas de son statut."""
    if hasattr(passages, "passages"):
        passages = passages.passages
    if not isinstance(passages, (list, tuple)) or len(passages) > 8:
        raise GroundingError("Fournir au plus huit passages RAG.")
    root = Path(documents_directory).resolve()
    registry = load_registry(root)
    output, checked = [], {}
    for index, passage in enumerate(passages, 1):
        data = asdict(passage) if is_dataclass(passage) else passage
        if not isinstance(data, dict):
            raise GroundingError("Format de passage invalide.")
        relative = data.get("relative_path", "")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise GroundingError("Chemin de PDF invalide.")
        path = (root / relative).resolve()
        page_number, text = data.get("page"), data.get("text")
        if not relative or not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".pdf":
            raise GroundingError("PDF cité absent ou hors du corpus autorisé.")
        if type(page_number) is not int or page_number < 1 or not isinstance(text, str) or not 10 <= len(text) <= 4000:
            raise GroundingError("Page ou texte de passage invalide.")
        if relative not in checked:
            digest = file_sha256(path)
            checked[relative] = document_metadata({"relative_path": relative, "sha256": digest}, registry)
        metadata = checked[relative]
        if data.get("pdf_sha256") != metadata["pdf_sha256"] or data.get("file_name") != metadata["file_name"]:
            raise GroundingError("L'empreinte ou le nom du PDF cité ne correspond pas au corpus.")
        for field in ("authority", "document_type", "applicability", "equipment_reference"):
            if field in data and data[field] != metadata[field]:
                raise GroundingError("La provenance ou la portée du passage ne correspond pas au registre.")
        try:
            with pymupdf.open(path) as document:
                if document.needs_pass or page_number > document.page_count:
                    raise GroundingError("Page citée inaccessible ou inexistante.")
                actual_text = document[page_number - 1].get_text("text", sort=True).replace("\x00", "")
                if text not in actual_text:
                    raise GroundingError("Texte cité absent de la page du PDF.")
        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
            raise GroundingError("PDF cité illisible.") from error
        output.append(SourceReference(id=f"S{index}", file_name=metadata["file_name"], relative_path=relative,
                                      page=page_number, pdf_sha256=metadata["pdf_sha256"], authority=metadata["authority"],
                                      document_type=metadata["document_type"], applicability=metadata["applicability"],
                                      equipment_reference=metadata["equipment_reference"], publisher=metadata["publisher"],
                                      source_url=metadata["source_url"], passage=text))
    return output
