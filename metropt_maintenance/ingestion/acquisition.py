"""Téléchargement explicite de l'archive officielle, extraction en flux."""

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests

from .csv_reader import inspect_header
from .schema import ARCHIVE_URL, CSV_NAME, SOURCE_URL


def download_dataset(directory: str | Path, *, overwrite: bool = False) -> Path:
    """Télécharger sans charger l'archive en mémoire et sans extraction arbitraire.

    Le CSV existant est préservé tant que le téléchargement, la vérification CRC
    de l'archive et la validation du schéma du nouveau fichier n'ont pas réussi.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / CSV_NAME
    if target.exists() and not overwrite:
        raise FileExistsError(f"Le CSV existe déjà : {target}. Utiliser --overwrite explicitement.")
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="metropt-download-", dir=directory) as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.zip"
        with requests.get(ARCHIVE_URL, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    output.write(block)
                    digest.update(block)
        candidate = temporary / CSV_NAME
        try:
            with ZipFile(archive) as source:
                with source.open(CSV_NAME) as member, candidate.open("wb") as output:
                    shutil.copyfileobj(member, output, length=1024 * 1024)
        except (BadZipFile, KeyError) as error:
            raise ValueError("Archive UCI invalide ou CSV attendu absent.") from error
        inspect_header(candidate)
        candidate.replace(target)
    provenance = {
        "source": SOURCE_URL, "archive_url": ARCHIVE_URL,
        "archive_sha256": digest.hexdigest(), "csv_bytes": target.stat().st_size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0", "historical_data": True,
    }
    target.with_suffix(".source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target
