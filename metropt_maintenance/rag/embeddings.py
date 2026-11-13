"""Embeddings Sentence Transformers locaux ; aucun LLM et aucune clé API."""

from functools import lru_cache
import os
from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class SentenceTransformerEncoder:
    """Préfixes E5 query/passage et normalisation conformes à la fiche du modèle."""

    signature = {"model": MODEL_NAME, "revision": MODEL_REVISION, "dimension": 384,
                 "normalization": "l2", "prefixes": "query:/passage:-v1", "max_tokens": 512}

    def __init__(self, *, cache_directory: str | Path = "data/local/embedding_models", allow_download: bool = False):
        self.cache_directory = Path(cache_directory).resolve()
        self.allow_download = allow_download
        self._model = None

    def prepare(self):
        """Charger paresseusement ; les tests courants injectent un encodeur local factice."""
        if self._model is None:
            # Les bibliothèques HF/Xet ont leurs propres caches, distincts du
            # cache_folder de Sentence Transformers : tout reste dans le projet.
            os.environ.setdefault("HF_HOME", str(self.cache_directory / "hf_home"))
            os.environ.setdefault("HF_XET_CACHE", str(self.cache_directory / "xet"))
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            from sentence_transformers import SentenceTransformer
            import torch
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            torch.set_num_threads(2)
            try:
                self._model = SentenceTransformer(
                    MODEL_NAME, revision=MODEL_REVISION, device="cpu",
                    cache_folder=str(self.cache_directory), local_files_only=not self.allow_download,
                    trust_remote_code=False, model_kwargs={"use_safetensors": True},
                )
                self._model.max_seq_length = 512
            except (OSError, ValueError) as error:
                raise RuntimeError("Modèle d'embeddings indisponible. Exécuter : python -m metropt_maintenance.rag prepare-model") from error
        return self

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        self.prepare()
        inputs = [prefix + text for text in texts]
        for text in inputs:
            tokens = self._model.tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if len(tokens) > 512:
                raise ValueError("Texte dépassant 512 tokens : réduire les chunks ou raccourcir la question ; aucune troncature silencieuse.")
        return np.asarray(self._model.encode(inputs, batch_size=16, show_progress_bar=False,
                                            normalize_embeddings=True, convert_to_numpy=True), dtype="float32")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def encode_query(self, question: str) -> np.ndarray:
        return self._encode([question], "query: ")[0]


@lru_cache(maxsize=1)
def default_encoder() -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder()
