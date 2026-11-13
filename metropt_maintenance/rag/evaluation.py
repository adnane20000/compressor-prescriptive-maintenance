"""Mesures de recherche sur questions annotées ; aucune mesure industrielle inventée."""

from pathlib import Path
from importlib.metadata import version

from .index import read_manifest
from .search import search_documents


def evaluate_retrieval(cases: list[dict], *, index_directory: str | Path, encoder=None,
                       top_k: int = 3, min_similarity: float = 0.80) -> dict:
    """Mesurer hit@k, rang réciproque et abstention sur ce seul jeu annoté."""
    if not cases:
        raise ValueError("Le benchmark doit contenir des questions annotées.")
    outcomes = []
    for case in cases:
        if not isinstance(case.get("question"), str) or not isinstance(case.get("should_abstain"), bool):
            raise ValueError("Chaque cas exige question et should_abstain explicites.")
        expected = case.get("expected_references", [])
        if not case["should_abstain"] and not expected:
            raise ValueError("Une question répondable exige une référence annotée.")
        result = search_documents(case["question"], index_directory=index_directory, encoder=encoder,
                                  top_k=top_k, min_similarity=min_similarity)
        if result.reason in ("index_missing", "no_extractable_text", "corpus_changed_reindex_required"):
            raise ValueError("Corpus non prêt pour évaluer la qualité : indexer ou réindexer les PDF d'abord.")
        rank = None
        for position, hit in enumerate(result.passages, 1):
            if any(hit.relative_path == reference["file"] and hit.page == reference["page"] for reference in expected):
                rank = position
                break
        outcomes.append({"question": case["question"], "should_abstain": case["should_abstain"],
                         "expected_references": expected, "status": result.status, "reason": result.reason,
                         "first_relevant_rank": rank,
                         "returned_references": [{"file": hit.relative_path, "page": hit.page, "similarity": hit.similarity,
                                                  "authority": hit.authority} for hit in result.passages]})
    positives = [outcome for outcome in outcomes if not outcome["should_abstain"]]
    negatives = [outcome for outcome in outcomes if outcome["should_abstain"]]
    manifest = read_manifest(index_directory)
    return {"question_count": len(cases), "answerable_count": len(positives), "unanswerable_count": len(negatives),
            "hit_at_1": sum(outcome["first_relevant_rank"] == 1 for outcome in positives) / len(positives) if positives else None,
            "hit_at_k": sum(outcome["first_relevant_rank"] is not None for outcome in positives) / len(positives) if positives else None,
            "mean_reciprocal_rank_at_k": sum(1 / outcome["first_relevant_rank"] if outcome["first_relevant_rank"] else 0 for outcome in positives) / len(positives) if positives else None,
            "unanswerable_abstention_rate": sum(outcome["status"] == "information_insuffisante" for outcome in negatives) / len(negatives) if negatives else None,
            "top_k": top_k, "min_similarity": min_similarity, "cases": outcomes,
            "encoder": manifest["encoder"] if manifest else None,
            "corpus_inventory": manifest["inventory"] if manifest else [],
            "indexed_pages": manifest["page_count"] if manifest else 0,
            "indexed_chunks": manifest["chunk_count"] if manifest else 0,
            "library_versions": {name: version(name) for name in ("sentence-transformers", "chromadb", "PyMuPDF", "torch")},
            "relevance_definition": "Correspondance au PDF et à la page annotés ; pas une vérification de réponse complète.",
            "limits": "Petit jeu annoté : résultats locaux, sans généralisation aux manuels absents ni certification de suffisance."}
