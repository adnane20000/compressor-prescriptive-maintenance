"""Préparer le modèle, indexer les PDF et tester une question en JSON."""

import argparse
import json
from pathlib import Path

from .embeddings import SentenceTransformerEncoder
from .index import index_documents
from .search import search_documents
from .evaluation import evaluate_retrieval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recherche documentaire MetroPT — aucune génération LLM")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-model", help="Télécharger explicitement les poids publics, sans clé API")
    prepare.add_argument("--cache", default="data/local/embedding_models")
    index = commands.add_parser("index", help="Extraire et indexer documents/*.pdf")
    index.add_argument("--documents", default="documents")
    index.add_argument("--index", default="data/local/rag")
    index.add_argument("--chunk-chars", type=int, default=900)
    index.add_argument("--overlap-chars", type=int, default=150)
    search = commands.add_parser("search", help="Retrouver des passages et leurs vraies références")
    search.add_argument("question")
    search.add_argument("--index", default="data/local/rag")
    search.add_argument("--authority", choices=("all", "official", "pedagogical", "unverified"), default="all")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--min-similarity", type=float, default=0.80)
    evaluate = commands.add_parser("evaluate", help="Mesurer la recherche sur un petit benchmark annoté")
    evaluate.add_argument("benchmark", type=Path)
    evaluate.add_argument("--index", default="data/local/rag")
    evaluate.add_argument("--output", type=Path, default=Path("data/local/rag_quality.json"))
    for command in (index, search, evaluate):
        command.add_argument("--cache", default="data/local/embedding_models")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-model":
            encoder = SentenceTransformerEncoder(cache_directory=args.cache, allow_download=True).prepare()
            result = {"prepared": True, "encoder": encoder.signature}
        elif args.command == "index":
            result = index_documents(args.documents, index_directory=args.index,
                                     encoder=SentenceTransformerEncoder(cache_directory=args.cache),
                                     chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars)
        elif args.command == "search":
            result = search_documents(args.question, index_directory=args.index, authority=args.authority,
                                      encoder=SentenceTransformerEncoder(cache_directory=args.cache),
                                      top_k=args.top_k, min_similarity=args.min_similarity).to_dict()
        else:
            benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
            result = evaluate_retrieval(benchmark["cases"], index_directory=args.index,
                                        encoder=SentenceTransformerEncoder(cache_directory=args.cache))
            result["scope"] = benchmark.get("scope", "non précisé")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Erreur de recherche documentaire : {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
