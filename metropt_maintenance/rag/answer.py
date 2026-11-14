"""Réponse documentaire : sélection LLM de citations vérifiées, sans procédure libre."""

import re

from pydantic import Field, ValidationError

from ..diagnostic.grounding import verify_sources
from ..diagnostic.providers import LLMApiError, LLMConfigurationError, LLMResponseError, provider_from_env
from ..diagnostic.schemas import DiagnosticDraft, SourceReference, StrictModel
from ..ingestion.schema import SIGNAL_NAMES
from .search import signal_pattern


class DocumentAnswer(StrictModel):
    status: str
    message: str
    simulation: bool
    provider: str
    model: str
    excerpts: list[dict] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    sufficiency_verified: bool = False


def generate_document_answer(question, retrieval, *, documents_directory="documents", provider=None):
    """Réutiliser le fournisseur structuré existant, sans inventer d'événement ML.

    Le contrat de sélection des arguments est réutilisé ; les hypothèses restent
    interdites en l'absence de mesures. Aucune mesure factice n'est transmise.
    """
    if question.strip() != retrieval.question:
        raise ValueError("La question doit correspondre à la recherche documentaire.")
    name, model, simulation = "non_configuré", "non_configuré", False
    def result(status, message, **kwargs):
        return DocumentAnswer(status=status, message=message, simulation=simulation, provider=name, model=model, **kwargs)
    if retrieval.status != "passages_found" or not retrieval.passages or retrieval.safety_blocked:
        return result("information_insuffisante", "information insuffisante")
    try:
        sources = verify_sources(retrieval.passages, documents_directory=documents_directory)
    except (OSError, ValueError):
        return result("sources_invalides", "Les passages ne sont plus vérifiables dans les PDF. Réindexer les documents.")
    try:
        provider = provider or provider_from_env()
        name, model, simulation = provider.name, provider.model, provider.simulation
        selected_sources = [source.model_dump() for source in sources]
        if simulation:
            # Démo déterministe : centrer le passage sur le vrai nom du signal
            # demandé, tout en conservant un extrait littéral de la page vérifiée.
            requested = [signal for signal in SIGNAL_NAMES if re.search(signal_pattern(signal),question,re.I)]
            for source in selected_sources:
                positions = [match.start() for signal in requested if (match:=re.search(signal_pattern(signal),source["passage"],re.I))]
                if positions:
                    start = max(0,min(positions)-30)
                    source["passage"] = source["passage"][start:start+300]
        raw = provider.complete({"task":"documentary_answer_only", "question":question, "event":None, "facts":[],
                                 "sources":selected_sources,
                                 "constraints":"Sélectionner les arguments littéraux répondant à la question ; aucune mesure disponible, hypotheses doit rester vide. Aucune procédure propre à MetroPT à partir d'un guide général."})
        if not isinstance(raw, str) or len(raw)>32_000:
            raise ValueError("Réponse trop volumineuse.")
        draft = DiagnosticDraft.model_validate_json(raw)
        if draft.hypotheses:
            raise ValueError("Aucune hypothèse sans mesures.")
        by_id = {source.id:source for source in sources}
        excerpts = []
        for argument in draft.arguments:
            if argument.source_id not in by_id or not argument.quote.strip() or argument.quote not in by_id[argument.source_id].passage:
                raise ValueError("Citation non vérifiable.")
            excerpts.append({"source_id":argument.source_id,"text":argument.quote})
        if not excerpts:
            return result("information_insuffisante", "information insuffisante", sources=sources)
        return result("answered", "Passages sélectionnés pour éclairer la question ; vérifier leur suffisance et leur applicabilité.", excerpts=excerpts, sources=sources)
    except LLMConfigurationError:
        return result("erreur_configuration", "Le fournisseur LLM n'est pas configuré correctement.", sources=sources)
    except LLMApiError:
        return result("erreur_api", "Le fournisseur LLM est indisponible. Aucun repli silencieux vers le mock.", sources=sources)
    except (LLMResponseError, ValidationError, ValueError):
        return result("reponse_invalide", "Réponse refusée : format ou citations non conformes.", sources=sources)
