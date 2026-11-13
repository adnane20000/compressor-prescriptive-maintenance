"""Graphe LangGraph persistant et API de démarrage/revue, sans boucle autonome."""

import re
import sqlite3
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langsmith import tracing_context

from ..diagnostic import generate_diagnostic
from ..rag import search_documents
from ..storage.decisions import DecisionStore
from .monitoring import monitor_measurements
from .nodes import WorkflowNodes, WorkflowServices
from .state import WorkflowRequest, WorkflowState


def build_workflow(*, checkpointer, services: WorkflowServices):
    """Compiler un graphe borné ; seul le nœud de revue attend une entrée externe."""
    if checkpointer is None:
        raise ValueError("Un checkpointer est obligatoire pour la revue humaine.")
    nodes = WorkflowNodes(services)
    graph = StateGraph(WorkflowState)
    for name in ("monitor_node", "anomaly_router", "retrieval_node", "diagnosis_node", "planning_node", "human_review_node", "finalize_node"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "monitor_node")
    graph.add_edge("monitor_node", "anomaly_router")
    graph.add_conditional_edges("anomaly_router", lambda state: state["route"],
                               {"normal": "finalize_node", "anomaly": "retrieval_node", "insufficient": "planning_node", "error": "planning_node"})
    graph.add_conditional_edges("retrieval_node", lambda state: state["phase"] == "retrieved",
                               {True: "diagnosis_node", False: "planning_node"})
    graph.add_edge("diagnosis_node", "planning_node")
    graph.add_edge("planning_node", "human_review_node")
    graph.add_conditional_edges("human_review_node", lambda state: state["review"] is not None,
                               {True: "finalize_node", False: "human_review_node"})
    graph.add_edge("finalize_node", END)
    return graph.compile(checkpointer=checkpointer, name="metropt_documentary_review")


class WorkflowRunner:
    """API locale : aucun appelant ne peut soumettre une validation dans l'état initial."""
    def __init__(self, graph):
        self.graph = graph

    def _config(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Identifiant de dossier invalide.")
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 30}

    def start(self, request: WorkflowRequest | dict) -> dict:
        request = request if isinstance(request, WorkflowRequest) else WorkflowRequest.model_validate(request)
        run_id = uuid4().hex
        state = WorkflowState(run_id=run_id, request=request.model_dump(mode="json"), phase="started", route=None,
                              ml_event=None, measurements={}, monitoring_summary=None, retrieval=None, diagnostic=None,
                              proposal=None, review=None, review_error=None, decision=None, errors=[], journal=[])
        with tracing_context(enabled=False):
            self.graph.invoke(state, self._config(run_id), durability="sync")
        return self.get_state(run_id)

    def get_state(self, run_id: str) -> dict:
        snapshot = self.graph.get_state(self._config(run_id))
        if not snapshot.values:
            raise KeyError("Dossier introuvable dans les checkpoints locaux.")
        return {**snapshot.values, "pending_nodes": list(snapshot.next),
                "interrupts": [{"id": item.id, "value": item.value} for item in snapshot.interrupts]}

    def review(self, run_id: str, submission: dict) -> dict:
        state = self.get_state(run_id)
        if state["pending_nodes"] != ["human_review_node"] or not state["interrupts"]:
            raise ValueError("Ce dossier n'attend pas de revue humaine.")
        with tracing_context(enabled=False):
            # Lier explicitement la réponse à l'interruption : {} désigne sinon
            # une table vide d'interruptions dans l'API LangGraph, sans reprise.
            resume = {state["interrupts"][0]["id"]: submission}
            self.graph.invoke(Command(resume=resume), self._config(run_id), durability="sync")
        return self.get_state(run_id)

    def retry_finalization(self, run_id: str) -> dict:
        state = self.get_state(run_id)
        if state["pending_nodes"] != ["finalize_node"] or state["interrupts"]:
            raise ValueError("Aucune finalisation en échec à reprendre.")
        with tracing_context(enabled=False):
            self.graph.invoke(None, self._config(run_id), durability="sync")
        return self.get_state(run_id)


@contextmanager
def open_workflow(*, directory="data/local/workflow", model_path="models/isolation_forest/baseline.joblib",
                  index_directory="data/local/rag", documents_directory="documents", provider=None, services=None):
    """Ouvrir les checkpoints SQLite ; fermer les connexions après usage.

    Réouvrir le même dossier permet de retrouver une interruption après redémarrage.
    Les modèles Joblib et bases locales doivent être de confiance. Aucun secret ou
    objet client n'est sérialisé dans l'état ; le traçage distant est désactivé.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if services is None:
        services = WorkflowServices(
            monitor=partial(monitor_measurements, model_path=str(model_path)),
            retrieve=partial(search_documents, index_directory=index_directory),
            diagnose=partial(generate_diagnostic, documents_directory=documents_directory, provider=provider),
            decisions=DecisionStore(directory / "decisions.sqlite"),
        )
    connection = sqlite3.connect(directory / "checkpoints.sqlite", check_same_thread=False, timeout=10)
    # L'état ne contient que des primitives : aucun fallback pickle ou import d'objet.
    serializer = JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=[], allowed_msgpack_modules=[])
    try:
        saver = SqliteSaver(connection, serde=serializer)
        yield WorkflowRunner(build_workflow(checkpointer=saver, services=services))
    finally:
        connection.close()
