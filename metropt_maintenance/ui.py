"""Exploration locale et rejeu pas à pas de mesures historiques."""

from pathlib import Path
import json
import os
import sqlite3

import pandas as pd
import streamlit as st

from .ingestion.exploration import profile_csv
from .ingestion.replay import iter_replay
from .ingestion.report import aggregate_charts, build_figures, write_report
from .ingestion.schema import CSV_NAME, PLOT_COLUMNS, SIGNALS


def _diagnostic_panel() -> None:
    """Relier explicitement une alerte ML aux passages courants du RAG."""
    from .diagnostic import generate_diagnostic

    st.subheader("Diagnostic préliminaire — ML et documents")
    provider_name = os.getenv("LLM_PROVIDER", "mock")
    st.caption(
        "Le modèle ne commande aucun équipement. Les contrôles restent documentaires "
        "ou analytiques et nécessitent une validation humaine. "
        "En mode API, les mesures et les extraits sélectionnés sont envoyés au fournisseur configuré."
    )
    st.write(f"Fournisseur configuré : {provider_name}")
    if provider_name == "mock":
        st.info("SIMULATION MOCK — aucun appel à un service LLM.")
    if st.button("Utiliser la première alerte ML disponible"):
        path = Path("data/local/ml_baseline/agent_alerts.jsonl")
        try:
            with path.open(encoding="utf-8") as source:
                line = next(source, "")
            if not line:
                raise ValueError("Aucune alerte ML disponible dans ce fichier.")
            event = json.loads(line)
            st.session_state.diagnostic_event_json = json.dumps(event, ensure_ascii=False, indent=2)
            st.session_state.diagnostic_measures_json = json.dumps(event.get("observed_signals", {}), ensure_ascii=False, indent=2)
            st.session_state.pop("diagnostic_result", None)
        except (OSError, ValueError) as error:
            st.error(str(error))
    event_json = st.text_area("Événement d'anomalie ML (JSON)", value="", key="diagnostic_event_json", height=180)
    measures_json = st.text_area("Mesures disponibles (JSON)", value="{}", key="diagnostic_measures_json", height=160)
    rag_result = st.session_state.get("rag_result")
    passages = rag_result.passages if rag_result is not None else []
    diagnostic_identity = (event_json, measures_json, repr(passages),
                           st.session_state.get("diagnostic_documents_root"), provider_name,
                           os.getenv("LLM_MODEL"), os.getenv("LLM_BASE_URL"))
    if st.session_state.get("diagnostic_identity") != diagnostic_identity:
        st.session_state.pop("diagnostic_result", None)
    st.caption(f"{len(passages)} passage(s) de la dernière recherche RAG utilisables ; chaque citation sera revérifiée dans le PDF.")
    if st.button("Générer le diagnostic préliminaire"):
        try:
            with st.spinner("Validation des données, des sources et du diagnostic…"):
                result = generate_diagnostic(json.loads(event_json), json.loads(measures_json), passages,
                                             documents_directory=st.session_state.get("diagnostic_documents_root", "documents"))
            st.session_state.diagnostic_result = result
            st.session_state.diagnostic_identity = diagnostic_identity
        except (OSError, ValueError) as error:
            st.session_state.pop("diagnostic_result", None)
            st.error(str(error))
    result = st.session_state.get("diagnostic_result")
    if result is not None:
        if result.simulation:
            st.warning("SIMULATION MOCK — ce résultat ne constitue pas un diagnostic causal issu d'un LLM.")
        st.write(f"Statut : {result.statut} — {result.etat_generation} ; incertitude {result.niveau_incertitude}.")
        st.caption("Ce résultat n'autorise aucune intervention. Aucun contrôle physique n'est déclenché.")
        st.json(result.model_dump(mode="json"))


def _review_status(state: dict) -> str:
    """Afficher la décision persistée ; une attente ou une erreur ne vaut jamais approbation."""
    decision = state.get("decision")
    if decision:
        return {
            "approved_documentary_proposal": "Proposition documentaire approuvée",
            "rejected": "Proposition refusée",
            "revision_requested": "Révision demandée — proposition non approuvée",
            "information_requested": "Informations demandées — proposition non approuvée",
            "closed_without_alert": "Clôturé sans alerte — aucune validation humaine",
        }[decision["outcome"]]
    if state.get("review"):
        return "Décision humaine reçue — enregistrement en attente"
    if state.get("proposal"):
        return "En attente de validation humaine"
    return "Clôture en attente d'enregistrement"


def _render_review_card(state: dict) -> None:
    """Présenter les preuves et leur statut, avec le texte documentaire rendu comme donnée."""
    st.subheader("Fiche de supervision humaine")
    st.text(f"Workflow : {state['run_id']}")
    st.text(f"Statut de validation : {_review_status(state)}")
    diagnostic = state.get("diagnostic") or {}
    proposal = state.get("proposal") or {}
    event = state.get("ml_event") or {}
    if state.get("request", {}).get("data_context") == "synthetic_test":
        st.info("Mesures SYNTHÉTIQUES déclarées pour ce dossier de test.")
    if diagnostic.get("simulation"):
        st.warning("SIMULATION MOCK — le diagnostic est simulé, sans appel à un LLM.")
    st.caption("L'approbation porte sur la préparation documentaire. Aucun diagnostic confirmé, aucune autorisation ni commande d'intervention physique.")

    st.markdown("#### Mesures et anomalies observées")
    if event:
        st.text(f"Minute analysée : {event.get('window_start') or 'non renseignée'} ; disponible à {event.get('available_at', 'non renseigné')} (horodatages historiques).")
        if event.get("status") == "scored":
            observed = "Anomalie statistique détectée" if event["is_anomaly"] else "Aucune anomalie statistique détectée sur cette fenêtre"
            st.text(f"{observed} ; score {event['anomaly_score']:.6f} ; seuil {event['threshold']:.6f}.")
        else:
            st.text("Fenêtre non scorée : données ou historique insuffisants.")
        if event.get("mode_interpreted"):
            st.text(f"Mode interprété : {event['mode_interpreted']} ; mode mixte : {event.get('mode_mixed')}.")
    else:
        st.text("Résultat ML indisponible. Aucune normalité ne peut être conclue.")
    st.caption("Le score d'anomalie n'est pas une probabilité de panne.")
    units = {signal.name: signal.unit for signal in SIGNALS}
    measurements = []
    for name, value in state.get("measurements", {}).items():
        measure = value if isinstance(value, dict) else {"value": value}
        measurements.append({"Signal": name, "Valeur": measure.get("value"),
                             "Unité": measure.get("unit", units.get(name)), "Agrégation": measure.get("aggregation", "non précisée")})
    if measurements:
        st.dataframe(pd.DataFrame(measurements), hide_index=True)
    else:
        st.text("Aucune mesure exploitable disponible.")

    st.markdown("#### Hypothèses du diagnostic préliminaire")
    hypotheses = diagnostic.get("hypotheses_de_defaillance", [])
    for hypothesis in hypotheses:
        st.text(f"{hypothesis['label']} — hypothèse non confirmée ; source {hypothesis['source_id']} ; applicabilité {hypothesis['equipment_relevance']}.")
    if not hypotheses:
        st.text("Aucune hypothèse documentée disponible. Une anomalie ne confirme aucune cause.")
    if diagnostic:
        st.text(f"Incertitude : {diagnostic['niveau_incertitude']}.")
    if diagnostic.get("erreur"):
        st.text(diagnostic["erreur"])

    st.markdown("#### Sources documentaires")
    sources = diagnostic.get("sources", [])
    for source in sources:
        st.text(f"{source['id']} — {source['file_name']}, p. {source['page']} (page physique) ; statut {source['authority']} ; portée {source['applicability']}.")
        with st.expander(f"Passage {source['id']} et provenance"):
            st.text(source["passage"])
            st.text(f"SHA-256 : {source['pdf_sha256']} ; équipement : {source['equipment_reference'] or 'applicabilité spécifique non établie'}.")
    if not sources:
        st.text("Aucune source documentaire vérifiée disponible pour le diagnostic.")
    for argument in diagnostic.get("arguments_documentaires", []):
        st.text(f"Argument {argument['id']} — source {argument['source_id']} : {argument['excerpt']}")

    st.markdown("#### Contrôles recommandés")
    for action in proposal.get("actions", []):
        st.text(action["description"])
    if not proposal.get("actions"):
        st.text("Aucun contrôle proposé.")
    st.caption("Contrôles documentaires ou analyses de données uniquement ; aucune procédure de terrain générée.")

    st.markdown("#### Informations manquantes")
    missing = list(dict.fromkeys(proposal.get("missing_information", []) + [item["description"] for item in diagnostic.get("informations_manquantes", [])]))
    for description in missing:
        st.text(description)
    if not missing:
        st.text("Aucune information manquante consignée ; la suffisance du dossier reste à examiner.")
    if proposal:
        st.text(f"Référence de la proposition soumise : {proposal['proposal_id']}")
        if not proposal["approval_allowed"]:
            st.warning("Approbation bloquée : dossier insuffisant ou traitement en erreur. Un refus ou une demande de révision reste possible.")

    decision = state.get("decision")
    if decision and decision.get("review"):
        receipt = decision["review"]
        st.markdown("#### Décision enregistrée")
        st.text(f"Identifiant du validateur : {decision.get('validator_id') or receipt['submission']['reviewer']}")
        st.text(f"Horodatage de la revue (UTC) : {receipt['recorded_at_utc']}")
        st.text(f"Enregistrement de la décision (UTC) : {decision['recorded_at_utc']}")
        st.text(f"Commentaires : {receipt['submission']['comment']}")


def _workflow_panel(csv_path: Path) -> None:
    """Démarrer un dossier borné et soumettre une revue explicite au checkpoint."""
    from .orchestration import open_workflow
    from .orchestration.monitoring import load_measurement_window
    from .orchestration.nodes import WorkflowPersistenceError

    st.subheader("Workflow LangGraph — revue humaine persistante")
    st.caption("La dernière minute close est analysée avec son contexte passé. Fournir au moins six minutes closes et une mesure suivante. Les checkpoints et décisions restent locaux ; aucune commande d'équipement.")
    model_path = st.text_input("Modèle local de confiance du workflow", value="models/isolation_forest/baseline.joblib")
    start = st.text_input("Début du contexte du workflow", value="2020-03-01 04:08:00")
    end = st.text_input("Fin des mesures du workflow", value="2020-03-01 04:15:30")
    context = st.selectbox("Origine déclarée des mesures du workflow", ["Historique rejoué", "Synthétique (test)"])
    question = st.text_input("Question documentaire du workflow", value="Quels documents décrivent Oil_temperature, Motor_current, TP2 et TP3 ?")
    if st.button("Démarrer le workflow sur cette fenêtre"):
        st.session_state.pop("workflow_active_id", None)
        try:
            with st.spinner("Analyse ML et parcours du graphe jusqu'à la clôture ou la revue…"):
                request = load_measurement_window(csv_path, start, end, data_context="historical_replay" if context == "Historique rejoué" else "synthetic_test", question=question)
                with open_workflow(model_path=model_path) as runner:
                    state = runner.start(request)
            st.session_state.workflow_active_id = state["run_id"]
        except WorkflowPersistenceError as error:
            st.session_state.workflow_active_id = error.run_id
            st.error(str(error))
        except (OSError, ValueError, sqlite3.Error) as error:
            st.error(str(error))
    reopen = st.text_input("Identifiant du dossier à rouvrir", value="")
    if st.button("Rouvrir le dossier enregistré"):
        try:
            with open_workflow(model_path=model_path) as runner:
                runner.get_state(reopen)
            st.session_state.workflow_active_id = reopen
        except (OSError, ValueError, KeyError, sqlite3.Error) as error:
            st.error(str(error))
    run_id = st.session_state.get("workflow_active_id")
    if not run_id:
        return
    try:
        with open_workflow(model_path=model_path) as runner:
            state = runner.get_state(run_id)
            _render_review_card(state)
            if state["interrupts"]:
                st.markdown("#### Votre décision")
                form_key = f"{run_id}_{state['proposal']['proposal_id']}"
                reviewer = st.text_input("Identifiant du validateur", max_chars=200, key=f"workflow_reviewer_{form_key}")
                comment = st.text_area("Commentaire de la revue", max_chars=2000, key=f"workflow_comment_{form_key}")
                acknowledged = st.checkbox("Je confirme cette revue documentaire ; aucune intervention physique n'est autorisée.", key=f"workflow_ack_{form_key}")
                ready = len(reviewer.strip()) >= 2 and bool(comment.strip()) and acknowledged
                approve, reject, revise = st.columns(3)
                choice = None
                if approve.button("Approuver", disabled=not (ready and state["proposal"]["approval_allowed"])):
                    choice = "approve"
                if reject.button("Refuser", disabled=not ready):
                    choice = "reject"
                if revise.button("Demander une révision", disabled=not ready):
                    choice = "request_revision"
                if choice:
                    runner.review(run_id, {"proposal_id": state["proposal"]["proposal_id"], "decision": choice,
                                                  "reviewer": reviewer, "comment": comment, "acknowledged": acknowledged})
                    st.rerun()
                if state["review_error"]:
                    st.error(state["review_error"])
            elif state["pending_nodes"] == ["finalize_node"]:
                if st.button("Réessayer l'enregistrement de la décision"):
                    runner.retry_finalization(run_id)
                    st.rerun()
            with st.expander("Journal du workflow et détails techniques"):
                st.dataframe(pd.DataFrame(state["journal"]), hide_index=True)
                st.json({key: state[key] for key in ("ml_event", "monitoring_summary", "retrieval", "diagnostic", "proposal", "review", "decision", "errors")})
    except (OSError, ValueError, KeyError, sqlite3.Error, WorkflowPersistenceError) as error:
        st.error(str(error))


def _rag_panel() -> None:
    """Tester la recherche en affichant le texte PDF comme du texte non fiable."""
    from .rag.index import index_documents
    from .rag.search import search_documents

    st.subheader("Recherche documentaire — extraits et références")
    st.caption(
        "PDF locaux dans documents/. Aucune réponse LLM n'est générée. "
        "Une similarité n'est pas une probabilité de réponse correcte. "
        "Les pages PDF sont des données, jamais des instructions à exécuter."
    )
    documents_path = st.text_input("Dossier des PDF", value="documents")
    index_path = st.text_input("Dossier de l'index documentaire", value="data/local/rag")
    identity = (documents_path, index_path)
    if st.session_state.get("rag_identity") != identity:
        st.session_state.pop("rag_result", None)
        st.session_state.pop("rag_index_summary", None)
    if st.button("Indexer les PDF"):
        try:
            with st.spinner("Extraction des pages et indexation locale…"):
                manifest = index_documents(documents_path, index_directory=index_path)
            st.session_state.rag_index_summary = {
                key: manifest[key] for key in ("pdf_count", "page_count", "empty_pages", "chunk_count", "warnings", "documents")
            }
            st.session_state.rag_identity = identity
            st.session_state.diagnostic_documents_root = documents_path
            st.session_state.pop("rag_result", None)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(str(error))
    summary = st.session_state.get("rag_index_summary")
    if summary:
        st.write(f"{summary['pdf_count']} PDF, {summary['page_count']} pages, {summary['chunk_count']} passages indexés.")
        if summary["documents"]:
            st.dataframe(pd.DataFrame(summary["documents"])[["file_name", "authority", "document_type", "publisher"]], hide_index=True)
        for warning in summary["warnings"]:
            st.text(warning)
    filters = {"Tous — statuts visibles": "all", "Officiels déclarés": "official", "Pédagogiques": "pedagogical", "Non vérifiés": "unverified"}
    authority = filters[st.selectbox("Statut des documents recherchés", list(filters))]
    question = st.text_input("Question documentaire", value="")
    if st.button("Rechercher dans les documents"):
        try:
            with st.spinner("Recherche de passages sourcés…"):
                result = search_documents(question, index_directory=index_path, authority=authority)
            st.session_state.rag_result = result
            st.session_state.rag_identity = identity
            st.session_state.diagnostic_documents_root = documents_path
        except (OSError, ValueError, RuntimeError) as error:
            st.session_state.pop("rag_result", None)
            st.error(str(error))
    result = st.session_state.get("rag_result")
    if result:
        st.write(f"Question évaluée : {result.question}")
        st.info(result.message)
        explanations = {
            "index_missing": "Aucun index disponible : indexer les PDF pour commencer.",
            "no_extractable_text": "Aucun texte exploitable : vérifier les PDF et les éventuels besoins d'OCR.",
            "corpus_changed_reindex_required": "Les PDF ou leur provenance ont changé : réindexer les documents.",
            "question_without_searchable_terms": "Préciser la question avec le sujet recherché.",
            "requested_source_missing": "L'éditeur demandé n'est pas présent dans le corpus déclaré.",
            "unsafe_intervention_request": "Le prototype refuse de fournir un mode opératoire pour cette demande à risque.",
            "operational_source_missing": "Aucune source technique officielle adaptée à cette demande opérationnelle.",
            "equipment_specific_source_missing": "Aucune source liée au compresseur MetroPT-3 pour cette demande ; les ouvrages généraux ne suffisent pas.",
            "no_relevant_passage": "Aucun passage ne satisfait les critères de recherche.",
            "retrieval_only": "Vérifier les extraits dans leur document d'origine ; leur suffisance n'est pas certifiée.",
        }
        st.caption(explanations[result.reason])
        for warning in result.warnings:
            st.text(warning)
        for group, passages in (("Passages retrouvés", result.passages), ("Contexte associé — insuffisant pour cette demande", result.related_passages)):
            if passages:
                st.write(group)
            for hit in passages:
                st.text(f"{hit.reference} | {hit.authority} | {hit.document_type} | {hit.applicability} | similarité {hit.similarity:.3f}")
                st.text(hit.text)
                st.caption(f"Provenance déclarée : {hit.publisher or 'non fournie'} ; empreinte {hit.pdf_sha256}.")
        if result.reason == "corpus_changed_reindex_required":
            st.caption("Le corpus ou son registre a changé : réindexer les PDF.")


def _ml_panel(path: Path, identity: tuple | None) -> None:
    """Exécuter la baseline explicitement et afficher uniquement les résultats réduits."""
    from .ml.model import ModelConfig
    from .ml.report import score_figure
    from .ml.workflow import run_baseline

    st.subheader("Détection d'anomalies — baseline Isolation Forest")
    st.caption(
        "Fenêtres historiques d'une minute et variables causales. Le score mesure "
        "l'atypie statistique, jamais une probabilité de panne ou une cause. "
        "Le préfixe d'exploration ne limite pas l'entraînement ML."
    )
    train_end = st.text_input("Fin de l'entraînement ML (exclusive pour les mesures brutes)", value="2020-02-22 00:00:00")
    calibration_end = st.text_input("Fin de la calibration ML (exclusive pour les mesures brutes)", value="2020-03-01 00:00:00")
    ml_identity = (identity, train_end, calibration_end)
    if st.session_state.get("ml_identity") != ml_identity:
        st.session_state.pop("ml_result", None)
    if st.button("Entraîner et évaluer Isolation Forest", disabled=identity is None):
        try:
            config = ModelConfig(train_end=train_end, calibration_end=calibration_end)
            with st.spinner("Prétraitement causal, entraînement, calibration et test…"):
                result = run_baseline(path, config=config)
            st.session_state.ml_result = result
            st.session_state.ml_identity = ml_identity
        except (OSError, ValueError) as error:
            st.error(str(error))
    if "ml_result" in st.session_state:
        result = st.session_state.ml_result
        metrics = result.metrics
        st.write(f"Seuil de calibration : {metrics['threshold']:.6f} ; modèle {metrics['model_id']}.")
        st.dataframe(pd.DataFrame(metrics["by_split"]), hide_index=True)
        st.dataframe(pd.DataFrame(metrics["by_mode"]), hide_index=True)
        test = next(entry for entry in metrics["by_split"] if entry["split"] == "test")
        st.write(f"Test : {test['alert_windows']:,} alertes sur {test['scored_windows']:,} fenêtres scorées.")
        st.caption(metrics["limitations"])
        comparisons = pd.DataFrame(metrics["documented_period_comparisons"])
        st.dataframe(comparisons, hide_index=True)
        hourly = pd.read_csv(result.report_path.with_name("score_plot_data.csv"), parse_dates=["timestamp"])
        st.plotly_chart(score_figure(hourly, metrics["threshold"], metrics["protocol"]), use_container_width=True)
        st.download_button("Télécharger le rapport ML", result.report_path.read_bytes(), file_name="metropt3_ml.html", mime="text/html")


def _clear_replay() -> None:
    """Fermer le lecteur CSV lorsqu'une session de rejeu est réinitialisée."""
    iterator = st.session_state.pop("replay_iterator", None)
    if iterator is not None:
        iterator.close()
    st.session_state.pop("replay_rows", None)
    st.session_state.pop("replay_identity", None)


def _replay_panel(path: Path, identity: tuple | None) -> None:
    """Piloter le même générateur en flux que la CLI, sans boucle bloquante."""
    st.subheader("Rejeu chronologique — simulation historique")
    st.caption(
        "Mode pas à pas : les boutons font avancer les mesures, sans attente automatique. "
        "Le délai proposé vaut l'écart entre dates source divisé par la vitesse. "
        "Le rejeu utilise le CSV indépendamment du préfixe d'analyse. "
        "Ce n'est pas un flux de capteurs en temps réel."
    )
    speed = st.number_input("Facteur de vitesse pour les délais proposés", min_value=0.1, value=60.0)
    batch = st.selectbox("Mesures par pas", [1, 10, 100])
    start = st.text_input("Début du rejeu (facultatif, AAAA-MM-JJ HH:MM:SS)")
    end = st.text_input("Fin du rejeu (facultatif, AAAA-MM-JJ HH:MM:SS)")
    replay_identity = (identity, speed, start, end)
    if st.session_state.get("replay_identity") != replay_identity:
        _clear_replay()
    if st.button("Initialiser le rejeu", disabled=identity is None):
        _clear_replay()
        st.session_state.replay_iterator = iter_replay(
            path, speed=speed, chunksize=1000, start=start or None, end=end or None,
        )
        st.session_state.replay_identity = replay_identity
        st.session_state.replay_rows = []
    if st.button("Avancer dans le rejeu", disabled="replay_iterator" not in st.session_state):
        try:
            for _ in range(batch):
                event = next(st.session_state.replay_iterator)
                st.session_state.replay_rows.append({
                    "timestamp": event.timestamp, **event.values, "mode_interprete": event.mode,
                    "ecart_source_s": event.source_delta_seconds, "delai_propose_s": event.delay_seconds,
                })
                st.session_state.replay_rows = st.session_state.replay_rows[-1000:]
        except StopIteration:
            st.session_state.pop("replay_iterator").close()
            st.success("Fin du rejeu pour la plage demandée.")
        except (OSError, ValueError) as error:
            _clear_replay()
            st.error(str(error))
    rows = st.session_state.get("replay_rows", [])
    if rows:
        last = rows[-1]
        st.write(f"Horodatage historique : {last['timestamp']} — {last['mode_interprete']}")
        st.caption(f"Écart source : {last['ecart_source_s']} s ; délai proposé : {last['delai_propose_s']:.3f} s.")
        st.dataframe(pd.DataFrame(rows)[["timestamp", *PLOT_COLUMNS, "mode_interprete"]], hide_index=True)
        st.caption("Affichage limité aux 1 000 dernières mesures rejouées.")


def main() -> None:
    """Afficher l'exploration, sans charger un grand CSV au simple démarrage."""
    st.set_page_config(page_title="MetroPT — Maintenance", layout="wide")
    st.title("MetroPT — Maintenance prescriptive")
    st.caption("Prototype académique • Compresseur industriel • MetroPT-3")

    if os.getenv("LLM_PROVIDER", "mock") == "mock":
        st.info("Mode démonstration — aucune clé API nécessaire. Les mesures chargées sont historiques.")
    else:
        st.info("Fournisseur LLM externe configuré pour le diagnostic. Les mesures chargées sont historiques.")
    st.warning(
        "Une anomalie ne constitue pas un diagnostic confirmé. "
        "Tout futur plan de maintenance nécessitera une validation humaine. "
        "Cette application ne contrôle aucun équipement."
    )
    st.subheader("Acquisition et exploration — MetroPT-3")
    st.write("Téléchargez le CSV depuis UCI ou utilisez la commande de téléchargement du README.")
    st.caption("Pour les gros fichiers, indiquez un chemin local : la lecture se fait par blocs, sans téléversement en mémoire.")
    path = Path(st.text_input("Chemin du CSV local", value=str(Path("data/raw") / CSV_NAME))).expanduser()
    try:
        stat = path.stat()
        identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns) if path.is_file() else None
    except OSError:
        identity = None
    if st.session_state.get("analysis_identity") != identity:
        st.session_state.pop("analysis", None)
    limited = st.checkbox("Limiter l'analyse à un préfixe du fichier", value=False)
    max_rows = st.number_input("Nombre maximal de lignes (mode limité)", min_value=1, value=10_000, disabled=not limited)
    if st.button("Analyser le CSV"):
        try:
            with st.spinner("Analyse par blocs et création du rapport…"):
                profile = profile_csv(path, max_rows=int(max_rows) if limited else None)
                charts = aggregate_charts(path, profile)
                report = write_report(profile, charts, Path("data/local/exploration"))
            st.session_state.analysis = (profile, charts, str(report))
            st.session_state.analysis_identity = identity
        except (OSError, ValueError) as error:
            st.error(str(error))
    if "analysis" in st.session_state:
        profile, charts, report_path = st.session_state.analysis
        scope = "fichier complet" if profile["scope"] == "entire_file" else "PRÉFIXE LIMITÉ"
        st.success(f"{profile['rows_analyzed']:,} mesures analysées — {scope}")
        st.write(f"Période observée : {profile['date_min']} → {profile['date_max']} ; fuseau non documenté.")
        st.write(f"Doublons de mesures : {profile['duplicate_measurement_rows']} ; horodatages répétés : {profile['duplicate_timestamps']}.")
        st.dataframe(pd.DataFrame(profile["statistics"]), hide_index=True)
        st.dataframe(pd.DataFrame({"manquantes": profile["missing_raw"], "invalides": profile["invalid_values"]}).fillna(0))
        st.caption(f"Graphiques agrégés en {len(charts)} fenêtres ; min/max/moyenne, aucune imputation.")
        for figure in build_figures(profile, charts):
            st.plotly_chart(figure, use_container_width=True)
        st.caption(profile["mode_rules"])
        st.download_button("Télécharger le rapport HTML", Path(report_path).read_bytes(), file_name="metropt3_exploration.html", mime="text/html")
    _ml_panel(path, identity)
    _rag_panel()
    _diagnostic_panel()
    _workflow_panel(path)
    _replay_panel(path, identity)
