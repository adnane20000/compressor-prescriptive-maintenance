"""Interface française en cinq pages : mesures, ML, documents, diagnostic et revue humaine."""

from pathlib import Path
from html import escape
import json
import os
import sqlite3

import pandas as pd
import streamlit as st

from .ingestion.schema import CSV_NAME, SIGNALS
from .presentation import (anomaly_figure, create_synthetic_demo, file_identity, mode_figure,
                           preview_measurements, read_alerts, read_json, read_score_view,
                           sensor_figure, workflow_figure)

PAGES = ("Dashboard", "Détection d'anomalies", "Assistant RAG", "Diagnostic agentique", "Gestion des interventions")
OUTCOMES = {"approved_documentary_proposal":"Approuvée · documentaire", "rejected":"Refusée",
            "revision_requested":"Révision demandée", "information_requested":"Informations demandées",
            "closed_without_alert":"Clôture sans alerte"}
AUTHORITIES = {"official":"Officiel déclaré", "pedagogical":"Pédagogique", "unverified":"Non vérifié"}
APPLICABILITIES = {"general_reference":"Référence générale", "equipment_specific":"Spécifique à l'équipement", "not_verified":"Applicabilité non vérifiée"}
WORKFLOW_DIRECTORY = "data/local/workflow"

CSS = """
<style>
.stApp {background:#F5F7FB}
[data-testid="stMainBlockContainer"] {padding-top:4.7rem!important;max-width:1500px;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#101D32;color:#D9E3F2;border-right:0}
[data-testid="stSidebar"] * {color:#D9E3F2}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {font-size:.88rem}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.42rem}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.7rem .75rem;border-radius:8px;width:100%;background:transparent}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#183849;border-left:3px solid #36C5C4}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {color:#18243A}
[data-testid="stSidebar"] button {color:#18243A}
h1 {font-size:2.1rem!important;letter-spacing:-.045em;font-weight:720!important;padding-bottom:.25rem!important}
h2 {font-size:1.3rem!important;letter-spacing:-.025em}
h3 {font-size:1.05rem!important}
h4 {font-size:.98rem!important}
[data-testid="stMetric"] {background:#fff;padding:1.1rem 1.25rem;border:1px solid #E5EAF1;border-radius:12px;min-height:112px}
[data-testid="stMetricLabel"] {color:#69778D;font-size:.82rem}
[data-testid="stMetricValue"] {font-size:1.8rem!important;letter-spacing:-.035em}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#E5EAF1!important;border-radius:12px!important;background:#fff}
[data-testid="stPlotlyChart"] {border-radius:10px;overflow:hidden}
.stButton>button {border-radius:8px;font-weight:600}
.stButton>button[kind="primary"] {background:#078B92;border-color:#078B92;color:white}
[data-testid="stExpander"] {border-color:#E2E8F0;border-radius:10px}
.eyebrow {color:#7D8CA2;font-size:.71rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin:0 0 .15rem}
.brand {display:flex;align-items:center;gap:12px;margin:12px 0 28px}
.brandmark {width:40px;height:40px;display:grid;place-items:center;background:#078B92;border-radius:10px;color:white;font-weight:800;font-size:23px}
.brandname {font-weight:750;font-size:23px;letter-spacing:-.04em;color:#fff}
.brandtag {font-size:10px;color:#8FA2BE;letter-spacing:.12em;text-transform:uppercase}
.source-strip {background:#E9F3F3;border:1px solid #D2E9E8;padding:11px 15px;border-radius:9px;font-size:.84rem;color:#32666B;margin-bottom:17px}
.status-pill {display:inline-block;font-size:.73rem;font-weight:650;padding:5px 10px;border-radius:20px;background:#E4F3EF;color:#166851}
.status-pill.orange {background:#FFF0DB;color:#9B5B0C}
.caption-label {font-size:.79rem;color:#6C7C91;margin-bottom:8px}
[data-testid="stHeader"] {background:rgba(245,247,251,.94)}
[data-testid="stAppDeployButton"] {display:none}
@media(max-width:700px) {
[data-testid="stMainBlockContainer"] {padding:4.7rem 1rem 1.2rem}
h1 {font-size:1.7rem!important}
[data-testid="stMetricValue"] {font-size:1.5rem!important}
}
</style>
"""


def _plot(figure, key=None):
    st.plotly_chart(figure, width="stretch", config={"displayModeBar":False, "scrollZoom":False}, key=key)


def _number(value, decimals=0):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def _header(title, subtitle, section):
    st.markdown(f'<div class="eyebrow">METROPT / {escape(section)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def _go_page(page):
    st.session_state.nav_page = page


def _provider():
    from .diagnostic.providers import MockProvider, provider_from_env
    return MockProvider() if st.session_state.get("interface_mode", "Démonstration sans API") == "Démonstration sans API" else provider_from_env()


def _open_workflow(*, generate=False):
    from .orchestration import open_workflow
    rag_config = st.session_state.get("rag_config", {})
    return open_workflow(directory=WORKFLOW_DIRECTORY,
                         model_path=st.session_state.get("model_path", "models/isolation_forest/baseline.joblib"),
                         documents_directory=rag_config.get("documents","documents"),
                         index_directory=rag_config.get("index","data/local/rag"),
                         provider=_provider() if generate else None)


def _clear_replay():
    iterator = st.session_state.pop("replay_iterator", None)
    if iterator is not None:
        iterator.close()
    for key in ("replay_rows", "replay_identity"):
        st.session_state.pop(key, None)


def _sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brandmark">M</div><div><div class="brandname">MetroPT</div><div class="brandtag">Maintenance 4.0 · Prototype</div></div></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        st.divider()
        st.selectbox("Mode de l'assistant", ["Démonstration sans API", "Fournisseur configuré"], key="interface_mode")
        if st.session_state.interface_mode == "Démonstration sans API":
            st.caption("DÉMONSTRATION · aucune clé API nécessaire. Réponses mock explicitement simulées.")
        else:
            st.caption("Les mesures et extraits sélectionnés sont envoyés au fournisseur configuré lors d'une génération.")
        with st.expander("Source des mesures", expanded=False):
            st.caption("Source du Dashboard et des nouvelles analyses agentiques.")
            if "csv_path" not in st.session_state:
                st.session_state.csv_path = str(Path("data/raw") / CSV_NAME)
            if st.button("Créer un exemple synthétique", width="stretch"):
                st.session_state.csv_path = str(create_synthetic_demo())
                st.session_state.origin_label = "Synthétique (test)"
            st.text_input("Chemin du CSV local", key="csv_path")
            st.selectbox("Origine déclarée des mesures", ["Historique rejoué", "Synthétique (test)"], key="origin_label")
        st.caption("REJEU HISTORIQUE · Aucun flux temps réel ni contrôle d'équipement.")
        st.divider()
        st.caption("PROJET ACADÉMIQUE\n\nUne anomalie n'est pas un diagnostic confirmé. Validation humaine obligatoire avant toute proposition approuvée.")
    path = Path(st.session_state.csv_path).expanduser()
    try:
        identity = file_identity(path)
    except OSError:
        identity = None
    if st.session_state.get("source_identity") != identity:
        _clear_replay()
        st.session_state.pop("analysis", None)
        st.session_state.pop("replay_settings", None)
        for key in ("replay_speed_input","replay_start_input","replay_end_input","replay_batch_input"):
            st.session_state.pop(key,None)
        st.session_state.source_identity = identity
    return page, path, identity


@st.cache_data(show_spinner=False, max_entries=6)
def _preview(path, identity):
    return preview_measurements(path)


def _context_notice(path, *, count=None):
    if st.session_state.origin_label == "Synthétique (test)":
        st.warning("DONNÉES SYNTHÉTIQUES déclarées — exemple pédagogique, aucune mesure industrielle réelle.")
    else:
        st.markdown('<div class="source-strip">↺ &nbsp; Données historiques rejouées · Jamais une acquisition en temps réel</div>', unsafe_allow_html=True)
    suffix = f" · {count} mesures affichées" if count is not None else ""
    st.caption(f"Source sélectionnée : {path.name}{suffix}. La déclaration d'origine ne certifie pas la provenance du fichier.")


def _replay_controls(path, identity):
    settings = st.session_state.get("replay_settings", {"speed":60.,"start":"","end":"","batch":1})
    with st.expander("Paramètres du rejeu"):
        a,b,c = st.columns(3)
        speed = a.number_input("Facteur de vitesse pour les délais proposés", min_value=.1, value=settings["speed"], key="replay_speed_input")
        start = b.text_input("Début du rejeu (facultatif, AAAA-MM-JJ HH:MM:SS)", value=settings["start"], key="replay_start_input")
        end = c.text_input("Fin du rejeu (facultatif, AAAA-MM-JJ HH:MM:SS)", value=settings["end"], key="replay_end_input")
        st.caption("Pas à pas, sans attente automatique. La vitesse calcule seulement un délai proposé.")
    a,b,c = st.columns([1.1,1.3,2])
    batch = a.selectbox("Mesures par pas", [1,10,100], index=[1,10,100].index(settings["batch"]), key="replay_batch_input")
    st.session_state.replay_settings={"speed":speed,"start":start,"end":end,"batch":batch}
    replay_identity = (identity,speed,start,end)
    if st.session_state.get("replay_identity") not in (None,replay_identity):
        _clear_replay()
    with b:
        st.caption("REJEU CHRONOLOGIQUE")
        if st.button("Initialiser le rejeu", disabled=identity is None, width="stretch"):
            from .ingestion.replay import iter_replay
            _clear_replay()
            st.session_state.replay_iterator = iter_replay(path, speed=speed, chunksize=1000, start=start or None, end=end or None)
            st.session_state.replay_identity = replay_identity
            st.session_state.replay_rows = []
    with c:
        st.caption(" ")
        if st.button("Avancer dans le rejeu", disabled="replay_iterator" not in st.session_state, type="primary", width="stretch"):
            try:
                for _ in range(batch):
                    event = next(st.session_state.replay_iterator)
                    st.session_state.replay_rows.append({"timestamp":event.timestamp, **event.values, "mode_interprete":event.mode,
                                                        "delai_propose_s":event.delay_seconds})
                    st.session_state.replay_rows = st.session_state.replay_rows[-1000:]
            except StopIteration:
                st.session_state.pop("replay_iterator").close()
                st.success("Fin du rejeu pour la plage demandée.")
            except (OSError,ValueError) as error:
                _clear_replay(); st.error(str(error))


def _profile_panel(path):
    with st.expander("Exploration et rapport complet"):
        st.caption("Analyse explicite du CSV par blocs ; le profil global est distinct de la fenêtre affichée.")
        limited = st.checkbox("Limiter l'analyse à un préfixe du fichier", value=True)
        max_rows = st.number_input("Nombre maximal de lignes (mode limité)", min_value=1, value=10_000, disabled=not limited)
        if st.button("Analyser le CSV"):
            from .ingestion.exploration import profile_csv
            from .ingestion.report import aggregate_charts, write_report
            try:
                with st.spinner("Analyse par blocs et création du rapport…"):
                    profile = profile_csv(path, max_rows=int(max_rows) if limited else None)
                    charts = aggregate_charts(path, profile)
                    report = write_report(profile,charts,Path("data/local/exploration"))
                st.session_state.analysis = (profile,charts,str(report))
                st.success(f"{profile['rows_analyzed']:,} mesures analysées — {'PRÉFIXE LIMITÉ' if limited else 'fichier complet'}")
            except (OSError,ValueError) as error:
                st.error(str(error))
        if "analysis" in st.session_state:
            profile,_,report = st.session_state.analysis
            st.dataframe(pd.DataFrame(profile["statistics"]),hide_index=True)
            st.caption(f"Doublons : {profile['duplicate_measurement_rows']} · Horodatages répétés : {profile['duplicate_timestamps']}")
            st.download_button("Télécharger le rapport HTML",Path(report).read_bytes(),file_name="metropt3_exploration.html",mime="text/html")


def _dashboard_page(path, identity):
    _header("Dashboard", "Observer le fonctionnement du compresseur, dans son contexte historique.", "VUE D'ENSEMBLE")
    _replay_controls(path,identity)
    frame = pd.DataFrame()
    if "replay_rows" in st.session_state:
        frame = pd.DataFrame(st.session_state.replay_rows)
        if frame.empty:
            st.info("Rejeu initialisé. Cliquez sur Avancer pour émettre les premières mesures.")
    elif identity:
        try:
            with st.spinner("Lecture de l'aperçu historique…"):
                frame = _preview(str(path),identity)
            st.caption("Aperçu initial : au plus 360 premières mesures. Initialiser le rejeu pour avancer pas à pas.")
        except (OSError,ValueError) as error:
            st.error(str(error))
    else:
        st.info("Aucun CSV disponible. Renseignez un chemin dans Source des mesures ou créez un exemple synthétique clairement identifié.")
    _context_notice(path,count=len(frame))
    if frame.empty:
        for column,label in zip(st.columns(4),["Température d'huile","Pression TP3","Courant moteur","État observé"]):
            column.metric(label,"—")
        _profile_panel(path)
        return
    last = frame.iloc[-1]
    state = str(last["mode_interprete"])
    st.markdown(f'<span class="status-pill">{escape(state.capitalize())}</span>',unsafe_allow_html=True)
    st.caption(f"Dernière mesure affichée : {pd.Timestamp(last['timestamp']).isoformat(sep=' ')} · Mode interprété par règles descriptives, sans diagnostic confirmé.")
    a,b,c,d = st.columns(4)
    def value(name,unit):
        return "—" if pd.isna(last[name]) else f"{_number(last[name],2)} {unit}"
    a.metric("Température d'huile",value("Oil_temperature","°C"))
    b.metric("Pression TP3",value("TP3","bar"))
    c.metric("Courant moteur",value("Motor_current","A"))
    d.metric("Mesures dans la vue",_number(len(frame)))
    left,right = st.columns(2)
    with left,st.container(border=True):
        st.subheader("Température de l'huile")
        _plot(sensor_figure(frame,["Oil_temperature"],unit="°C"),key="dash_temperature")
    with right,st.container(border=True):
        st.subheader("Pressions du système")
        _plot(sensor_figure(frame,["TP2","TP3"],unit="bar"),key="dash_pressure")
    left,right = st.columns([1.4,1])
    with left,st.container(border=True):
        st.subheader("Courant moteur")
        _plot(sensor_figure(frame,["Motor_current"],unit="A"),key="dash_current")
    with right,st.container(border=True):
        st.subheader("Modes de fonctionnement")
        counts = frame.mode_interprete.value_counts().to_dict()
        _plot(mode_figure(counts),key="dash_modes")
        st.caption("Répartition des mesures affichées, pas des durées ni des états de panne.")
    st.caption("Lignes brutes, sans lissage ; les trous de plus de 30 secondes sont séparés sur les courbes. Au plus 1 000 mesures de rejeu conservées.")
    _profile_panel(path)


def _ml_page(path, identity):
    _header("Détection d'anomalies", "Repérer les comportements inhabituels, sans les confondre avec une panne confirmée.", "ISOLATION FOREST")
    with st.expander("Entraînement et paramètres de la baseline"):
        from .ml.model import ModelConfig
        from .ml.workflow import run_baseline
        train = st.text_input("Fin de l'entraînement ML (exclusive pour les mesures brutes)",value="2020-02-22 00:00:00")
        calibration = st.text_input("Fin de la calibration ML (exclusive pour les mesures brutes)",value="2020-03-01 00:00:00")
        st.caption("Traite le CSV sélectionné complet. Entraînement, calibration et test restent chronologiques ; le seuil n'est pas ajusté sur le test.")
        if st.button("Entraîner et évaluer Isolation Forest",disabled=identity is None):
            try:
                with st.spinner("Prétraitement causal, entraînement et évaluation…"):
                    st.session_state.ml_result = run_baseline(path,config=ModelConfig(train_end=train,calibration_end=calibration))
            except (OSError,ValueError) as error:
                st.error(str(error))
    root = Path("data/local/ml_baseline")
    try:
        metrics = read_json(root/"metrics.json")
    except (OSError,ValueError) as error:
        st.error(f"Résultats ML illisibles : {error}"); return
    if not metrics:
        st.info("Aucun résultat ML enregistré. Entraînez la baseline sur votre CSV pour afficher les scores et l'historique.")
        return
    source = metrics.get("provenance",{}).get("csv_path","source non renseignée")
    st.caption(f"Résultats du modèle {metrics['model_id']} · Source : {Path(source).name}. Ces artefacts sont indépendants du CSV actuellement choisi dans la barre latérale.")
    test = next((row for row in metrics["by_split"] if row["split"]=="test"),{})
    a,b,c,d=st.columns(4)
    a.metric("Fenêtres de test scorées",_number(test.get('scored_windows')))
    b.metric("Alertes sur le test",_number(test.get('alert_windows')))
    c.metric("Part de fenêtres en alerte",f"{_number(100*test['alert_rate'],2)} %" if test else "—")
    d.metric("Seuil de calibration",_number(metrics['threshold'],6))
    st.caption("Un score d'atypie n'est pas une probabilité de panne. Aucun indicateur de précision supervisée n'est annoncé sans vérité terrain.")
    period = st.selectbox("Période des scores",["Tout l'historique","30 derniers jours","7 derniers jours"])
    try:
        data = read_score_view(root/"score_plot_data.csv")
        if not data.empty and period != "Tout l'historique":
            days=30 if period.startswith("30") else 7
            data=read_score_view(root/"score_plot_data.csv",start=data.attrs["last_timestamp"]-pd.Timedelta(days=days))
        if not data.empty:
            with st.container(border=True):
                st.subheader("Évolution du score d'anomalie")
                _plot(anomaly_figure(data,metrics["threshold"]),key="ml_scores")
                st.caption(f"{data.attrs.get('aggregation','maxima horaires')} · Les points orange désignent des fenêtres contenant des alertes, sans supposer une anomalie continue.")
        st.subheader("Historique des anomalies")
        page=int(st.number_input("Page de l'historique des anomalies",min_value=1,value=1))
        records=read_alerts(root/"agent_alerts.jsonl",offset=(page-1)*50,limit=50)
        if records:
            rows=[{"Horodatage":record["available_at"],"Score":record["anomaly_score"],"Seuil":record["threshold"],
                   "Mode interprété":record.get("mode_interpreted"),"Partition":record.get("partition")} for record in records]
            st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
            st.download_button("Exporter ces alertes",json.dumps(records,ensure_ascii=False,indent=2),file_name="alertes_affichees.json",mime="application/json")
        else:
            st.info("Aucune alerte sur cette page.")
        st.caption("50 alertes maximum par page, dans l'ordre du fichier. Les fenêtres non documentées ne deviennent pas des faux positifs étiquetés.")
    except (OSError,ValueError,KeyError) as error:
        st.error(f"Affichage des scores indisponible : {error}")
    with st.expander("Protocole, modes et comparaison aux rapports documentés"):
        st.dataframe(pd.DataFrame(metrics["by_split"]),hide_index=True)
        st.dataframe(pd.DataFrame(metrics["by_mode"]),hide_index=True)
        st.dataframe(pd.DataFrame(metrics["documented_period_comparisons"]),hide_index=True)
        st.write(metrics["limitations"])


def _render_sources(sources):
    for source in sources:
        data=source.model_dump() if hasattr(source,"model_dump") else source
        with st.container(border=True):
            st.text(f"{data['file_name']} · p. {data['page']} (page physique)")
            st.caption(f"{AUTHORITIES.get(data['authority'], 'Non vérifié')} · {APPLICABILITIES.get(data['applicability'], 'Applicabilité non vérifiée')}")
            st.text(data.get("passage",data.get("text","")))
            with st.expander("Provenance de la référence"):
                st.text(f"SHA-256 : {data['pdf_sha256']}")
                st.text(data.get("publisher",""))
                st.text(f"Équipement : {data.get('equipment_reference') or 'applicabilité spécifique non établie'}")


def _rag_page():
    from .rag import search_documents
    from .rag.answer import generate_document_answer
    from .rag.index import index_documents,read_manifest
    _header("Assistant RAG","Interroger le corpus technique et retrouver les preuves, document par document.","RECHERCHE DOCUMENTAIRE")
    config=st.session_state.get("rag_config",{"documents":"documents","index":"data/local/rag"})
    with st.expander("Corpus documentaire et indexation"):
        documents=st.text_input("Dossier des PDF",value=config["documents"],key="rag_documents_input")
        index=st.text_input("Dossier de l'index documentaire",value=config["index"],key="rag_index_input")
        identity=(str(Path(documents).resolve()),str(Path(index).resolve()))
        if st.session_state.get("rag_identity") != identity:
            st.session_state.pop("rag_result",None); st.session_state.pop("rag_answer",None)
            st.session_state.rag_identity=identity
        st.session_state.rag_config={"documents":documents,"index":index}
        if st.button("Indexer les PDF"):
            try:
                with st.spinner("Extraction des pages et indexation locale…"):
                    manifest=index_documents(documents,index_directory=index)
                st.session_state.rag_index_summary=manifest
                st.session_state.pop("rag_result",None); st.session_state.pop("rag_answer",None)
                st.success(f"{manifest['pdf_count']} PDF · {manifest['page_count']} pages · {manifest['chunk_count']} passages.")
            except (OSError,ValueError,RuntimeError) as error:
                st.error(str(error))
        try:
            manifest=read_manifest(index)
            if manifest:
                st.caption(f"{manifest['pdf_count']} PDF · {manifest['page_count']} pages physiques · {manifest['chunk_count']} passages indexés")
        except (OSError,ValueError) as error:
            st.error(f"Index documentaire illisible : {error}")
    st.caption("Exemples de questions")
    a,b,c=st.columns(3)
    for column,label,question in [(a,"Température d'huile","Selon UCI, que mesure Oil_temperature et dans quelle unité ?"),
                                  (b,"Maintenance RCM","Selon le guide NASA, que signifie la maintenance RCM ?"),
                                  (c,"Consignation","Selon l'INRS, que signifie la consignation des énergies ?")]:
        if column.button(label,width="stretch"):
            st.session_state.rag_question_text=question
    authority={"Tous les documents":"all","Officiels déclarés":"official","Pédagogiques":"pedagogical","Non vérifiés":"unverified"}
    with st.container(border=True):
        question=st.text_input("Question documentaire",key="rag_question_text",placeholder="Posez une question sur le compresseur ou la documentation…")
        selected=st.selectbox("Statut des documents recherchés",list(authority))
        submit=st.button("Rechercher dans les documents",type="primary")
    if submit:
        st.session_state.pop("rag_answer",None)
        try:
            with st.spinner("Recherche des passages puis sélection des citations…"):
                result=search_documents(question,index_directory=index,authority=authority[selected])
                answer=generate_document_answer(question,result,documents_directory=documents,provider=_provider())
            st.session_state.rag_result=result
            st.session_state.rag_answer=answer
            st.session_state.diagnostic_documents_root=documents
        except (OSError,ValueError,RuntimeError) as error:
            st.session_state.pop("rag_result",None)
            st.error(str(error))
    answer=st.session_state.get("rag_answer")
    result=st.session_state.get("rag_result")
    if not answer or not result:
        st.info("Posez une question pour obtenir une réponse documentée. Aucun LLM n'est appelé au chargement de cette page.")
        return
    left,right=st.columns([1.4,1])
    with left:
        with st.chat_message("user"):
            st.text(result.question)
        with st.chat_message("assistant"):
            if answer.simulation:
                st.warning("SIMULATION MOCK — sélection déterministe de citations, sans appel à un LLM.")
            st.text(answer.message)
            for excerpt in answer.excerpts:
                st.text(f"{excerpt['text']} [{excerpt['source_id']}]")
            if answer.status in ("erreur_api","erreur_configuration","reponse_invalide","sources_invalides"):
                st.error("La réponse n'a pas été publiée. Vérifiez la configuration ou les documents.")
            st.caption("Réponse composée de citations vérifiées sélectionnées par le fournisseur, sans procédure technique libre. Une référence générale ne valide pas une intervention MetroPT.")
            if any(getattr(passage,"retrieval_basis","")=="exact_dataset_signal_and_requested_publisher" for passage in result.passages):
                st.caption("Définition UCI retrouvée par nom exact du signal ; ce repérage lexical n'est ni une probabilité ni une validation de procédure.")
    with right:
        st.subheader("Sources et pages")
        if answer.sources:
            _render_sources(answer.sources)
        else:
            st.info("Aucune source suffisamment exploitable.")
            for warning in result.warnings:
                st.text(warning)
    if result.related_passages:
        with st.expander("Contexte associé — insuffisant pour répondre"):
            _render_sources([{"file_name":p.file_name,"page":p.page,"authority":p.authority,"applicability":p.applicability,
                              "text":p.text,"pdf_sha256":p.pdf_sha256,"publisher":p.publisher,"equipment_reference":p.equipment_reference} for p in result.related_passages])


def _workflow_controls(path):
    from .orchestration.monitoring import load_measurement_window
    from .orchestration.nodes import WorkflowPersistenceError
    synthetic=st.session_state.origin_label=="Synthétique (test)"
    with st.expander("Préparer une nouvelle analyse",expanded="workflow_active_id" not in st.session_state):
        st.text_input("Modèle local de confiance du workflow",value="models/isolation_forest/baseline.joblib",key="model_path")
        a,b=st.columns(2)
        start=a.text_input("Début du contexte du workflow",value="2020-01-01 00:00:00" if synthetic else "2020-03-01 04:08:00")
        end=b.text_input("Fin des mesures du workflow",value="2020-01-01 00:08:00" if synthetic else "2020-03-01 04:15:30")
        question=st.text_input("Question documentaire du workflow",value="Selon le DOE, que décrit la documentation des systèmes d'air comprimé ?")
        st.caption("Dernière minute close, cinq minutes de contexte passé, et une mesure ultérieure pour fermer la fenêtre. 6 000 mesures maximum.")
        if st.button("Démarrer le workflow sur cette fenêtre",type="primary"):
            st.session_state.pop("workflow_active_id",None)
            try:
                with st.spinner("Analyse ML, recherche documentaire et diagnostic…"):
                    request=load_measurement_window(path,start,end,data_context="synthetic_test" if synthetic else "historical_replay",question=question)
                    with _open_workflow(generate=True) as runner:
                        state=runner.start(request)
                st.session_state.workflow_active_id=state["run_id"]
            except WorkflowPersistenceError as error:
                st.session_state.workflow_active_id=error.run_id; st.error(str(error))
            except (OSError,ValueError,sqlite3.Error) as error:
                st.error(str(error))
    _reopen_control()


def _reopen_control():
    with st.expander("Rouvrir un dossier par identifiant"):
        identifier=st.text_input("Identifiant du dossier à rouvrir",value="")
        if st.button("Rouvrir le dossier enregistré"):
            try:
                with _open_workflow() as runner:
                    runner.get_state(identifier)
                st.session_state.workflow_active_id=identifier
            except (OSError,ValueError,KeyError,sqlite3.Error) as error:
                st.error(str(error))


def _get_active_state():
    if not st.session_state.get("workflow_active_id"):
        return None
    try:
        with _open_workflow() as runner:
            return runner.get_state(st.session_state.workflow_active_id)
    except (OSError,ValueError,KeyError,sqlite3.Error) as error:
        st.error(str(error))
        return None


def _diagnostic_page(path):
    _header("Diagnostic agentique","Suivre les preuves et les décisions d'un workflow supervisé.","LANGGRAPH / DIAGNOSTIC PRÉLIMINAIRE")
    _context_notice(path)
    _workflow_controls(path)
    state=_get_active_state()
    if state:
        st.caption(f"Workflow : {state['run_id']}")
        with st.container(border=True):
            _plot(workflow_figure(state),key="workflow_progress")
        if state["interrupts"]:
            st.warning("Le workflow est suspendu : une validation humaine explicite est nécessaire.")
            st.button("Examiner dans Gestion des interventions",on_click=_go_page,args=("Gestion des interventions",),type="primary")
        tabs=st.tabs(["Faits et hypothèses","Proposition de maintenance","Sources documentaires","Journal"])
        with tabs[0]:
            diagnostic=state.get("diagnostic") or {}
            if diagnostic.get("simulation"):
                st.info("SIMULATION MOCK — aucune hypothèse de défaillance inventée.")
            if diagnostic:
                st.subheader("Faits observés")
                facts=[]
                for fact in diagnostic["faits_observes"]:
                    value=fact["value"]
                    facts.append({"Fait":fact["id"],"Observation":fact["description"],"Signal":fact["sensor"] or "Modèle ML",
                                  "Valeur":("oui" if value else "non") if isinstance(value,bool) else _number(value,6),
                                  "Unité":fact["unit"] or "—","Agrégation":{"minute_mean":"Moyenne par minute","instantaneous":"Instantanée"}.get(fact["aggregation"],"Non précisée")})
                st.dataframe(pd.DataFrame(facts),hide_index=True)
                st.subheader("Hypothèses et justifications")
                for hypothesis in diagnostic["hypotheses_de_defaillance"]:
                    st.text(f"{hypothesis['label']} — non confirmée")
                if not diagnostic["hypotheses_de_defaillance"]:
                    st.info("Aucune hypothèse documentée disponible.")
                for argument in diagnostic["arguments_documentaires"]:
                    st.text(f"{argument['source_id']} : {argument['excerpt']}")
            else:
                st.info("Diagnostic indisponible. Les erreurs ou les sources manquantes restent visibles dans le dossier.")
                if state["ml_event"]:
                    st.json(state["ml_event"])
        with tabs[1]:
            proposal=state.get("proposal")
            if proposal:
                st.subheader("Proposition documentaire avant intervention")
                for action in proposal["actions"]:
                    st.text(action["description"])
                st.warning("Aucune procédure physique générée. Les guides généraux ne remplacent pas une notice identifiée du compresseur MetroPT.")
                for missing in proposal["missing_information"]:
                    st.text(missing)
                st.text(_review_status(state))
            else:
                st.info("Aucune proposition émise pour ce dossier.")
        with tabs[2]:
            _render_sources((state.get("diagnostic") or {}).get("sources",[]))
        with tabs[3]:
            st.dataframe(pd.DataFrame(state["journal"]),hide_index=True)
            if state["errors"]:
                st.json(state["errors"])
    else:
        st.info("Lancez une analyse ou rouvrez un dossier enregistré pour suivre les étapes du workflow.")
    with st.expander("Diagnostic manuel à partir d'un événement ML"):
        _diagnostic_panel()


def _review_actions(state):
    from .orchestration.nodes import WorkflowPersistenceError
    run_id=state["run_id"]
    if state["interrupts"]:
        st.subheader("Votre décision")
        key=f"{run_id}_{state['proposal']['proposal_id']}"
        reviewer=st.text_input("Identifiant du validateur",max_chars=200,key=f"reviewer_{key}")
        comment=st.text_area("Commentaire de la revue",max_chars=2000,key=f"comment_{key}")
        ack=st.checkbox("Je confirme cette revue documentaire ; aucune intervention physique n'est autorisée.",key=f"ack_{key}")
        ready=len(reviewer.strip())>=2 and bool(comment.strip()) and ack
        a,b,c=st.columns(3); choice=None
        if a.button("Approuver",disabled=not(ready and state["proposal"]["approval_allowed"]),type="primary"):
            choice="approve"
        if b.button("Refuser",disabled=not ready):
            choice="reject"
        if c.button("Demander une révision",disabled=not ready):
            choice="request_revision"
        if choice:
            try:
                with _open_workflow() as runner:
                    runner.review(run_id,{"proposal_id":state["proposal"]["proposal_id"],"decision":choice,"reviewer":reviewer,"comment":comment,"acknowledged":ack})
                st.rerun()
            except (OSError,ValueError,sqlite3.Error,WorkflowPersistenceError) as error:
                st.error(str(error))
        if state["review_error"]:
            st.error(state["review_error"])
    elif state["pending_nodes"]==["finalize_node"]:
        if st.button("Réessayer l'enregistrement de la décision"):
            try:
                with _open_workflow() as runner:
                    runner.retry_finalization(run_id)
                st.rerun()
            except (OSError,ValueError,sqlite3.Error,WorkflowPersistenceError) as error:
                st.error(str(error))


def _interventions_page():
    _header("Gestion des interventions","Examiner les propositions, consigner les décisions et préserver leur traçabilité.","SUPERVISION HUMAINE")
    try:
        with _open_workflow() as runner:
            runs=runner.list_runs(limit=100)
            decisions=runner.decisions.list_records(limit=100)
    except (OSError,ValueError,sqlite3.Error) as error:
        st.error(f"Inventaire local indisponible : {error}")
        return
    pending=[run for run in runs if run["pending"]]
    approved=[item for item in decisions if item["outcome"]=="approved_documentary_proposal"]
    refused=[item for item in decisions if item["outcome"]=="rejected"]
    revisions=[item for item in decisions if item["outcome"]=="revision_requested"]
    for column,label,value in zip(st.columns(4),["Dossiers en attente","Approbations documentaires","Refus enregistrés","Révisions demandées"],
                                  [len(pending),len(approved),len(refused),len(revisions)]):
        column.metric(label,value)
    st.caption("Comptages sur les 100 derniers dossiers et les 100 dernières décisions locales. Une approbation documentaire n'autorise aucune intervention physique.")
    tabs=st.tabs(["Dossiers en attente","Historique des décisions"])
    with tabs[0]:
        if pending:
            st.dataframe(pd.DataFrame([{"Workflow":run["run_id"],"Fenêtre":run["available_at"],"Approbation possible":run["approval_allowed"],
                                       "Origine":"Synthétique (test)" if run["data_context"]=="synthetic_test" else "Historique rejoué"} for run in pending]),hide_index=True,width="stretch")
            selected=st.selectbox("Dossier en attente à examiner",[""]+[run["run_id"] for run in pending],format_func=lambda value:value[:12] if value else "Choisir un dossier")
            if st.button("Ouvrir la fiche sélectionnée",disabled=not selected):
                st.session_state.workflow_active_id=selected
        else:
            st.info("Aucune proposition en attente dans les dossiers affichés.")
    with tabs[1]:
        if decisions:
            st.dataframe(pd.DataFrame([{"Workflow":item["run_id"],"Décision":OUTCOMES[item["outcome"]],
                                       "Horodatage UTC":item["recorded_at_utc"],"Validateur":item["validator_id"],
                                       "Commentaires":item["comments"]} for item in decisions]),hide_index=True,width="stretch")
        else:
            st.info("Aucune décision enregistrée. Aucune validation n'est créée automatiquement.")
    _reopen_control()
    state=_get_active_state()
    if state:
        with st.container(border=True):
            _render_review_card(state)
            _review_actions(state)
        with st.expander("Journal du dossier"):
            st.dataframe(pd.DataFrame(state["journal"]),hide_index=True)


def main():
    """Router cinq pages sans recharger les données brutes ou appeler un LLM au démarrage."""
    st.set_page_config(page_title="MetroPT · Maintenance prescriptive",page_icon="⚙",layout="wide")
    st.markdown(CSS,unsafe_allow_html=True)
    page,path,identity=_sidebar()
    try:
        if page=="Dashboard":
            _dashboard_page(path,identity)
        elif page=="Détection d'anomalies":
            _ml_page(path,identity)
        elif page=="Assistant RAG":
            _rag_page()
        elif page=="Diagnostic agentique":
            _diagnostic_page(path)
        else:
            _interventions_page()
    except (OSError,ValueError,KeyError,sqlite3.Error) as error:
        st.error(f"Cette vue ne peut pas être chargée : {error}")
        st.caption("Vérifiez les fichiers locaux et les paramètres. Les traitements et validations ne sont pas poursuivis.")
    st.caption("MetroPT · Prototype académique de maintenance 4.0 — Anomalie ≠ diagnostic confirmé · Aucune commande vers un équipement.")


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
            mixed=event.get('mode_mixed')
            st.text(f"Mode interprété : {event['mode_interpreted']} ; mode mixte : {'non renseigné' if mixed is None else 'oui' if mixed else 'non'}.")
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
        st.text(f"{source['id']} — {source['file_name']}, p. {source['page']} (page physique) ; statut {AUTHORITIES.get(source['authority'], 'Non vérifié')} ; portée {APPLICABILITIES.get(source['applicability'], 'Applicabilité non vérifiée')}.")
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
                                             documents_directory=st.session_state.get("diagnostic_documents_root", "documents"), provider=_provider())
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
