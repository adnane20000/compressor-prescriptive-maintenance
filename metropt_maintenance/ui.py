"""Exploration locale et rejeu pas à pas de mesures historiques."""

from pathlib import Path

import pandas as pd
import streamlit as st

from .ingestion.exploration import profile_csv
from .ingestion.replay import iter_replay
from .ingestion.report import aggregate_charts, build_figures, write_report
from .ingestion.schema import CSV_NAME, PLOT_COLUMNS


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

    st.info("Mode démonstration — aucune clé API nécessaire. Les mesures chargées sont historiques.")
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
    _replay_panel(path, identity)
