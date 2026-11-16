"""Rapports Markdown et HTML Plotly autonomes, construits uniquement depuis les mesures."""

import json
import statistics
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from .campaign import CONFIGURATIONS

LIMITS = [
    "Aucun LLM génératif réel n'a été évalué : le même mock déterministe est utilisé pour A, B et C. L'erreur API est injectée. Les latences ne mesurent pas une API ni une inférence LLM.",
    "A et B sont des adaptateurs d'ablation, pas deux nouveaux modes de production. Les mesures sont identiques ; ils n'ont ni score ML ni validation persistante. Leur absence de capacité est déclarée non prise en charge, pas assimilée à une hallucination.",
    "Dans les dix scénarios, mesures et PDF sont fabriqués et le résultat ML est injecté. Le vrai graphe, Chroma, PyMuPDF, les validateurs et SQLite sont exécutés. Les vecteurs synthétiques ne mesurent pas la qualité E5.",
    "Les scénarios de fuite restent des suspicions. Aucune fuite n'est confirmée par l'absence ou la présence d'une alerte. Aucun nouveau label de panne n'est créé.",
    "La fidélité calculée est la proportion de citations littéralement présentes dans les sources vérifiées. Une citation exacte peut être non pertinente ou insuffisante ; cette mesure n'est pas une fidélité sémantique ni une qualité causale.",
    "La qualité clinique des diagnostics exige une évaluation indépendante par des personnes compétentes et des références applicables à l'équipement. Elle reste non mesurée. Le mock ne produit aucune hypothèse : ce comportement ne prouve pas la qualité d'un LLM réel.",
    "La contradiction est un oracle pédagogique explicite, pas une contradiction découverte dans les vrais manuels. Deux sources présentes et une incertitude générique ne suffisent pas : les affirmations opposées doivent être citées et l'insuffisance signalée.",
    "Pas de labels binaires exhaustifs sur MetroPT-3 : Precision, Recall de défaillance et F1 restent NA. La couverture des quatre périodes documentées est uniquement descriptive ; les autres dates restent non étiquetées.",
    "Les annotations documentaires préexistent à cette campagne mais ont servi au développement. Jeu petit, pages parfois alternatives, sans nouveau jeu tenu à l'écart ; les métriques ne sont pas une estimation indépendante de généralisation.",
    "Les questions UCI avaient été annotées pour un corpus limité au PDF du dataset. Le corpus actuel contient aussi INRS : une demande générale sur la consignation peut maintenant retrouver des sources. L'écart à l'abstention historique exige une réannotation indépendante ; il ne démontre à lui seul ni hallucination ni applicabilité à MetroPT. Les annotations et scores historiques sont conservés sans correction après observation.",
    "Répétitions déterministes locales, caches partagés et effectifs faibles : médiane et p95 descriptifs, sans test statistique de supériorité. Les temps de préparation communs sont séparés ; la première exécution peut inclure des caches froids.",
    "Les décisions d'approbation/refus de la campagne sont injectées uniquement sur des fixtures synthétiques et dans un stockage séparé. Aucun dossier réel n'est approuvé. Une approbation documentaire ne déclenche aucune action physique.",
]


def fmt(value, *, percent=False):
    if value is None:
        return "NA"
    return (f"{value * 100:.1f} %" if percent else f"{value:.6f}" if 0 < abs(value) < .001 else f"{value:.3f}").replace(".", ",")


def table(headers, rows):
    """Échapper les cellules : même un rapport n'exécute pas le texte d'une source."""
    clean = lambda value: ("NA" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def report_html(markdown):
    """Rendre nos titres et tableaux en HTML, en échappant chaque cellule."""
    output, in_table = [], False
    for line in markdown.splitlines():
        if line.startswith("| "):
            if set(line.replace("|", "").replace(" ", "")) == {"-"}:
                continue
            cells = line.strip("| ").split(" | ")
            tag = "td" if in_table else "th"
            if not in_table:
                output.append('<div class="table-wrap"><table>')
                in_table = True
            output.append("<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in cells) + "</tr>")
            continue
        if in_table:
            output.append("</table></div>")
            in_table = False
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 4)
            output.append(f"<h{level}>" + escape(line.lstrip("# ")) + f"</h{level}>")
        elif line.strip():
            output.append("<p>" + escape(line) + "</p>")
    if in_table:
        output.append("</table></div>")
    return "\n".join(output)


def figures(campaign, real):
    """Pas de barre nulle pour une métrique absente ; les valeurs NA sont exclues."""
    result = []
    summary = campaign["summary"]
    coverage = go.Figure()
    for field, name, color in (("passed", "Traités selon l'oracle", "#078B92"), ("failed", "Critère non satisfait", "#E7A343"), ("not_supported", "Capacité absente", "#9DA9BB")):
        coverage.add_bar(x=[row["configuration"] for row in summary], y=[row[field] for row in summary], name=name, marker_color=color)
    coverage.update_layout(title="Scénarios simulés — comptages, sans inférence LLM réelle", barmode="stack", yaxis_title="Exécutions")
    result.append(coverage)
    latency = go.Figure()
    for configuration in CONFIGURATIONS:
        latency.add_trace(go.Box(y=[row["latency_seconds"] * 1000 for row in campaign["rows"] if row["configuration"] == configuration], name=configuration, boxpoints="all", jitter=.2))
    latency.update_layout(title="Latence locale des scénarios — mock, hors initialisation", yaxis_title="Millisecondes (échelle logarithmique)", yaxis_type="log", yaxis_dtick=1)
    result.append(latency)
    codes = list(dict.fromkeys(row["scenario"] for row in campaign["rows"]))
    values = [[next(row for row in campaign["rows"] if row["scenario"] == code and row["configuration"] == config)["status"] for config in CONFIGURATIONS] for code in codes]
    grid = go.Figure(go.Heatmap(z=[[{"passed": 1, "failed": .5, "not_supported": 0}[value] for value in row] for row in values],
        x=list(CONFIGURATIONS), y=codes, text=values, texttemplate="%{text}", showscale=False,
        colorscale=[[0, "#DFE4EC"], [.5, "#F9D796"], [1, "#9BDBCE"]], zmin=0, zmax=1))
    grid.update_layout(title="Matrice des scénarios — première répétition", height=540)
    result.append(grid)
    if real and real.get("retrieval"):
        retrieval = real["retrieval"]
        figure = go.Figure()
        for field, label in (("macro_annotated_page_recall", "Recall macro des pages annotées"), ("hit_rate", "Hit : au moins une page annotée")):
            figure.add_bar(x=list(retrieval["scores"]), y=[values[field] for values in retrieval["scores"].values()], name=label)
        figure.update_layout(title="Recherche réelle E5 — références préexistantes, sans nouveau jeu indépendant", xaxis_title="k", yaxis_tickformat=".0%", yaxis_range=[0, 1], barmode="group")
        result.append(figure)
    if real and real.get("ml"):
        periods = real["ml"]["documented_periods"]
        figure = go.Figure()
        figure.add_bar(x=[row["identifier"] for row in periods], y=[row["scored_windows"] for row in periods], name="Minutes scorées")
        figure.add_bar(x=[row["identifier"] for row in periods], y=[row["alert_windows"] for row in periods], name="Minutes alertées")
        figure.update_layout(title="Périodes UCI documentées — comparaison descriptive après entraînement", yaxis_title="Fenêtres d'une minute", barmode="group")
        result.append(figure)
    for figure in result:
        figure.update_layout(template="plotly_white", separators=", ", font={"family": "Arial", "color": "#18243A"}, colorway=["#078B92", "#5269C8", "#E7A343"], margin={"l": 50, "r": 25, "t": 70, "b": 55})
        figure.update_xaxes(automargin=True)
        figure.update_yaxes(automargin=True)
    return result


def write_reports(campaign, real, output, *, tests=None):
    """Écrire les nombres observés, les oracles non satisfaits et les métriques NA."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {"campaign": campaign, "real": real, "tests": tests, "methodological_limits": LIMITS}
    (output / "metrics.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    parts = ["# Campagne d'évaluation du prototype MetroPT", "Exécution UTC : " + campaign["executed_at_utc"],
        "## Protocole et configurations", campaign["scope"],
        table(["Configuration", "Périmètre"], CONFIGURATIONS.items()),
        f"{campaign['scenario_count']} scénarios × 3 configurations × {campaign['repeats']} répétitions = {len(campaign['rows'])} exécutions simulées. Rotation de l'ordre A/B/C. Initialisation du corpus exclue des latences.",
        "## Résultats des scénarios simulés",
        table(["Configuration", "Traités / total", "Critère non satisfait", "Capacité absente", "Taux traité", "Médiane (s)", "p95 (s)", "Fidélité littérale / citations"],
              [[r["configuration"], f"{r['passed']} / {r['runs']}", r["failed"], r["not_supported"], fmt(r["scenario_handled_rate"], percent=True), fmt(r["latency_median_seconds"]), fmt(r["latency_p95_seconds"]), f"{fmt(r['literal_fidelity'], percent=True)} / {r['citation_count']}" ] for r in campaign["summary"]]),
        "Le taux traité utilise tous les scénarios au dénominateur, y compris les capacités absentes. Ce taux compare l'étendue des architectures selon des oracles explicites, sans démontrer une supériorité de raisonnement.",
        table(["Scénario", "Critère", "A", "B", "C"], [[next(r["label"] for r in campaign["rows"] if r["scenario"] == code),
            next(r["requirement"] for r in campaign["rows"] if r["scenario"] == code)] +
            [next(r["status"] for r in campaign["rows"] if r["scenario"] == code and r["configuration"] == config) for config in CONFIGURATIONS]
            for code in dict.fromkeys(r["scenario"] for r in campaign["rows"])]),
        "### Diagnostic et contradictions",
        table(["Configuration", "Faits conservés / dossiers vérifiés", "Incertitude explicite / dossiers", "Hypothèses mock", "Qualité clinique"],
              [[r["configuration"], f"{fmt(r['facts_preserved_rate'], percent=True)} / {r['facts_checked']}", f"{fmt(r['explicit_uncertainty_rate'], percent=True)} / {r['uncertainty_checked']}", r["hypothesis_count"], "NA"] for r in campaign["summary"]]),
        "La conformité des faits, l'incertitude explicite et les citations sont auditées automatiquement. Le nombre d'hypothèses produites par le mock est publié dans metrics.json. Une absence d'hypothèse ne démontre pas la justesse d'un diagnostic.",
        "Le scénario contradictoire exige les deux affirmations opposées comme arguments et un signalement d'insuffisance documentaire. Le mock sélectionne un seul extrait ; B et C ne satisfont pas cet oracle. Le prototype ne dispose pas de détecteur sémantique de contradictions. Une proposition peut donc rester approuvable comme préparation documentaire malgré ce conflit non identifié : point à traiter avant tout usage au-delà de la démonstration.",
        "### Validation humaine",
        table(["Configuration", "Approbations avant revue", "Dossiers exigeant revue", "Revues explicites", "Interruptions conformes", "Reprises conformes", "API LLM réelle"], [[r["configuration"], r["automatic_approvals"], r["human_review_opportunities"], r["explicit_submissions"], fmt(r["human_review_compliance"], percent=True), fmt(r["persistent_resume_success_rate"], percent=True), r["real_llm_api_calls"]] for r in campaign["summary"]]),
        "A et B ne possèdent pas de mécanisme de revue ni de reprise : leurs zéros ne prouvent pas l'existence d'un garde-fou. C est vérifié par interruption réelle, soumission explicite sur fixture synthétique, réouverture du checkpoint et lecture du reçu SQLite."]
    if real:
        parts += ["## Données réelles et PDF réels", "Ce volet réutilise les artefacts figés et le modèle existant. Il n'entraîne ni ne calibre le modèle. Les éventuelles générations restent mock."]
        if real.get("errors"):
            parts.append("Sections non mesurées : " + ", ".join(real["errors"]))
        if real.get("ml"):
            ml = real["ml"]
            parts += ["### Détection ML", table(["Partition", "Fenêtres scorées", "Alertes", "Fraction alertée"],
                [[name, row["scored_windows"], row["alert_windows"], fmt(row["alert_rate"], percent=True)] for name, row in ml["counts_by_split"].items()]),
                table(["Période UCI", "Minutes scorées", "Minutes alertées", "Couverture observée", "Fraction alertée"],
                [[row["identifier"], row["scored_windows"], row["alert_windows"], fmt(row["observation_coverage"], percent=True), fmt(row["alert_fraction_of_scored_minutes"], percent=True)] for row in ml["documented_periods"]]),
                f"Périodes observables avec au moins une alerte : {ml['period_coverage']['with_at_least_one_alert']} / {ml['period_coverage']['with_observations']}. " + ml["limits"],
                "Precision = NA ; Recall de défaillance = NA ; F1 = NA. Aucune de ces valeurs n'est remplacée par zéro."]
        if real.get("retrieval"):
            retrieval = real["retrieval"]
            parts += ["### Recall@k documentaire", f"{retrieval['answerable_count']} questions avec références et {retrieval['unanswerable_count']} demandes d'abstention, sur {retrieval['indexed_pages']} pages physiques et {retrieval['indexed_chunks']} chunks.",
                table(["k", "Recall macro pages annotées", "Hit@k"], [[k, fmt(score["macro_annotated_page_recall"], percent=True), fmt(score["hit_rate"], percent=True)] for k, score in retrieval["scores"].items()]),
                "Recall@k(q) = nombre de pages annotées retrouvées parmi les k premiers passages / nombre de pages annotées. Moyenne par question ; doublons dédupliqués. Hit@k exige seulement une page annotée.",
                "Abstention selon les annotations historiques : " + fmt(retrieval["unanswerable_abstention_rate"], percent=True) + ". " + retrieval["annotation_limits"],
                "Attention au périmètre de référence : le cas UCI « Quelle procédure de consignation de sécurité appliquer avant démontage ? » avait été déclaré non répondable sans guides de sécurité. Des passages INRS sont désormais disponibles. Le désaccord avec cette annotation est publié, mais ne prouve pas un échec sémantique ; leur applicabilité exacte reste à examiner. Le jeu n'a pas été réétiqueté après observation des résultats.",
                table(["Question", "Références attendues", "Références retournées", "Recall@5", "Latence (s)"],
                    [[r["question"], "; ".join(f"{v['file']} p.{v['page']}" for v in r.get("expected_references", [])),
                      "; ".join(f"{v['file']} p.{v['page']}" for v in r["returned_references"]), fmt(r["scores"]["5"]["recall"], percent=True), fmt(r["latency_seconds"])] for r in retrieval["cases"]])]
        if real.get("workflows"):
            rows = real["workflows"]["rows"]
            parts += ["### Parcours sur mesures historiques", real["workflows"]["selection"], real["workflows"]["latency_scope"],
                table(["Cas choisi par le modèle", "Configuration", "n", "Médiane (s)", "Appels mock", "Dossiers en attente", "Revues soumises"],
                [[case, config, len(group), fmt(float(statistics.median(r["latency_seconds"] for r in group))), sum(r["calls"]["mock"] for r in group), sum(r["awaiting_human"] for r in group), sum(r["review_submitted"] for r in group)]
                 for case in dict.fromkeys(r["case"] for r in rows) for config in CONFIGURATIONS
                 if (group := [r for r in rows if r["case"] == case and r["configuration"] == config])]),
                "Aucune décision humaine n'a été soumise sur ces mesures réelles. Aucune validation de cause ni procédure physique n'a été produite."]
    if (output / "figures").is_dir():
        parts += ["## Graphiques exportés"] + [f"![{path.stem}](figures/{path.name})" for path in sorted((output / "figures").glob("*.svg"))]
    parts += ["## Qualité diagnostique : grille restant à faire évaluer",
        table(["Dimension", "Mesure disponible", "Référence nécessaire"], [
            ["Faits et unités", "Correspondance exacte aux entrées", "Mesures fournies"],
            ["Citations", "Fidélité littérale avec dénominateur", "PDF, page et empreinte"],
            ["Pertinence des hypothèses", "NA — pas de notation clinique", "Experts indépendants, dossier anonymisé"],
            ["Exhaustivité et contradictions", "Oracle pédagogique ; pas de score sémantique général", "Corpus applicable et annotation experte"],
            ["Cause réelle", "NA", "Diagnostic terrain confirmé, indépendant du modèle"],
            ["Utilité et applicabilité des contrôles", "NA", "Notice exacte et validation compétente"]]),
        "## Tests automatisés", json.dumps(tests, ensure_ascii=False) if tests else "Non renseignés dans ce rapport : exécuter pytest et réattacher son fichier JUnit.",
        "## Limites méthodologiques"] + [f"- {limit}" for limit in LIMITS]
    markdown = "\n\n".join(parts) + "\n"
    (output / "rapport.md").write_text(markdown, encoding="utf-8")
    plots = figures(campaign, real)
    chart_html = "\n".join(pio.to_html(figure, full_html=False, include_plotlyjs=True if i == 0 else False) for i, figure in enumerate(plots))
    html = '<!doctype html><html lang="fr"><meta charset="utf-8"><title>Évaluation MetroPT</title><style>body{font:16px system-ui;color:#18243A;background:#F5F7FB;max-width:1300px;margin:auto;padding:32px}p{line-height:1.6}h1{color:#078B92}.table-wrap{overflow:auto;background:white;margin:20px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;border-bottom:1px solid #DFE4EC;text-align:left}th{background:#E4F3EF}</style><h1>Campagne d’évaluation MetroPT</h1>' + chart_html + report_html(markdown) + '</html>'
    (output / "rapport.html").write_text(html, encoding="utf-8")
    return [output / name for name in ("metrics.json", "rapport.md", "rapport.html")]
