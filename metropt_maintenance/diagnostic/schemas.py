"""Contrats stricts du diagnostic préliminaire et de la sélection du LLM."""

from typing import Literal
from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

MISSING_INFORMATION = {
    "equipment_identity": "Identification exacte du compresseur et de sa configuration non établie.",
    "equipment_manual": "Notice et procédure propres à cet équipement non fournies ou non établies.",
    "safety_status": "État de mise en sécurité et analyse des risques du site non établis par ces données.",
    "sensor_context": "Contexte, disponibilité ou validité des mesures à compléter.",
    "operating_context": "Conditions de fonctionnement et transitions à documenter.",
    "maintenance_history": "Historique de maintenance à confronter aux observations.",
    "reference_context": "Représentativité de la référence ML et contexte de calibration à vérifier.",
    "documentary_evidence": "Passages documentaires vérifiables absents ou insuffisants.",
}
CHECKS = {
    "review_sensor_data": "Faire examiner la cohérence, les horodatages et les valeurs manquantes des données capteurs.",
    "review_operating_context": "Faire comparer la fenêtre au contexte et au mode de fonctionnement observés.",
    "review_history": "Faire consulter l'historique de maintenance disponible.",
    "request_equipment_manual": "Obtenir la notice et les documents applicables à l'équipement identifié.",
    "request_qualified_safety_review": "Soumettre le dossier à une personne compétente pour l'analyse de sécurité propre au site et à l'équipement.",
    "compare_reference_windows": "Faire comparer les données à des fenêtres historiques de même contexte, sans réajustement sur le test.",
}
MissingCode = Literal["equipment_identity", "equipment_manual", "safety_status", "sensor_context", "operating_context", "maintenance_history", "reference_context", "documentary_evidence"]
CheckCode = Literal["review_sensor_data", "review_operating_context", "review_history", "request_equipment_manual", "request_qualified_safety_review", "compare_reference_windows"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AnomalyEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    available_at: str = Field(validation_alias=AliasChoices("available_at", "timestamp"))
    model_id: str = Field(min_length=1, max_length=200)
    anomaly_score: FiniteFloat | None
    threshold: FiniteFloat
    is_anomaly: bool | None
    status: Literal["scored", "insufficient_data", "insufficient_history"] = "scored"
    data_context: Literal["historical_replay", "synthetic_test", "unspecified"] = "unspecified"
    diagnosis_confirmed: Literal[False] = False
    score_is_failure_probability: Literal[False] = False
    window_start: str | None = None
    mode_interpreted: str | None = Field(default=None, max_length=200)
    mode_mixed: bool | None = None
    sample_count: int | None = Field(default=None, ge=0)
    missing_fraction: FiniteFloat | None = Field(default=None, ge=0, le=1)
    source_file_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def consistent_event(self):
        datetime.fromisoformat(self.available_at.replace("Z", "+00:00"))
        if self.window_start is not None:
            datetime.fromisoformat(self.window_start.replace("Z", "+00:00"))
        if self.status == "scored":
            if self.anomaly_score is None or self.is_anomaly is None:
                raise ValueError("Un événement scoré exige score et is_anomaly.")
            if self.is_anomaly != (self.anomaly_score > self.threshold):
                raise ValueError("L'indicateur d'anomalie est incohérent avec score > seuil.")
        elif self.anomaly_score is not None or self.is_anomaly is not None:
            raise ValueError("Un événement non scoré ne porte pas de score ou d'alerte.")
        return self


class Measurement(StrictModel):
    value: FiniteFloat | None
    unit: str | None = None
    aggregation: Literal["instantaneous", "minute_mean", "unspecified"] = "unspecified"


class SourceReference(StrictModel):
    id: str
    file_name: str
    relative_path: str
    page: int = Field(ge=1)
    pdf_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    authority: str
    document_type: str
    applicability: str
    equipment_reference: str
    publisher: str
    source_url: str
    passage: str


class ObservedFact(StrictModel):
    id: str
    origin: Literal["ml", "measurement"]
    description: str
    value: FiniteFloat | bool
    sensor: str | None = None
    unit: str | None = None
    aggregation: str | None = None


class ArgumentDraft(StrictModel):
    id: str = Field(pattern=r"^A[1-8]$")
    source_id: str = Field(pattern=r"^S[1-8]$")
    quote: str = Field(min_length=10, max_length=600)


class HypothesisDraft(StrictModel):
    source_id: str = Field(pattern=r"^S[1-8]$")
    label_quote: str = Field(min_length=5, max_length=180)
    argument_ids: list[str] = Field(min_length=1, max_length=8)
    fact_ids: list[str] = Field(min_length=1, max_length=20)


class DiagnosticDraft(StrictModel):
    """Le LLM sélectionne des citations et codes ; aucun champ de mesures ou procédure libre."""
    hypotheses: list[HypothesisDraft] = Field(max_length=5)
    arguments: list[ArgumentDraft] = Field(max_length=8)
    missing_information: list[MissingCode] = Field(max_length=8)
    recommended_checks: list[CheckCode] = Field(max_length=6)
    uncertainty: Literal["élevé", "très élevé"]


class DocumentaryArgument(StrictModel):
    id: str
    source_id: str
    excerpt: str
    interpretation: Literal["Élément documentaire à confronter aux faits ; aucune causalité démontrée."] = "Élément documentaire à confronter aux faits ; aucune causalité démontrée."


class FailureHypothesis(StrictModel):
    label: str
    status: Literal["non confirmée"] = "non confirmée"
    fact_ids: list[str]
    argument_ids: list[str]
    source_id: str
    equipment_relevance: Literal["à vérifier", "non établie"]


class MissingInformation(StrictModel):
    code: MissingCode
    description: str


class RecommendedCheck(StrictModel):
    code: CheckCode
    description: str
    nature: Literal["revue documentaire ou analyse de données"] = "revue documentaire ou analyse de données"
    human_validation_required: Literal[True] = True


class DiagnosticResult(StrictModel):
    schema_version: Literal["preliminary-diagnostic-v1"] = "preliminary-diagnostic-v1"
    statut: Literal["diagnostic préliminaire"] = "diagnostic préliminaire"
    etat_generation: Literal["ok", "information_insuffisante", "erreur_api", "erreur_configuration", "reponse_invalide", "sources_invalides"]
    simulation: bool
    fournisseur: str
    modele: str
    contexte_donnees: str
    evenement: AnomalyEvent
    faits_observes: list[ObservedFact]
    hypotheses_de_defaillance: list[FailureHypothesis]
    arguments_documentaires: list[DocumentaryArgument]
    sources: list[SourceReference]
    informations_manquantes: list[MissingInformation]
    controles_recommandes: list[RecommendedCheck]
    niveau_incertitude: Literal["élevé", "très élevé"]
    limites: list[str]
    erreur: str | None = None
    diagnostic_confirme: Literal[False] = False
    validation_humaine_requise: Literal[True] = True
    autorisation_intervention: Literal[False] = False
    intervention_physique_declenchee: Literal[False] = False
