"""Dictionnaire des vrais signaux du CSV distribué par UCI, dataset 791."""

from dataclasses import asdict, dataclass

SOURCE_URL = "https://archive.ics.uci.edu/dataset/791/metropt+3+"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
CSV_NAME = "MetroPT3(AirCompressor).csv"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class Signal:
    name: str
    kind: str
    unit: str
    description: str


SIGNALS = (
    Signal("TP2", "analogique", "bar", "Pression au compresseur"),
    Signal("TP3", "analogique", "bar", "Pression au panneau pneumatique"),
    Signal("H1", "analogique", "bar", "Pression associée à la décharge du séparateur"),
    Signal("DV_pressure", "analogique", "bar", "Pression liée à la décharge des sécheurs"),
    Signal("Reservoirs", "analogique", "bar", "Pression en aval des réservoirs"),
    Signal("Oil_temperature", "analogique", "°C", "Température d'huile du compresseur"),
    Signal("Motor_current", "analogique", "A", "Courant d'une phase du moteur triphasé"),
    Signal("COMP", "numérique", "sans unité", "Signal électrique de la vanne d'admission"),
    Signal("DV_eletric", "numérique", "sans unité", "Commande électrique de la vanne de sortie"),
    Signal("Towers", "numérique", "sans unité", "Sélection de la tour de séchage"),
    Signal("MPG", "numérique", "sans unité", "Signal de commande de mise en charge"),
    Signal("LPS", "numérique", "sans unité", "Signal de basse pression"),
    Signal("Pressure_switch", "numérique", "sans unité", "Signal de décharge des tours"),
    Signal("Oil_level", "numérique", "sans unité", "Signal électrique du niveau d'huile"),
    Signal("Caudal_impulses", "numérique", "sans unité", "Signal d'impulsions du débit d'air"),
)
SIGNAL_NAMES = tuple(signal.name for signal in SIGNALS)
REQUIRED_COLUMNS = ("timestamp", *SIGNAL_NAMES)
PLOT_COLUMNS = ("Oil_temperature", "Motor_current", "TP2", "TP3")


def schema_records() -> list[dict]:
    """Retourner les unités documentées, sans déduire de capteurs supplémentaires."""
    return [asdict(signal) for signal in SIGNALS]
