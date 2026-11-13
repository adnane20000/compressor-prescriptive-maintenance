"""Refus logiciels de demandes à risque ; aucune procédure d'intervention."""

import re
import unicodedata
from dataclasses import dataclass

SAFETY_NOTICE = (
    "Le prototype ne fournit pas de mode opératoire pour cette demande à risque. "
    "Une évaluation de sécurité par une personne compétente et une procédure "
    "adaptée à l'équipement restent nécessaires. Aucun extrait ne vaut autorisation d'intervenir."
)


@dataclass(frozen=True)
class SafetyAssessment:
    blocked: bool
    reasons: tuple[str, ...]


def assess_safety_request(question: str) -> SafetyAssessment:
    """Détecter quelques demandes explicites ; ce filtre n'est pas une certification.

    Les questions de prévention restent recherchables. ED 6129 décrit des modes
    encadrés : la règle du prototype ne prétend pas les interdire universellement.
    """
    text = "".join(c for c in unicodedata.normalize("NFKD", question.lower()) if not unicodedata.combining(c))
    informative = bool(re.search(r"\b(pourquoi|risques?|precautions?|why|hazards?|que dit|quelles conditions|quelles limites|quelles exigences|what requirements)\b", text))
    asks_how = bool(re.search(r"\b(comment|how|etapes?|procedure|instructions?|donne|donnez|puis.je|peut.on|je veux)\b", text))
    removal = bool(re.search(r"\b(shunt\w*|contourn\w*|desactiv\w*|neutralis\w*|supprim\w*|bypass\w*|disabl\w*|override)\b", text))
    protection = bool(re.search(r"\b(pressostat|soupape|securite|protections?|protecteurs?|carter|interverrouill\w*|arret d.urgence|safety|interlock|guard|emergency stop)\b", text))
    action = bool(re.search(r"\b(intervenir|intervention|demonter|demontage|ouvrir|remplacer|reparer|desserrer|travailler|work|open|replace|dismantle)\b", text))
    energy = bool(re.search(r"\b(sous pression|sous tension|en marche|sans arreter|pressurized|energized|without stopping|\d+(?:[.,]\d+)?\s*(?:bar|psi|mpa))\b", text))
    omission = bool(re.search(r"\b(sans consignation|sans depressuris\w*|sans isoler|without lockout|without isolation)\b", text))
    reasons = []
    if not informative or asks_how:
        if removal and protection:
            reasons.append("neutralisation_ou_contournement_de_protection")
        if action and energy:
            reasons.append("intervention_avec_energie_potentiellement_presente")
        if (action or asks_how) and omission:
            reasons.append("demande_de_suppression_de_mise_en_securite")
    return SafetyAssessment(bool(reasons), tuple(reasons))
