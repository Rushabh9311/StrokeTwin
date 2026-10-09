FRIBERG_STROKE_RISK_BY_SCORE = {
    0: 0.2,
    1: 0.6,
    2: 2.2,
    3: 3.2,
    4: 4.8,
    5: 7.2,
    6: 9.7,
    7: 11.2,
    8: 10.8,  # lower than score 7 — real quirk in the Friberg 2012 data, not a bug
    9: 12.2,
}


def cha2ds2_vasc_breakdown(patient) -> list[dict]:
    """Which risk factors this patient has, and the points each one adds."""
    items = []
    if patient.congestive_heart_failure:
        items.append({"factor": "Congestive heart failure", "points": 1})
    if patient.hypertension_status:
        items.append({"factor": "Hypertension", "points": 1})
    if patient.age >= 75:
        items.append({"factor": "Age 75 or over", "points": 2})
    elif patient.age >= 65:
        items.append({"factor": "Age 65-74", "points": 1})
    if patient.diabetes_status:
        items.append({"factor": "Diabetes", "points": 1})
    if patient.prior_tia_or_stroke:
        items.append({"factor": "Prior stroke/TIA", "points": 2})
    if patient.vascular_disease:
        items.append({"factor": "Vascular disease", "points": 1})
    if patient.sex == "F":
        items.append({"factor": "Female sex", "points": 1})
    return items


def cha2ds2_vasc_score(patient) -> int:
    return sum(item["points"] for item in cha2ds2_vasc_breakdown(patient))


def annual_stroke_risk_pct(score: int) -> float:
    return FRIBERG_STROKE_RISK_BY_SCORE[score]


def alert_level(fused: float) -> str:
    # Fixed demo cutoffs used only to organize sample records, not clinical bands.
    if fused >= 3.0:
        return "high"
    if fused >= 1.0:
        return "moderate"
    return "low"
