"""Transparent cross-stream demo watch for synthetic patient profiles.

This is a rule-based interface demonstration, not a clinical prediction model.
It compares a patient's simulated wearable baseline with recent readings, then
uses synthetic EHR cardiovascular context to decide whether to raise a demo
review watch. The thresholds are product-demo values, not medical cutoffs.
"""

from __future__ import annotations

from typing import Iterable


BASELINE_READINGS = 3
RECENT_READINGS = 3

# Deliberately labelled demo thresholds. They are not clinical reference ranges.
DEMO_THRESHOLDS = {
    "resting_hr": ("Resting heart rate", "increase", 10.0, "bpm"),
    "hrv_ms": ("Heart-rate variability", "decrease", 20.0, "ms"),
    "sleep_hours": ("Sleep", "decrease", 15.0, "hours"),
    "steps": ("Daily steps", "decrease", 25.0, "steps"),
}


def _value(reading, field: str) -> float:
    if isinstance(reading, dict):
        return float(reading[field])
    return float(getattr(reading, field))


def _context_factors(profile) -> list[str]:
    factors = []
    if int(getattr(profile, "age", 0) or 0) >= 65:
        factors.append("Age 65+")
    if bool(getattr(profile, "hypertension_status", False)):
        factors.append("Hypertension in profile")
    if bool(getattr(profile, "diabetes_status", False)):
        factors.append("Diabetes in profile")
    if bool(getattr(profile, "prior_tia_or_stroke", False)):
        factors.append("Prior stroke/TIA in profile")
    if bool(getattr(profile, "congestive_heart_failure", False)):
        factors.append("Heart failure in profile")
    if bool(getattr(profile, "vascular_disease", False)):
        factors.append("Vascular disease in profile")
    return factors


def analyze_wearable_trend(readings: Iterable, profile) -> dict:
    """Compare first three wearable readings with the latest three.

    A cross-stream watch is triggered only when at least two demo wearable
    thresholds are crossed and at least one static profile context factor is
    present. The ECG sample is intentionally not part of this calculation.
    """
    ordered = sorted(
        list(readings),
        key=lambda item: item["timestamp"] if isinstance(item, dict) else item.timestamp,
    )
    context_factors = _context_factors(profile)
    if len(ordered) < BASELINE_READINGS + RECENT_READINGS:
        return {
            "status": "insufficient",
            "triggered": False,
            "summary": "Add at least six wearable readings to compare a baseline with recent values.",
            "changes": [],
            "context_factors": context_factors,
            "baseline_days": min(len(ordered), BASELINE_READINGS),
            "recent_days": 0,
            "thresholds": DEMO_THRESHOLDS,
        }

    baseline = ordered[:BASELINE_READINGS]
    recent = ordered[-RECENT_READINGS:]
    changes = []
    for field, (label, direction, threshold, unit) in DEMO_THRESHOLDS.items():
        baseline_mean = sum(_value(row, field) for row in baseline) / len(baseline)
        recent_mean = sum(_value(row, field) for row in recent) / len(recent)
        delta_pct = ((recent_mean - baseline_mean) / baseline_mean * 100.0) if baseline_mean else 0.0
        crossed = delta_pct >= threshold if direction == "increase" else delta_pct <= -threshold
        changes.append({
            "field": field,
            "label": label,
            "baseline": round(baseline_mean, 1),
            "recent": round(recent_mean, 1),
            "delta_pct": round(delta_pct, 1),
            "direction": direction,
            "threshold_pct": threshold,
            "unit": unit,
            "crossed": crossed,
        })

    crossed_count = sum(1 for item in changes if item["crossed"])
    if crossed_count >= 2 and context_factors:
        status = "cross_stream_watch"
        summary = (
            f"{crossed_count} wearable trends crossed demo change thresholds, "
            f"with {len(context_factors)} profile context factor(s). Review the sample with the scenario in mind."
        )
    elif crossed_count >= 2:
        status = "wearable_trend"
        summary = "Several wearable trends crossed demo change thresholds; no listed profile context factor was found."
    else:
        status = "no_demo_watch"
        summary = "Fewer than two wearable trends crossed the demo change thresholds. This does not rule out a health problem."

    return {
        "status": status,
        "triggered": status == "cross_stream_watch",
        "summary": summary,
        "changes": changes,
        "context_factors": context_factors,
        "baseline_days": len(baseline),
        "recent_days": len(recent),
        "crossed_count": crossed_count,
        "thresholds": DEMO_THRESHOLDS,
    }
