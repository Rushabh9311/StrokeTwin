"""WHO 2019 South Asia non-laboratory CVD chart lookup.

The chart estimates 10-year fatal or non-fatal cardiovascular events (stroke
or myocardial infarction). It is not a stroke-only score or a diagnosis.
Values are transcribed from the WHO non-laboratory South Asia chart (page 13).
"""

from __future__ import annotations


# For each age band and SBP band, values follow the WHO chart's 20 cells:
# men non-smoker (5 BMI bands), men smoker (5), women non-smoker (5),
# women smoker (5). BMI bands are <20, 20-24, 25-29, 30-35, >=35 kg/m².
_ROWS = {
    70: [
        ("ge180", "24 26 28 30 32 31 33 35 37 40 21 21 22 23 24 29 30 31 32 33"),
        ("160", "20 22 23 25 27 26 28 30 32 34 17 18 19 19 20 25 26 26 27 28"),
        ("140", "17 18 19 21 22 22 23 25 27 28 15 15 16 16 17 21 22 22 23 24"),
        ("120", "14 15 16 17 18 18 19 21 22 24 12 13 13 14 14 18 18 19 20 20"),
        ("lt120", "11 12 13 14 15 15 16 17 18 20 10 11 11 12 12 15 15 16 17 17"),
    ],
    65: [
        ("ge180", "19 20 22 24 26 26 28 30 33 36 16 17 18 18 19 25 26 27 28 29"),
        ("160", "15 17 18 20 22 21 23 25 27 30 13 14 14 15 16 21 21 22 23 24"),
        ("140", "12 14 15 16 18 17 19 21 22 25 11 11 12 12 13 17 18 19 19 20"),
        ("120", "10 11 12 13 14 14 15 17 18 20 9 9 10 10 11 14 15 15 16 17"),
        ("lt120", "8 9 10 11 12 11 12 14 15 16 7 8 8 8 9 12 12 13 13 14"),
    ],
    60: [
        ("ge180", "15 16 18 20 22 21 24 26 29 32 13 13 14 14 15 21 22 23 24 26"),
        ("160", "12 13 14 16 18 17 19 21 24 26 10 11 11 12 12 17 18 19 20 21"),
        ("140", "9 10 11 13 14 14 15 17 19 21 8 9 9 9 10 14 15 15 16 17"),
        ("120", "7 8 9 10 11 11 12 14 15 17 7 7 7 8 8 11 12 12 13 14"),
        ("lt120", "6 6 7 8 9 9 10 11 12 13 5 5 6 6 6 9 9 10 11 11"),
    ],
    55: [
        ("ge180", "11 13 14 16 18 18 20 23 26 29 10 10 11 11 12 18 19 20 21 22"),
        ("160", "9 10 11 13 14 14 16 18 20 23 8 8 9 9 10 14 15 16 17 18"),
        ("140", "7 8 9 10 11 11 12 14 16 18 6 6 7 7 7 11 12 13 13 14"),
        ("120", "5 6 7 8 9 9 10 11 12 14 5 5 5 6 6 9 9 10 11 11"),
        ("lt120", "4 5 5 6 7 7 7 8 10 11 4 4 4 4 5 7 7 8 8 9"),
    ],
    50: [
        ("ge180", "9 10 11 13 15 15 17 20 22 26 8 8 9 9 10 15 16 17 18 19"),
        ("160", "7 8 9 10 11 11 13 15 17 20 6 6 7 7 7 12 13 13 14 15"),
        ("140", "5 6 7 8 9 9 10 12 13 15 5 5 5 5 6 9 10 10 11 12"),
        ("120", "4 4 5 6 7 7 8 9 10 12 3 4 4 4 4 7 7 8 9 9"),
        ("lt120", "3 3 4 4 5 5 6 7 8 9 3 3 3 3 3 5 6 6 7 7"),
    ],
    45: [
        ("ge180", "7 8 9 11 12 12 14 17 20 23 6 6 7 7 8 13 14 15 16 17"),
        ("160", "5 6 7 8 9 9 11 13 15 17 5 5 5 5 6 10 10 11 12 13"),
        ("140", "4 4 5 6 7 7 8 9 11 13 3 4 4 4 4 7 8 8 9 10"),
        ("120", "3 3 4 4 5 5 6 7 8 10 2 3 3 3 3 6 6 6 7 7"),
        ("lt120", "2 2 3 3 4 4 4 5 6 7 2 2 2 2 2 4 4 5 5 6"),
    ],
    40: [
        ("ge180", "5 6 7 9 10 10 12 14 17 20 5 5 5 6 6 11 12 13 14 15"),
        ("160", "4 5 5 6 7 8 9 11 13 15 3 4 4 4 4 8 9 9 10 11"),
        ("140", "3 3 4 5 5 5 6 8 9 11 2 3 3 3 3 6 6 7 7 8"),
        ("120", "2 2 3 3 4 4 5 6 7 8 2 2 2 2 2 4 5 5 6 6"),
        ("lt120", "2 2 2 2 3 3 3 4 5 6 1 1 1 2 2 3 3 4 4 4"),
    ],
}

_CHART: dict[tuple[int, str], tuple[int, ...]] = {}
for age_floor, rows in _ROWS.items():
    for sbp_band, values in rows:
        parsed = tuple(int(value) for value in values.split())
        if len(parsed) != 20:
            raise RuntimeError(f"Bad WHO chart row: {age_floor}/{sbp_band}")
        _CHART[(age_floor, sbp_band)] = parsed


def estimate_who_south_asia_cvd_risk(
    *, age: int, sex: str, smoker: bool, systolic_bp: float, bmi: float
) -> int:
    """Return the integer percent read from the WHO chart for valid inputs."""
    if not 40 <= age <= 74:
        raise ValueError("The WHO South Asia chart supports ages 40–74.")
    if sex not in {"F", "M"}:
        raise ValueError("Choose female or male to match the chart categories.")
    if not 70 <= systolic_bp <= 250:
        raise ValueError("Enter a systolic blood pressure from 70 to 250 mmHg.")
    if not 10 <= bmi <= 80:
        raise ValueError("Enter a BMI from 10 to 80 kg/m².")

    age_floor = 40 + ((age - 40) // 5) * 5
    if systolic_bp >= 180:
        sbp_band = "ge180"
    elif systolic_bp >= 160:
        sbp_band = "160"
    elif systolic_bp >= 140:
        sbp_band = "140"
    elif systolic_bp >= 120:
        sbp_band = "120"
    else:
        sbp_band = "lt120"

    bmi_band = 0 if bmi < 20 else 1 if bmi < 25 else 2 if bmi < 30 else 3 if bmi < 35 else 4
    sex_offset = 0 if sex == "M" else 10
    smoking_offset = 5 if smoker else 0
    return _CHART[(age_floor, sbp_band)][sex_offset + smoking_offset + bmi_band]


def risk_band(percent: int) -> str:
    if percent < 5:
        return "<5%"
    if percent < 10:
        return "5% to <10%"
    if percent < 20:
        return "10% to <20%"
    if percent < 30:
        return "20% to <30%"
    return "≥30%"
