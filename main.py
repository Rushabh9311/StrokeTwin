from types import SimpleNamespace
from pathlib import Path
import base64
import hmac
import ipaddress
import math
import os
import random
import re
import csv
import json
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import String, cast, func, inspect, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from fastapi import Request

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse
from who_south_asia_risk import estimate_who_south_asia_cvd_risk, risk_band
import neurokit2 as nk
from scipy.io import loadmat

templates = Jinja2Templates(directory="templates")

import models
from database import Base, engine, get_db, upgrade_demo_schema
from extract_ecg_features import DATA_DIR, SAMPLING_RATE
from predictor import analyze_record_for_replay, classify_record
from scoring import (
    alert_level,
    annual_stroke_risk_pct,
    cha2ds2_vasc_breakdown,
    cha2ds2_vasc_score,
)
from wearable_watch import analyze_wearable_trend

Base.metadata.create_all(bind=engine)
upgrade_demo_schema()

app = FastAPI(title="StrokeTwin — Heart Health Workspace")
app.mount("/static", StaticFiles(directory="static"), name="static")


def sync_wearable_watch_alerts(db: Session, patient_ids: list[int] | None = None) -> int:
    """Persist one in-app review suggestion per triggered wearable sample window."""
    patients_query = db.query(models.Patient)
    if patient_ids is not None:
        if not patient_ids:
            return db.query(models.WearableWatchAlert).filter(
                models.WearableWatchAlert.reviewed_at.is_(None)
            ).count()
        patients_query = patients_query.filter(models.Patient.id.in_(patient_ids))
    patients = patients_query.order_by(models.Patient.id).all()
    if not patients:
        return db.query(models.WearableWatchAlert).filter(
            models.WearableWatchAlert.reviewed_at.is_(None)
        ).count()

    ids = [patient.id for patient in patients]
    readings_by_patient = {patient_id: [] for patient_id in ids}
    readings = (
        db.query(models.WearableReading)
        .filter(models.WearableReading.patient_id.in_(ids))
        .order_by(models.WearableReading.patient_id, models.WearableReading.timestamp)
        .all()
    )
    for reading in readings:
        readings_by_patient[reading.patient_id].append(reading)

    existing_rows = db.query(
        models.WearableWatchAlert.patient_id,
        models.WearableWatchAlert.trigger_timestamp,
    ).filter(models.WearableWatchAlert.patient_id.in_(ids)).all()
    existing_keys = set(existing_rows)
    open_ids = {
        patient_id
        for (patient_id,) in db.query(models.WearableWatchAlert.patient_id)
        .filter(
            models.WearableWatchAlert.patient_id.in_(ids),
            models.WearableWatchAlert.reviewed_at.is_(None),
        )
        .all()
    }

    created = False
    for patient in patients:
        if patient.id in open_ids:
            continue
        patient_readings = readings_by_patient[patient.id]
        watch = analyze_wearable_trend(patient_readings, patient)
        if not watch.get("triggered") or not patient_readings:
            continue
        trigger_timestamp = patient_readings[-1].timestamp
        if (patient.id, trigger_timestamp) in existing_keys:
            continue
        snapshot = {
            "summary": watch["summary"],
            "crossed_count": watch.get("crossed_count", 0),
            "changes": [change for change in watch["changes"] if change["crossed"]],
            "context_factors": watch["context_factors"],
        }
        db.add(models.WearableWatchAlert(
            patient_id=patient.id,
            trigger_timestamp=trigger_timestamp,
            snapshot=snapshot,
        ))
        open_ids.add(patient.id)
        existing_keys.add((patient.id, trigger_timestamp))
        created = True
    if created:
        try:
            db.commit()
        except IntegrityError:
            # A concurrent page request may have inserted the same patient/sample.
            db.rollback()
    return db.query(models.WearableWatchAlert).filter(
        models.WearableWatchAlert.reviewed_at.is_(None)
    ).count()


@app.middleware("http")
async def demo_access_guard(request: Request, call_next):
    """Keep the unauthenticated demo local; allow optional shared demo credentials."""
    username = os.getenv("STROKETWIN_DEMO_USER", "").strip()
    password = os.getenv("STROKETWIN_DEMO_PASSWORD", "")
    if bool(username) != bool(password):
        return JSONResponse(
            {"detail": "Set both STROKETWIN_DEMO_USER and STROKETWIN_DEMO_PASSWORD, or clear both."},
            status_code=503,
        )

    if username and password:
        header = request.headers.get("authorization", "")
        supplied_user = supplied_password = ""
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
                supplied_user, separator, supplied_password = decoded.partition(":")
                if not separator:
                    supplied_user = supplied_password = ""
            except (ValueError, UnicodeDecodeError):
                supplied_user = supplied_password = ""
        valid_user = hmac.compare_digest(supplied_user.encode("utf-8"), username.encode("utf-8"))
        valid_password = hmac.compare_digest(supplied_password.encode("utf-8"), password.encode("utf-8"))
        if not (valid_user and valid_password):
            return JSONResponse(
                {"detail": "Authentication required."},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="StrokeTwin research prototype"'},
            )
        request.state.demo_user = username
    else:
        remote = request.client.host if request.client else ""
        try:
            is_local = ipaddress.ip_address(remote).is_loopback
        except ValueError:
            is_local = remote in {"localhost", "testclient"}
        if not is_local:
            return JSONResponse(
                {"detail": "This prototype accepts connections from this computer only. Configure shared credentials to enable remote access."},
                status_code=403,
            )
        request.state.demo_user = ""

    return await call_next(request)


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(request, "landing.html", {})


@app.get("/dashboard")
def dashboard(
    request: Request,
    page: int = Query(1, ge=1),
    band: str = Query("all"),
    search: str = Query("", max_length=80),
    db: Session = Depends(get_db),
):
    total = db.query(models.Patient).count()
    open_watch_alert_count = sync_wearable_watch_alerts(db)

    latest = (
        db.query(
            models.RiskAssessment.patient_id,
            func.max(models.RiskAssessment.timestamp).label("ts"),
        )
        .group_by(models.RiskAssessment.patient_id)
        .subquery()
    )
    patients_query = (
        db.query(models.Patient, models.RiskAssessment)
        .join(models.RiskAssessment, models.RiskAssessment.patient_id == models.Patient.id)
        .join(
            latest,
            (latest.c.patient_id == models.RiskAssessment.patient_id)
            & (latest.c.ts == models.RiskAssessment.timestamp),
        )
    )
    if band not in {"all", "high", "moderate", "low", "inconclusive"}:
        band = "all"
    search = search.strip()
    if band != "all":
        patients_query = patients_query.filter(models.RiskAssessment.alert_level == band)
    if search:
        pattern = f"%{search}%"
        sex_term = {"female": "F", "male": "M"}.get(search.casefold())
        search_filters = [
            cast(models.Patient.id, String).ilike(pattern),
            cast(models.Patient.age, String).ilike(pattern),
        ]
        if sex_term:
            search_filters.append(models.Patient.sex == sex_term)
        else:
            search_filters.append(models.Patient.sex.ilike(pattern))
        patients_query = patients_query.filter(
            or_(*search_filters)
        )

    filtered_total = patients_query.count()
    page_size = 10
    page_count = max(1, math.ceil(filtered_total / page_size))
    page = min(page, page_count)
    rows = (
        patients_query.order_by(models.RiskAssessment.fused_risk_score.desc().nullslast())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    page_patient_ids = [patient.id for patient, _ in rows]
    wearable_by_patient = {patient_id: [] for patient_id in page_patient_ids}
    if page_patient_ids:
        wearable_rows = (
            db.query(models.WearableReading)
            .filter(models.WearableReading.patient_id.in_(page_patient_ids))
            .order_by(models.WearableReading.timestamp.asc())
            .all()
        )
        for reading in wearable_rows:
            wearable_by_patient[reading.patient_id].append(reading)

    patients = [
        {
            "id": p.id,
            "age": p.age,
            "sex": p.sex,
            "score": ra.cha2ds2_vasc_score,
            "afib_probability": ra.afib_component,
            "fused_risk_score": ra.fused_risk_score,
            "alert_level": ra.alert_level,
            "ecg_result": ra.ecg_result,
            "wearable_watch": analyze_wearable_trend(wearable_by_patient[p.id], p),
        }
        for p, ra in rows
    ]

    # Count every patient's latest assessment so the summary cards remain
    # consistent while paging or filtering the table.
    count_rows = (
        db.query(models.RiskAssessment.alert_level, func.count())
        .join(
            latest,
            (latest.c.patient_id == models.RiskAssessment.patient_id)
            & (latest.c.ts == models.RiskAssessment.timestamp),
        )
        .group_by(models.RiskAssessment.alert_level)
        .all()
    )
    counts = {"low": 0, "moderate": 0, "high": 0, "inconclusive": 0}
    for level, count in count_rows:
        if level in counts:
            counts[level] = count

    return templates.TemplateResponse(
        request,
        "patients_list.html",
        {
            "patients": patients,
            "counts": counts,
            "total": total,
            "filtered_total": filtered_total,
            "current_page": page,
            "page_size": page_size,
            "page_count": page_count,
            "band": band,
            "search_query": search,
            "open_watch_alert_count": open_watch_alert_count,
        },
    )


@app.get("/dashboard/review-queue")
def wearable_review_queue(
    request: Request,
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    page_size = 15
    open_watch_alert_count = sync_wearable_watch_alerts(db)
    all_open_rows = (
        db.query(models.WearableWatchAlert, models.Patient)
        .join(models.Patient, models.Patient.id == models.WearableWatchAlert.patient_id)
        .filter(models.WearableWatchAlert.reviewed_at.is_(None))
        .all()
    )
    all_open_rows.sort(key=lambda row: (
        -int((row[0].snapshot or {}).get("crossed_count", 0)),
        row[1].id,
        row[0].created_at,
    ))
    total_open = len(all_open_rows)
    page_count = max(1, math.ceil(total_open / page_size))
    page = min(page, page_count)
    start = (page - 1) * page_size
    open_rows = all_open_rows[start : start + page_size]
    reviewed_rows = (
        db.query(models.WearableWatchAlert, models.Patient)
        .join(models.Patient, models.Patient.id == models.WearableWatchAlert.patient_id)
        .filter(models.WearableWatchAlert.reviewed_at.is_not(None))
        .order_by(models.WearableWatchAlert.reviewed_at.desc())
        .limit(10)
        .all()
    )
    return templates.TemplateResponse(
        request,
        "review_queue.html",
        {
            "alerts": [
                {"alert": alert, "patient": patient, "snapshot": alert.snapshot or {}}
                for alert, patient in open_rows
            ],
            "recently_reviewed": [
                {"alert": alert, "patient": patient}
                for alert, patient in reviewed_rows
            ],
            "open_watch_alert_count": open_watch_alert_count,
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "total_open": total_open,
        },
    )


@app.post("/dashboard/review-queue/{alert_id}/mark-reviewed")
def mark_wearable_alert_reviewed(
    alert_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    alert = db.get(models.WearableWatchAlert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Review suggestion not found")
    if alert.reviewed_at is None:
        alert.reviewed_at = models.utc_now_naive()
        alert.reviewer_name = getattr(request.state, "demo_user", "") or None
        db.commit()
    return RedirectResponse(url="/dashboard/review-queue", status_code=303)


@app.get("/dashboard/patients/{patient_id}")
def dashboard_patient_detail(patient_id: int, request: Request, db: Session = Depends(get_db)):
    p = db.get(models.Patient, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    score = cha2ds2_vasc_score(p)
    breakdown = cha2ds2_vasc_breakdown(p)
    risk_pct = annual_stroke_risk_pct(score)

    assessments = (
        db.query(models.RiskAssessment)
        .filter(models.RiskAssessment.patient_id == patient_id)
        .order_by(models.RiskAssessment.timestamp, models.RiskAssessment.id)
        .all()
    )
    latest = assessments[-1] if assessments else None
    reviews_by_assessment = {}
    if assessments:
        reviews = (
            db.query(models.ClinicianReview)
            .filter(
                models.ClinicianReview.assessment_id.in_(
                    [assessment.id for assessment in assessments]
                )
            )
            .all()
        )
        reviews_by_assessment = {review.assessment_id: review for review in reviews}
    latest_review = reviews_by_assessment.get(latest.id) if latest else None
    review_events = []
    if latest_review:
        review_events = (
            db.query(models.ClinicianReviewEvent)
            .filter(models.ClinicianReviewEvent.review_id == latest_review.id)
            .order_by(models.ClinicianReviewEvent.created_at.asc())
            .all()
        )

    wearable_readings = (
        db.query(models.WearableReading)
        .filter(models.WearableReading.patient_id == patient_id)
        .order_by(models.WearableReading.timestamp)
        .all()
    )
    wearable_history = [
        {
            "date": reading.timestamp.strftime("%b %d"),
            "steps": int(reading.steps),
            "sleep_hours": round(float(reading.sleep_hours), 1),
            "resting_hr": round(float(reading.resting_hr), 1),
            "hrv_ms": round(float(reading.hrv_ms), 1),
        }
        for reading in wearable_readings
    ]
    wearable_summary = {}
    if wearable_readings:
        baseline = wearable_readings[: min(3, len(wearable_readings))]
        current = wearable_readings[-1]
        fields = {
            "steps": ("Daily steps", "steps", "steps", 0),
            "sleep": ("Sleep", "sleep_hours", "hours", 1),
            "resting_hr": ("Resting heart rate", "resting_hr", "bpm", 0),
            "hrv": ("HRV", "hrv_ms", "ms", 0),
        }
        for key, (label, attr, unit, digits) in fields.items():
            baseline_value = sum(float(getattr(row, attr)) for row in baseline) / len(baseline)
            current_value = float(getattr(current, attr))
            change = ((current_value - baseline_value) / baseline_value * 100) if baseline_value else 0
            wearable_summary[key] = {
                "label": label,
                "value": f"{current_value:,.{digits}f}",
                "unit": unit,
                "change": f"{change:+.0f}% from first {len(baseline)} reading(s)",
            }
    wearable_watch = analyze_wearable_trend(wearable_readings, p)
    sync_wearable_watch_alerts(db, [patient_id])

    ecg_windows = (
        db.query(models.EcgWindow)
        .filter(models.EcgWindow.patient_id == patient_id)
        .all()
    )
    record_by_ts = {window.timestamp: window.source_record for window in ecg_windows}

    history = []
    for index, assessment in enumerate(assessments):
        review = reviews_by_assessment.get(assessment.id)
        previous = assessments[index - 1] if index else None
        if previous is None:
            change_summary = "First public ECG sample recorded for this profile."
        else:
            probability_delta = (
                float(assessment.afib_component) - float(previous.afib_component)
            ) * 100
            if abs(probability_delta) < 0.05:
                change_summary = "AFib model probability was similar to the previous sample."
            else:
                direction = "increased" if probability_delta > 0 else "decreased"
                change_summary = (
                    f"AFib model probability {direction} by "
                    f"{abs(probability_delta):.1f} percentage points from the previous sample."
                )
            if assessment.alert_level != previous.alert_level:
                change_summary += (
                    f" The experimental review band changed from "
                    f"{previous.alert_level} to {assessment.alert_level}."
                )

        history.append(
            {
                "assessment_id": assessment.id,
                "date": assessment.timestamp.strftime("%b %d, %Y"),
                "time": assessment.timestamp.strftime("%H:%M"),
                "fused_risk_score": (
                    round(float(assessment.fused_risk_score), 2)
                    if assessment.fused_risk_score is not None
                    else None
                ),
                "afib_probability": round(float(assessment.afib_component) * 100, 1),
                "alert_level": assessment.alert_level,
                "ecg_result": assessment.ecg_result,
                "ecg_record": record_by_ts.get(assessment.timestamp),
                "change_summary": change_summary,
                "reviewed_at": (
                    review.reviewed_at.strftime("%b %d, %Y · %H:%M")
                    if review
                    else None
                ),
                "review_note": review.note if review else "",
                "review_disposition": review.disposition if review else None,
                "reviewer_name": review.reviewer_name if review else None,
            }
        )

    return templates.TemplateResponse(
        request,
        "patient_detail.html",
        {
            "patient": p,
            "score": score,
            "breakdown": breakdown,
            "risk_pct": risk_pct,
            "latest": latest,
            "latest_review": latest_review,
            "review_events": review_events,
            "review_auth_enabled": bool(
                os.getenv("STROKETWIN_DEMO_USER") and os.getenv("STROKETWIN_DEMO_PASSWORD")
            ),
            "demo_reviewer": os.getenv("STROKETWIN_DEMO_USER", "").strip(),
            "history": history,
            "wearable_summary": wearable_summary,
            "wearable_history": wearable_history,
            "wearable_watch": wearable_watch,
        },
    )


class PreventionRiskRequest(BaseModel):
    age: int = Field(ge=40, le=74)
    sex: str
    smoker: bool
    systolic_bp: float = Field(ge=70, le=250)
    bmi: float = Field(ge=10, le=80)
    prior_cvd: bool


@app.post("/dashboard/patients/{patient_id}/prevention-risk")
def prevention_risk_estimate(
    patient_id: int,
    payload: PreventionRiskRequest,
    db: Session = Depends(get_db),
):
    """Look up a WHO South Asia chart value from explicitly entered inputs."""
    if db.get(models.Patient, patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    if payload.prior_cvd:
        raise HTTPException(
            status_code=422,
            detail="This primary-prevention chart is intended for people without known cardiovascular disease.",
        )
    if payload.sex not in {"F", "M"}:
        raise HTTPException(status_code=422, detail="Choose female or male to match the chart categories.")
    try:
        estimate = estimate_who_south_asia_cvd_risk(
            age=payload.age,
            sex=payload.sex,
            smoker=payload.smoker,
            systolic_bp=payload.systolic_bp,
            bmi=payload.bmi,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    actions = []
    if payload.smoker:
        actions.append("Smoking is one of the chart inputs; discuss evidence-based help to quit with a healthcare professional.")
    else:
        actions.append("Continue avoiding tobacco and second-hand smoke.")
    if payload.systolic_bp >= 140:
        actions.append("This BP falls in a higher chart band; confirm it with properly repeated measurements and discuss it with a healthcare professional.")
    else:
        actions.append("Keep checking blood pressure during routine care; one reading is not a diagnosis.")
    if payload.bmi >= 25:
        actions.append("If appropriate for the person, discuss sustainable food and movement goals with their care team.")
    else:
        actions.append("Maintain regular movement and a balanced eating pattern that fits the person's health and preferences.")

    return {
        "percent": estimate,
        "band": risk_band(estimate),
        "actions": actions,
        "note": "WHO 2019 South Asia non-laboratory chart lookup; combined fatal/non-fatal stroke or heart-attack outcome over 10 years.",
    }


@app.get("/dashboard/model-card")
def model_card(request: Request):
    evaluation_path = Path(DATA_DIR) / "afib_evaluation.json"
    calibration_path = Path(DATA_DIR) / "afib_calibration.csv"
    evaluation = None
    calibration_bins = []

    if evaluation_path.is_file():
        with open(evaluation_path, encoding="utf-8") as source:
            evaluation = json.load(source)
        if "classes" not in evaluation or "five_fold_cv" not in evaluation:
            evaluation = None
    if calibration_path.is_file():
        with open(calibration_path, newline="", encoding="utf-8") as source:
            calibration_bins = [
                {
                    "x": float(row["mean_predicted_probability"]),
                    "y": float(row["observed_afib_fraction"]),
                }
                for row in csv.DictReader(source)
            ]

    return templates.TemplateResponse(
        request,
        "model_card.html",
        {
            "evaluation": evaluation,
            "calibration_bins": calibration_bins,
            "needs_training": evaluation is None,
        },
    )


class ClinicianReviewRequest(BaseModel):
    reviewer_name: str = Field(default="", max_length=80)
    disposition: Literal["reviewed", "repeat_sample", "follow_up"]
    note: str = Field(default="", max_length=1000)


@app.post("/dashboard/patients/{patient_id}/review")
def mark_assessment_reviewed(
    patient_id: int,
    payload: ClinicianReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Save a demo review state and optional clinician note for the latest assessment."""
    patient = db.get(models.Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    latest = (
        db.query(models.RiskAssessment)
        .filter(models.RiskAssessment.patient_id == patient_id)
        .order_by(models.RiskAssessment.timestamp.desc())
        .first()
    )
    if latest is None:
        raise HTTPException(status_code=409, detail="There is no assessment to review")

    reviewer_name = getattr(request.state, "demo_user", "") or payload.reviewer_name.strip()
    if len(reviewer_name) < 2:
        raise HTTPException(status_code=422, detail="Enter a reviewer label with at least 2 characters")

    reviewed_at = models.utc_now_naive()
    review = (
        db.query(models.ClinicianReview)
        .filter(models.ClinicianReview.assessment_id == latest.id)
        .first()
    )
    if review is None:
        review = models.ClinicianReview(
            patient_id=patient_id,
            assessment_id=latest.id,
            reviewed_at=reviewed_at,
            note=payload.note.strip(),
            reviewer_name=reviewer_name,
            disposition=payload.disposition,
        )
        db.add(review)
        db.flush()
    else:
        review.reviewed_at = reviewed_at
        review.note = payload.note.strip()
        review.reviewer_name = reviewer_name
        review.disposition = payload.disposition

    db.add(
        models.ClinicianReviewEvent(
            review_id=review.id,
            created_at=reviewed_at,
            reviewer_name=reviewer_name,
            disposition=payload.disposition,
            note=payload.note.strip(),
        )
    )

    db.commit()
    return {
        "status": payload.disposition,
        "reviewed_at": reviewed_at.isoformat(),
        "reviewer_name": reviewer_name,
        "authenticated": bool(getattr(request.state, "demo_user", "")),
    }


@app.post("/dashboard/patients/{patient_id}/wearables/simulate")
def simulate_wearable_reading(patient_id: int, db: Session = Depends(get_db)):
    """Append a visibly synthetic sensor update for the dashboard demo."""
    patient = db.get(models.Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    latest = (
        db.query(models.WearableReading)
        .filter(models.WearableReading.patient_id == patient_id)
        .order_by(models.WearableReading.timestamp.desc())
        .first()
    )
    now = models.utc_now_naive()
    rng = random.Random(f"{patient_id}-{now.isoformat()}")
    if latest:
        reading = models.WearableReading(
            patient_id=patient_id,
            timestamp=now,
            steps=max(0, int(latest.steps + rng.gauss(0, 850))),
            sleep_hours=max(0, min(12, latest.sleep_hours + rng.gauss(0, 0.45))),
            resting_hr=max(35, min(180, latest.resting_hr + rng.gauss(0, 3))),
            hrv_ms=max(5, min(180, latest.hrv_ms + rng.gauss(0, 7))),
        )
    else:
        reading = models.WearableReading(
            patient_id=patient_id,
            timestamp=now,
            steps=rng.randint(2500, 9000),
            sleep_hours=round(rng.uniform(5.5, 8.5), 1),
            resting_hr=rng.randint(58, 88),
            hrv_ms=rng.randint(25, 75),
        )
    db.add(reading)
    db.commit()
    sync_wearable_watch_alerts(db, [patient_id])
    return RedirectResponse(url=f"/dashboard/patients/{patient_id}", status_code=303)


@app.get("/api/ecg/{record_name}")
def replay_ecg_record(record_name: str):
    """Provide a safe, downsampled view of a public ECG demo record."""
    if re.fullmatch(r"A\d{5}", record_name) is None:
        raise HTTPException(status_code=422, detail="Invalid ECG record name")

    record_path = Path(DATA_DIR) / f"{record_name}.mat"
    if not record_path.is_file():
        raise HTTPException(status_code=404, detail="ECG record not found")

    try:
        signal = loadmat(record_path)["val"][0].astype(float)
        cleaned = nk.ecg_clean(signal, sampling_rate=SAMPLING_RATE)
        _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=SAMPLING_RATE)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not process this ECG record") from exc

    analysis = analyze_record_for_replay(record_name)
    if analysis is None:
        raise HTTPException(status_code=422, detail="Too few usable heartbeats in this ECG record")

    stride = max(1, math.ceil(len(cleaned) / 1200))
    sample_indices = list(range(0, len(cleaned), stride))
    if sample_indices[-1] != len(cleaned) - 1:
        sample_indices.append(len(cleaned) - 1)
    waveform = [
        {"x": round(index / SAMPLING_RATE, 3), "y": round(float(cleaned[index]), 5)}
        for index in sample_indices
    ]
    r_peak_indices = [int(index) for index in peak_info["ECG_R_Peaks"]]
    r_peaks = [
        {"x": round(index / SAMPLING_RATE, 3), "y": round(float(cleaned[index]), 5)}
        for index in r_peak_indices
        if 0 <= index < len(cleaned)
    ]
    explanations = sorted(
        analysis["features"],
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )[:6]

    return {
        "record": record_name,
        "duration_seconds": round(len(cleaned) / SAMPLING_RATE, 2),
        "probability": analysis["probability"],
        "ecg_result": analysis["state"],
        "predicted_class": analysis["predicted_class"],
        "class_confidence": analysis["class_confidence"],
        "class_probabilities": analysis["class_probabilities"],
        "waveform": waveform,
        "r_peaks": r_peaks,
        "explanations": explanations,
        "data_note": "Public ECG recording only; it is not this synthetic profile’s ECG and is shown for interface reference.",
    }

@app.get("/health/db")
def db_health():
    inspector = inspect(engine)
    return {"tables": inspector.get_table_names()}


@app.get("/patients")
def list_patients(
    alert_level: str | None = Query(None, pattern="^(low|moderate|high|inconclusive)$"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Patients with their latest risk assessment, highest fused risk first."""
    latest = (
        db.query(
            models.RiskAssessment.patient_id,
            func.max(models.RiskAssessment.timestamp).label("ts"),
        )
        .group_by(models.RiskAssessment.patient_id)
        .subquery()
    )
    query = (
        db.query(models.Patient, models.RiskAssessment)
        .join(models.RiskAssessment, models.RiskAssessment.patient_id == models.Patient.id)
        .join(
            latest,
            (latest.c.patient_id == models.RiskAssessment.patient_id)
            & (latest.c.ts == models.RiskAssessment.timestamp),
        )
    )
    if alert_level:
        query = query.filter(models.RiskAssessment.alert_level == alert_level)

    rows = query.order_by(models.RiskAssessment.fused_risk_score.desc().nullslast()).limit(limit).all()

    return [
        {
            "id": p.id,
            "age": p.age,
            "sex": p.sex,
            "cha2ds2_vasc_score": ra.cha2ds2_vasc_score,
            "afib_probability": ra.afib_component,
            "fused_risk_score": ra.fused_risk_score,
            "alert_level": ra.alert_level,
            "ecg_result": ra.ecg_result,
            "last_updated": ra.timestamp,
        }
        for p, ra in rows
    ]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Full 'virtual patient' view: profile, score breakdown, risk history."""
    p = db.get(models.Patient, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    score = cha2ds2_vasc_score(p)

    assessments = (
        db.query(models.RiskAssessment)
        .filter(models.RiskAssessment.patient_id == patient_id)
        .order_by(models.RiskAssessment.timestamp)
        .all()
    )
    windows = (
        db.query(models.EcgWindow)
        .filter(models.EcgWindow.patient_id == patient_id)
        .all()
    )
    record_by_ts = {w.timestamp: w.source_record for w in windows}

    return {
        "patient": {
            "id": p.id,
            "age": p.age,
            "sex": p.sex,
            "diabetes": p.diabetes_status,
            "hypertension": p.hypertension_status,
            "prior_stroke_or_tia": p.prior_tia_or_stroke,
            "congestive_heart_failure": p.congestive_heart_failure,
            "vascular_disease": p.vascular_disease,
        },
        "cha2ds2_vasc": {
            "score": score,
            "annual_stroke_risk_pct": annual_stroke_risk_pct(score),
            "breakdown": cha2ds2_vasc_breakdown(p),
        },
        "history": [
            {
                "timestamp": a.timestamp,
                "ecg_record": record_by_ts.get(a.timestamp),
                "afib_probability": a.afib_component,
                "fused_risk_score": a.fused_risk_score,
                "alert_level": a.alert_level,
                "ecg_result": a.ecg_result,
            }
            for a in assessments
        ],
    }


class PredictRequest(BaseModel):
    age: int = Field(ge=18, le=120)
    sex: str = Field(pattern="^(M|F)$")
    diabetes: bool = False
    hypertension: bool = False
    prior_stroke_or_tia: bool = False
    congestive_heart_failure: bool = False
    vascular_disease: bool = False
    # A recording ID from the dataset, e.g. "A00004". Strict pattern so
    # nobody can use it to read arbitrary files from disk.
    ecg_record: str | None = Field(default=None, pattern=r"^A\d{5}$")


@app.post("/predict")
def predict(req: PredictRequest):
    patient = SimpleNamespace(
        age=req.age,
        sex=req.sex,
        diabetes_status=req.diabetes,
        hypertension_status=req.hypertension,
        prior_tia_or_stroke=req.prior_stroke_or_tia,
        congestive_heart_failure=req.congestive_heart_failure,
        vascular_disease=req.vascular_disease,
    )
    breakdown = cha2ds2_vasc_breakdown(patient)
    score = sum(item["points"] for item in breakdown)
    risk_pct = annual_stroke_risk_pct(score)

    result = {
        "cha2ds2_vasc": {
            "score": score,
            "annual_stroke_risk_pct": risk_pct,
            "breakdown": breakdown,
        },
        "afib_probability": None,
        "ecg_result": None,
        "predicted_class": None,
        "class_confidence": None,
        "class_probabilities": None,
        "fused_risk_score": None,
        "alert_level": None,
        "note": "No ECG supplied: the annual stroke risk applies only if the patient is in AFib.",
    }
    if req.ecg_record is None:
        return result

    try:
        classification = classify_record(req.ecg_record)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="ECG record not found")
    if classification is None:
        raise HTTPException(status_code=422, detail="Too few heartbeats detected in this ECG record")

    result.update(
        afib_probability=classification["afib_probability"],
        ecg_result=classification["state"],
        predicted_class=classification["predicted_class"],
        class_confidence=classification["class_confidence"],
        class_probabilities=classification["class_probabilities"],
    )
    if classification["state"] not in {"afib_pattern", "no_afib_pattern"}:
        result.update(
            fused_risk_score=None,
            alert_level="inconclusive",
            note="The ECG model marked this sample inconclusive; no combined score was calculated.",
        )
    else:
        fused = classification["afib_probability"] * risk_pct
        result.update(
            fused_risk_score=fused,
            alert_level=alert_level(fused),
            note="Experimental display calculation only; it is not a personal stroke prediction.",
        )
    return result
