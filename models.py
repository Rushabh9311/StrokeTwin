from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from database import Base


def utc_now_naive():
    """Return UTC now as a naive datetime for the existing SQL DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    synthea_patient_id = Column(String, unique=True, index=True, nullable=True)
    age = Column(Integer)
    sex = Column(String)
    diabetes_status = Column(Boolean, default=False)
    hba1c = Column(Float, nullable=True)
    hypertension_status = Column(Boolean, default=False)
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    prior_tia_or_stroke = Column(Boolean, default=False)
    congestive_heart_failure = Column(Boolean, default=False)
    vascular_disease = Column(Boolean, default=False)

    ecg_windows = relationship("EcgWindow", back_populates="patient")
    risk_assessments = relationship("RiskAssessment", back_populates="patient")
    alerts = relationship("Alert", back_populates="patient")
    wearable_readings = relationship("WearableReading", back_populates="patient")


class EcgWindow(Base):
    __tablename__ = "ecg_windows"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    timestamp = Column(DateTime, default=utc_now_naive)
    source_record = Column(String, nullable=True)
    features = Column(JSON)
    afib_probability = Column(Float, nullable=True)

    patient = relationship("Patient", back_populates="ecg_windows")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    timestamp = Column(DateTime, default=utc_now_naive)
    cha2ds2_vasc_score = Column(Integer)
    afib_component = Column(Float)
    fused_risk_score = Column(Float)
    alert_level = Column(String)
    ecg_result = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="risk_assessments")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    timestamp = Column(DateTime, default=utc_now_naive)
    message = Column(String)
    acknowledged = Column(Boolean, default=False)

    patient = relationship("Patient", back_populates="alerts")


class WearableWatchAlert(Base):
    """Persist a review suggestion for one triggered wearable sample window."""
    __tablename__ = "wearable_watch_alerts"
    __table_args__ = (
        UniqueConstraint("patient_id", "trigger_timestamp", name="uq_watch_patient_sample"),
    )

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    trigger_timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=utc_now_naive, nullable=False)
    snapshot = Column(JSON, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_name = Column(String(80), nullable=True)

    patient = relationship("Patient")


class WearableReading(Base):
    __tablename__ = "wearable_readings"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    timestamp = Column(DateTime, default=utc_now_naive, index=True)
    steps = Column(Integer)
    sleep_hours = Column(Float)
    resting_hr = Column(Float)
    hrv_ms = Column(Float)

    patient = relationship("Patient", back_populates="wearable_readings")


class ClinicianReview(Base):
    __tablename__ = "clinician_reviews"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True, nullable=False)
    assessment_id = Column(Integer, ForeignKey("risk_assessments.id"), unique=True, nullable=False)
    reviewed_at = Column(DateTime, default=utc_now_naive, nullable=False)
    note = Column(Text, default="", nullable=False)
    reviewer_name = Column(String(80), nullable=True)
    disposition = Column(String(24), nullable=True)


class ClinicianReviewEvent(Base):
    __tablename__ = "clinician_review_events"

    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(Integer, ForeignKey("clinician_reviews.id"), index=True, nullable=False)
    created_at = Column(DateTime, default=utc_now_naive, nullable=False)
    reviewer_name = Column(String(80), nullable=False)
    disposition = Column(String(24), nullable=False)
    note = Column(Text, default="", nullable=False)

    review = relationship("ClinicianReview")
