from collections import Counter
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

import models
from database import Base, SessionLocal, engine, upgrade_demo_schema
from predictor import classify_probabilities
from scoring import alert_level, annual_stroke_risk_pct, cha2ds2_vasc_score

Base.metadata.create_all(bind=engine)
upgrade_demo_schema()

DATA_DIR = "data/ecg"
WINDOWS_PER_PATIENT = 5
RANDOM_SEED = 42


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    model = joblib.load(f"{DATA_DIR}/afib_model.joblib")
    feature_cols = list(model.feature_names_in_)

    ecg = pd.read_csv(f"{DATA_DIR}/ecg_features.csv").dropna(subset=feature_cols)
    # Public ECGs are unrelated to these synthetic profiles, so selection is
    # independent of profile score and includes all reference classes.

    db = SessionLocal()

    # Clear old output so this script can be re-run safely
    db.query(models.ClinicianReviewEvent).delete()
    db.query(models.ClinicianReview).delete()
    db.query(models.Alert).delete()
    db.query(models.RiskAssessment).delete()
    db.query(models.EcgWindow).delete()
    db.commit()

    patients = db.query(models.Patient).all()
    now = models.utc_now_naive()

    # Step 1: decide which ECG recording each patient window gets
    plan = []
    for p in patients:
        score = cha2ds2_vasc_score(p)
        for i in range(WINDOWS_PER_PATIENT):
            row = ecg.iloc[int(rng.integers(len(ecg)))]
            plan.append((p, score, i, row))

    # Step 2: run the model once on all sampled recordings
    X = pd.DataFrame([row[feature_cols] for _, _, _, row in plan]).astype(float)
    class_probabilities = model.predict_proba(X)

    # Step 3: fuse and save
    latest_levels = Counter()
    for (p, score, i, row), probabilities in zip(plan, class_probabilities):
        classification = classify_probabilities(probabilities)
        prob = float(classification["afib_probability"])
        ts = now - timedelta(days=WINDOWS_PER_PATIENT - 1 - i)
        risk_pct = annual_stroke_risk_pct(score)
        conclusive_for_afib = classification["state"] in {"afib_pattern", "no_afib_pattern"}
        fused = prob * risk_pct if conclusive_for_afib else None
        level = alert_level(fused) if fused is not None else "inconclusive"

        db.add(
            models.EcgWindow(
                patient_id=p.id,
                timestamp=ts,
                source_record=row["record"],
                features={c: float(row[c]) for c in feature_cols},
                afib_probability=prob,
            )
        )
        db.add(
            models.RiskAssessment(
                patient_id=p.id,
                timestamp=ts,
                cha2ds2_vasc_score=score,
                afib_component=prob,
                fused_risk_score=fused,
                alert_level=level,
                ecg_result=classification["state"],
            )
        )

        if i == WINDOWS_PER_PATIENT - 1:  # latest reading only
            latest_levels[level] += 1

    db.commit()
    db.close()

    print(f"Created {len(plan)} ECG windows and risk assessments for {len(patients)} patients.")
    print(f"Latest-reading alert levels: {dict(latest_levels)}")


if __name__ == "__main__":
    main()
