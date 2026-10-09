from database import SessionLocal
import models
from scoring import cha2ds2_vasc_score, annual_stroke_risk_pct

PATIENT_IDS = [785, 392, 663, 740, 516, 545, 622, 301, 205, 808]

db = SessionLocal()
patients = db.query(models.Patient).filter(models.Patient.id.in_(PATIENT_IDS)).all()

for p in sorted(patients, key=lambda x: -x.age):
    score = cha2ds2_vasc_score(p)
    risk = annual_stroke_risk_pct(score)
    print(
        f"Patient {p.id}: age={p.age}, sex={p.sex}, diabetes={p.diabetes_status}, "
        f"htn={p.hypertension_status}, stroke_hx={p.prior_tia_or_stroke}, "
        f"chf={p.congestive_heart_failure}, vascular={p.vascular_disease} "
        f"-> CHA2DS2-VASc={score}, annual_stroke_risk={risk}%"
    )

db.close()