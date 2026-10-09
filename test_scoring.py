from database import SessionLocal
import models
from scoring import cha2ds2_vasc_score, annual_stroke_risk_pct

db = SessionLocal()
patients = db.query(models.Patient).limit(10).all()

for p in patients:
    score = cha2ds2_vasc_score(p)
    risk = annual_stroke_risk_pct(score)
    print(
        f"Patient {p.id}: age={p.age}, sex={p.sex}, diabetes={p.diabetes_status}, "
        f"htn={p.hypertension_status}, stroke_hx={p.prior_tia_or_stroke}, "
        f"chf={p.congestive_heart_failure}, vascular={p.vascular_disease} "
        f"-> CHA2DS2-VASc={score}, annual_stroke_risk={risk}%"
    )

db.close()