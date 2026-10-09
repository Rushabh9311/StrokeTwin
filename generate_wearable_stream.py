"""Create a reproducible, explicitly synthetic one-week wearable stream."""

from datetime import timedelta
import random

import models
from database import Base, SessionLocal, engine


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        patients = db.query(models.Patient).order_by(models.Patient.id).all()
        db.query(models.WearableWatchAlert).delete()
        db.query(models.WearableReading).delete()

        now = models.utc_now_naive()
        first_day = now.replace(hour=8, minute=0, second=0, microsecond=0) - timedelta(days=6)
        total = 0
        for patient in patients:
            rng = random.Random(2026 + patient.id)
            # A reproducible subset of synthetic profiles gets a deliberately
            # visible trend change so the cross-stream demo workflow can be shown.
            # This is a scripted scenario, not a simulated disease or outcome.
            change_scenario = rng.random() < 0.25
            steps_base = rng.randint(3000, 9000)
            sleep_base = rng.uniform(6.0, 8.0)
            resting_hr_base = rng.randint(60, 82)
            hrv_base = rng.randint(28, 68)

            for day in range(7):
                # These values are generated for interface and pipeline demos;
                # they are not observations or medical reference ranges.
                progress = max(0, day - 3) if change_scenario else 0
                db.add(
                    models.WearableReading(
                        patient_id=patient.id,
                        timestamp=first_day + timedelta(days=day),
                        steps=max(0, int(steps_base * (1 - 0.18 * progress) + rng.gauss(0, 500))),
                        sleep_hours=round(max(0, min(12, sleep_base * (1 - 0.12 * progress) + rng.gauss(0, 0.25))), 1),
                        resting_hr=round(max(35, min(180, resting_hr_base * (1 + 0.08 * progress) + rng.gauss(0, 2))), 1),
                        hrv_ms=round(max(5, min(180, hrv_base * (1 - 0.15 * progress) + rng.gauss(0, 4))), 1),
                    )
                )
                total += 1

        db.commit()
        print(f"Generated {total} synthetic wearable readings for {len(patients)} patient profiles.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
