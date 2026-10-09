from datetime import date

import pandas as pd

import models
from database import Base, SessionLocal, engine

Base.metadata.create_all(bind=engine)

DATA_DIR = "data/synthea/csv"
REFERENCE_DATE = date(2026, 9, 23)  # "today" for age calculation
MIN_AGE = 18  # CHA2DS2-VASc is only clinically meaningful for adults

VASCULAR_DESCRIPTIONS = {
    "Coronary Heart Disease",
    "Myocardial Infarction",
    "History of myocardial infarction (situation)",
}


def compute_age(birthdate):
    birth = pd.to_datetime(birthdate).date()
    return REFERENCE_DATE.year - birth.year - (
        (REFERENCE_DATE.month, REFERENCE_DATE.day) < (birth.month, birth.day)
    )


def main():
    patients_df = pd.read_csv(f"{DATA_DIR}/patients.csv")
    conditions_df = pd.read_csv(f"{DATA_DIR}/conditions.csv")

    # Only living, adult patients — CHA2DS2-VASc doesn't apply to children
    patients_df = patients_df[patients_df["DEATHDATE"].isna()].copy()
    patients_df["AGE"] = patients_df["BIRTHDATE"].apply(compute_age)
    before = len(patients_df)
    patients_df = patients_df[patients_df["AGE"] >= MIN_AGE]
    print(f"Filtered {before} living patients down to {len(patients_df)} adults (age >= {MIN_AGE})")

    diabetes_ids = set(conditions_df.loc[conditions_df["DESCRIPTION"] == "Diabetes", "PATIENT"])
    hypertension_ids = set(conditions_df.loc[conditions_df["DESCRIPTION"] == "Hypertension", "PATIENT"])
    stroke_ids = set(conditions_df.loc[conditions_df["DESCRIPTION"] == "Stroke", "PATIENT"])
    chf_ids = set(
        conditions_df.loc[
            conditions_df["DESCRIPTION"] == "Chronic congestive heart failure (disorder)", "PATIENT"
        ]
    )
    vascular_ids = set(conditions_df.loc[conditions_df["DESCRIPTION"].isin(VASCULAR_DESCRIPTIONS), "PATIENT"])

    db = SessionLocal()
    inserted = 0

    for _, row in patients_df.iterrows():
        pid = row["Id"]
        db.add(
            models.Patient(
                synthea_patient_id=pid,
                age=int(row["AGE"]),
                sex=row["GENDER"],
                diabetes_status=pid in diabetes_ids,
                hypertension_status=pid in hypertension_ids,
                prior_tia_or_stroke=pid in stroke_ids,
                congestive_heart_failure=pid in chf_ids,
                vascular_disease=pid in vascular_ids,
            )
        )
        inserted += 1

    db.commit()
    db.close()
    print(f"Inserted {inserted} patients.")


if __name__ == "__main__":
    main()