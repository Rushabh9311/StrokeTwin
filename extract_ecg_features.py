import numpy as np
import pandas as pd
from scipy.io import loadmat
import neurokit2 as nk

DATA_DIR = "data/ecg"
SAMPLING_RATE = 300  # Hz, per the 2017 Challenge documentation


def load_reference():
    return pd.read_csv(f"{DATA_DIR}/REFERENCE.csv", header=None, names=["record", "label"])


def extract_features_for_record(record_name):
    mat = loadmat(f"{DATA_DIR}/{record_name}.mat")
    signal = mat["val"][0].astype(float)

    cleaned = nk.ecg_clean(signal, sampling_rate=SAMPLING_RATE)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=SAMPLING_RATE)
    r_peaks = info["ECG_R_Peaks"]

    if len(r_peaks) < 4:
        return None

    hrv_time = nk.hrv_time(r_peaks, sampling_rate=SAMPLING_RATE, show=False)

    # Poincare SD1/SD2, computed directly instead of via nk.hrv_nonlinear() —
    # that function also computes DFA/entropy metrics that need much longer
    # recordings than our ~30s clips have, and will crash/hang on them.
    rr = np.diff(r_peaks) / SAMPLING_RATE * 1000  # ms
    diff_rr = np.diff(rr)
    sd1 = np.sqrt(0.5) * np.std(diff_rr, ddof=1)
    sd2 = np.sqrt(max(2 * np.var(rr, ddof=1) - 0.5 * np.var(diff_rr, ddof=1), 0))

    features = hrv_time.iloc[0].to_dict()
    features["HRV_SD1"] = float(sd1)
    features["HRV_SD2"] = float(sd2)
    features["HRV_SD1SD2"] = float(sd1 / sd2) if sd2 else float("nan")
    features["record"] = record_name
    return features


def main():
    ref = load_reference()
    rows = []
    skipped = 0

    for i, row in ref.iterrows():
        try:
            features = extract_features_for_record(row["record"])
        except Exception:
            features = None

        if features is None:
            skipped += 1
            continue

        features["label"] = row["label"]
        features["afib_flag"] = 1 if row["label"] == "A" else 0
        rows.append(features)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(ref)} records...")

    out = pd.DataFrame(rows)
    out.to_csv(f"{DATA_DIR}/ecg_features.csv", index=False)
    print(f"Done. Extracted features for {len(out)} records, skipped {skipped}.")


if __name__ == "__main__":
    main()