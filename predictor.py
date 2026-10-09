"""Experimental ECG rhythm classifier helpers for the local demo."""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from extract_ecg_features import extract_features_for_record

MODEL_PATH = "data/ecg/afib_model.joblib"
model = joblib.load(MODEL_PATH)
FEATURE_COLS = list(model.feature_names_in_)
CLASS_NAMES = {0: "Normal rhythm", 1: "AFib pattern", 2: "Other rhythm", 3: "Noisy recording"}
MIN_CLASS_CONFIDENCE = 0.50


def classify_probabilities(probabilities):
    """Map four-class model probabilities to safe, displayable demo states."""
    scores = np.asarray(probabilities, dtype=float)
    if len(scores) < 4:
        return {
            "state": "legacy_binary_model",
            "predicted_class": "Legacy binary model",
            "class_confidence": float(scores.max()) if len(scores) else 0.0,
            "afib_probability": float(scores[1]) if len(scores) > 1 else 0.0,
            "class_probabilities": {},
        }
    class_index = int(np.argmax(scores))
    confidence = float(scores[class_index])

    if confidence < MIN_CLASS_CONFIDENCE:
        state = "inconclusive_low_confidence"
    elif class_index == 1:
        state = "afib_pattern"
    elif class_index == 0:
        state = "no_afib_pattern"
    elif class_index == 2:
        state = "other_rhythm"
    else:
        state = "noisy_recording"

    return {
        "state": state,
        "predicted_class": CLASS_NAMES.get(class_index, "Unknown"),
        "class_confidence": confidence,
        "afib_probability": float(scores[1]) if len(scores) > 1 else 0.0,
        "class_probabilities": {
            CLASS_NAMES[index]: float(scores[index])
            for index in range(min(len(scores), len(CLASS_NAMES)))
        },
    }


def _predict_features(features):
    X = pd.DataFrame([features])[FEATURE_COLS].astype(float)
    if X.isna().any().any():
        return None
    probabilities = model.predict_proba(X)[0]
    result = classify_probabilities(probabilities)
    result["class_probabilities"] = {
        CLASS_NAMES[int(class_id)]: float(probabilities[position])
        for position, class_id in enumerate(model.classes_)
        if int(class_id) in CLASS_NAMES
    }
    result["afib_probability"] = result["class_probabilities"].get("AFib pattern", 0.0)
    return X, result


def analyze_record_for_replay(record_name: str):
    """Return the four-class result and AFib-class feature contributions."""
    features = extract_features_for_record(record_name)
    if features is None:
        return None
    predicted = _predict_features(features)
    if predicted is None:
        return None
    X, result = predicted

    contribution_values = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    contributions = np.asarray(contribution_values)
    class_position = list(model.classes_).index(1)
    if contributions.ndim == 3:
        if contributions.shape[1] == len(model.classes_):
            contribution_row = contributions[0, class_position, :]
        elif contributions.shape[2] == len(model.classes_):
            contribution_row = contributions[0, :, class_position]
        else:
            contribution_row = np.zeros(len(FEATURE_COLS) + 1)
    elif contributions.ndim == 2 and contributions.shape[0] == 1:
        if contributions.shape[1] == len(model.classes_) * (len(FEATURE_COLS) + 1):
            contribution_row = contributions[0].reshape(
                len(model.classes_), len(FEATURE_COLS) + 1
            )[class_position]
        else:
            contribution_row = contributions[0]
    else:
        contribution_row = np.zeros(len(FEATURE_COLS) + 1)

    result["probability"] = result["afib_probability"]
    result["features"] = [
        {
            "name": feature,
            "value": float(X.iloc[0][feature]),
            "contribution": float(contribution),
        }
        for feature, contribution in zip(FEATURE_COLS, contribution_row[:-1])
    ]
    return result


def afib_probability_for_record(record_name: str):
    """Return P(AFib-pattern) for the demo ECG, or None if it cannot be scored."""
    analysis = analyze_record_for_replay(record_name)
    return None if analysis is None else analysis["afib_probability"]


def classify_record(record_name: str):
    """Classify one short ECG recording, returning None when it cannot be scored."""
    features = extract_features_for_record(record_name)
    if features is None:
        return None
    predicted = _predict_features(features)
    return None if predicted is None else predicted[1]
