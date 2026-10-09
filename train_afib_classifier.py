"""Train and evaluate the four-class ECG demo classifier."""

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data/ecg"
LABEL_TO_ID = {"N": 0, "A": 1, "O": 2, "~": 3}
CLASS_NAMES = ["Normal rhythm", "AFib pattern", "Other rhythm", "Noisy recording"]
MIN_CLASS_CONFIDENCE = 0.50


def make_model(n_estimators=260):
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )


def display_state(probabilities):
    best = int(np.argmax(probabilities))
    if float(probabilities[best]) < MIN_CLASS_CONFIDENCE:
        return "Inconclusive"
    return ["No AFib pattern detected", "AFib pattern detected", "Inconclusive", "Inconclusive"][best]


def main():
    df = pd.read_csv(f"{DATA_DIR}/ecg_features.csv")

    # Long-window HRV features cannot be reliably computed from these short ECG clips.
    unreliable = {
        "HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2",
        "HRV_SDANN5", "HRV_SDNNI5",
    }
    features = [
        column for column in df.columns
        if column not in {"record", "label", "afib_flag"} and column not in unreliable
    ]
    df = df[df["label"].isin(LABEL_TO_ID)].dropna(subset=features).copy()
    df["class_id"] = df["label"].map(LABEL_TO_ID).astype(int)

    X = df[features].astype(float)
    y = df["class_id"]
    X_train, X_test, y_train, y_test, records_train, records_test = train_test_split(
        X,
        y,
        df["record"],
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train
    )
    row_weights = y_train.map(dict(enumerate(class_weights))).to_numpy()
    model = make_model()
    model.fit(X_train, y_train, sample_weight=row_weights)

    probabilities = model.predict_proba(X_test)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=list(range(len(CLASS_NAMES))))
    y_afib = (y_test == LABEL_TO_ID["A"]).astype(int)
    predicted_afib = (predictions == LABEL_TO_ID["A"]).astype(int)
    observed, predicted = calibration_curve(
        y_afib, probabilities[:, LABEL_TO_ID["A"]], n_bins=10, strategy="uniform"
    )
    afib_confusion = confusion_matrix(y_afib, predicted_afib, labels=[0, 1])
    tn, fp, fn, tp = afib_confusion.ravel()

    cv_rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    for fold_train, fold_valid in cv.split(X, y):
        fold_y = y.iloc[fold_train]
        fold_classes = np.arange(len(CLASS_NAMES))
        fold_weights = compute_class_weight(
            class_weight="balanced", classes=fold_classes, y=fold_y
        )
        fold_row_weights = fold_y.map(dict(enumerate(fold_weights))).to_numpy()
        fold_model = make_model()
        fold_model.fit(X.iloc[fold_train], fold_y, sample_weight=fold_row_weights)
        fold_predictions = fold_model.predict(X.iloc[fold_valid])
        fold_truth = y.iloc[fold_valid]
        cv_rows.append({
            "accuracy": float(accuracy_score(fold_truth, fold_predictions)),
            "macro_f1": float(f1_score(fold_truth, fold_predictions, average="macro", zero_division=0)),
            "afib_recall": float(recall_score(
                (fold_truth == LABEL_TO_ID["A"]).astype(int),
                (fold_predictions == LABEL_TO_ID["A"]).astype(int),
                zero_division=0,
            )),
        })

    cv_summary = {
        key: {
            "mean": float(np.mean([row[key] for row in cv_rows])),
            "std": float(np.std([row[key] for row in cv_rows], ddof=1)),
        }
        for key in ("accuracy", "macro_f1", "afib_recall")
    }

    metrics = {
        "model": "XGBoost four-class short-ECG rhythm classifier",
        "classes": CLASS_NAMES,
        "decision_rule": (
            "Maximum class confidence below 0.50 is inconclusive; other rhythm and noisy "
            "recording classes are also shown as inconclusive for AFib screening."
        ),
        "minimum_class_confidence": MIN_CLASS_CONFIDENCE,
        "split": "Stratified random recording-level holdout (80/20, random_state=42). The source files do not contain subject IDs, so a patient-grouped split is unavailable.",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "five_fold_cv": cv_summary,
        "afib_ovr_roc_auc": float(roc_auc_score(y_afib, probabilities[:, LABEL_TO_ID["A"]])),
        "afib_ovr_average_precision": float(
            average_precision_score(y_afib, probabilities[:, LABEL_TO_ID["A"]])
        ),
        "afib_sensitivity": float(recall_score(y_afib, predicted_afib, zero_division=0)),
        "afib_specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "afib_precision": float(precision_score(y_afib, predicted_afib, zero_division=0)),
        "afib_confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "operational_state_counts": pd.Series(
            [display_state(row) for row in probabilities]
        ).value_counts().to_dict(),
        "evaluation_note": (
            "Dataset-only experiment on the public PhysioNet 2017 training recordings. "
            "No patient identifiers or external clinical validation are available; do not "
            "interpret these metrics as clinical performance."
        ),
    }
    with open(f"{DATA_DIR}/afib_evaluation.json", "w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    pd.DataFrame(
        {"mean_predicted_probability": predicted, "observed_afib_fraction": observed}
    ).to_csv(f"{DATA_DIR}/afib_calibration.csv", index=False)
    joblib.dump(model, f"{DATA_DIR}/afib_model.joblib")

    print(classification_report(y_test, predictions, labels=list(range(4)), target_names=CLASS_NAMES, zero_division=0))
    print(f"Holdout accuracy: {metrics['accuracy']:.3f}; macro F1: {metrics['macro_f1']:.3f}")
    print(f"5-fold CV macro F1: {cv_summary['macro_f1']['mean']:.3f} ± {cv_summary['macro_f1']['std']:.3f}")
    print(f"AFib one-vs-rest ROC-AUC: {metrics['afib_ovr_roc_auc']:.3f}")
    print(f"Training records: {len(records_train)}; held-out records: {len(records_test)}")
    print("Saved model, multiclass evaluation, and AFib calibration bins.")


if __name__ == "__main__":
    main()
