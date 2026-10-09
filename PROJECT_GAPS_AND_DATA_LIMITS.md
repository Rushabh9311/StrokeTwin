# StrokeTwin: validation and data-use boundaries

StrokeTwin is a local research demonstration using simulated profiles, public ECG recordings, and simulated wearable readings. It is not validated for diagnosis, stroke prediction, treatment, or use with real patient records.

## Cross-stream demo watch

- The dashboard compares the first three wearable readings with the latest three and creates an in-app “Review suggested” item only when two or more illustrative changes are crossed and at least one profile context factor is present. It keeps at most one open item per profile; the queue snapshots the reasons, orders them by crossed-threshold count and profile ID, paginates 15 at a time, and supports a demo mark-reviewed action.
- The thresholds are demonstration choices, not medically established cutoffs. A reproducible subset of generated profiles receives a scripted time-series change so the interface can show a scenario. Queue entries are workflow prompts only; they are not medical alerts and do not send push or emergency notifications.
- This is a transparent rule-based pattern watch, not an outcome-prediction model. It does not estimate stroke, AFib, or any future event. Public ECG recordings are randomly assigned to synthetic profiles only for demonstration; they are not that profile's own ECG and are not used by the wearable watch. The dashboard also displays an experimental combined score from the assigned ECG's AFib-class probability and the profile's CHA₂DS₂-VASc reference rate; this is not a calibrated outcome risk.
- A real outcome model requires paired longitudinal EHR and sensor data with defined event labels, patient-level evaluation, and external validation.

## ECG model

- The ECG model has four reference classes: normal rhythm, AFib, other rhythm, and noisy recording. Other-rhythm, noisy, and low-confidence results are shown as inconclusive for AFib.
- The saved evaluation is a stratified record-level holdout plus five-fold record-level cross-validation. The public source files available in this project do not include subject identifiers, so a patient-grouped split cannot be calculated from them.
- The currently available data does not provide an independent external cohort. Model metrics are dataset experiments only. Cross-validation does not establish clinical performance.
- Retrain with `python .\train_afib_classifier.py` to create the updated four-class model and report. Then run `python .\fuse_patients.py` to regenerate the demo assessments. The fusion script clears prior ECG windows, assessments, legacy alert rows, and ECG review history before rebuilding ECG windows and assessments. Wearable review suggestions are separate and are created from the wearable watch.

## Scores and review workflow

- The WHO South Asia form is a separate 10-year combined stroke-or-heart-attack estimate. It does not return stroke-only risk.
- CHA₂DS₂-VASc is displayed as a factor score and historical reference rate for people with AFib. Absolute rates vary by cohort. The experimental combined display score and its bands are not clinical risk estimates.
- Reviewer labels and dispositions are demo workflow entries. Without configured access credentials, reviewer names are not verified. The event table retains updates for this local prototype.

## Local access and real-data readiness

- By default the app accepts requests from the same computer only. Configure both `STROKETWIN_DEMO_USER` and `STROKETWIN_DEMO_PASSWORD` in the environment (or `.env`) to require HTTP Basic authentication and allow remote demo access.
- Basic authentication must run behind HTTPS when traffic leaves the local machine. The configured account is shared; it does not provide individual roles or a clinical identity system.
- Do not enter or import real patient information. Real-world use would require an appropriate identity and role system, transport and storage protections, access and retention controls, deployment review, and clinical validation on representative data. This repository does not implement or establish those requirements.
