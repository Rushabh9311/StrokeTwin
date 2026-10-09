# StrokeTwin

StrokeTwin is a local hackathon prototype for exploring a clinician-facing cardiovascular review workflow. It combines synthetic EHR context with a simulated one-week wearable trend to produce a transparent **demo review watch**. A separate XGBoost module classifies public short ECG samples into rhythm categories.

**This is not a clinical tool.** The watch thresholds are illustrative. A triggered watch creates an in-app “Review suggested” item; it is a workflow prompt, not a medical alert. Public ECG recordings are randomly assigned to synthetic profiles for demonstration; they are not recordings from those profiles. For conclusive ECG results only, the dashboard's experimental combined display score multiplies the assigned ECG's AFib-class probability by the profile's CHA₂DS₂-VASc reference rate; inconclusive results receive no combined score. The wearable watch does not use ECG data, and the app does not predict stroke or another clinical event. Do not enter real patient information.

## Team and submission details

Fill these in before submitting:

- **Team name:** Entropy
- **College / incubator:** DY Patil College of Engineering, Pune
- **Team members:** Rushabh Chopada and Anjali Patil
- **Project title:** StrokeTwin — [CONFIRM FINAL SUBMISSION TITLE]
- **20-minute demo video:** [ADD UNLISTED VIDEO LINK]
- **Public repository:** [ADD PUBLIC GITHUB URL]
- **Architecture handout:** [StrokeTwin_Architecture.pdf](StrokeTwin_Architecture.pdf)
- **Presentation deck:** [StrokeTwin_Deck.pptx](StrokeTwin_Deck.pptx)

## What the prototype does

1. **Static context:** imports adult synthetic profiles and selected diagnoses from the Synthea CSV export.
2. **Dynamic stream:** generates seven days of synthetic steps, sleep, resting heart rate, and HRV. A reproducible subset of profiles contains a deliberately visible simulated trend shift for demonstrating the workflow.
3. **Cross-stream demo watch and review queue:** compares the first three readings with the latest three. When at least two illustrative wearable thresholds are crossed and the synthetic profile has at least one listed context factor, it creates an in-app “Review suggested” item. The queue shows the trigger changes and profile context, links to the profile, and lets the reviewer mark the suggestion reviewed.
4. **ECG experiment and experimental score:** an XGBoost classifier outputs Normal, AFib pattern, Other Rhythm, or Noisy Recording for a public PhysioNet recording. Other, noisy, and low-confidence results are shown as inconclusive and receive no combined score. A public ECG sample is assigned randomly, independent of the profile. For conclusive results only, the dashboard's experimental combined display score multiplies its AFib-class probability by that profile's CHA₂DS₂-VASc reference rate. This is not a calibrated personal stroke-risk estimate. The separate wearable watch excludes ECG data.
5. **Separate prevention reference:** an optional WHO South Asia chart lookup returns a 10-year combined cardiovascular risk estimate (fatal/non-fatal stroke or heart attack), not stroke-only risk.
6. **Clinician interaction concept:** clinicians can inspect sample history, replay an ECG recording, record a demo review disposition, view wearable trends, and inspect factor explanations.

## Cross-stream watch logic

The rule compares mean values from the first three readings with the latest three:

| Signal | Illustrative change threshold |
| --- | ---: |
| Resting heart rate | Increase of at least 10% |
| HRV | Decrease of at least 20% |
| Sleep | Decrease of at least 15% |
| Steps | Decrease of at least 25% |

At least two crossed thresholds plus at least one profile context factor produce a **Cross-stream demo watch** and an in-app review suggestion. These values were chosen to make a transparent synthetic demonstration; they are not medically established thresholds, probabilities, or event labels. A missing watch does not mean a person is healthy. The current watch detects a change pattern; it does not forecast a validated adverse outcome.

## Wearable review queue

The queue keeps at most one open review suggestion per profile, even if more simulated readings are added before it is reviewed. It snapshots the crossed changes and synthetic context used by the rule, so the explanation remains visible if later readings change. Open suggestions are ordered by the number of illustrative thresholds crossed, then profile ID, and displayed 15 at a time. “Mark reviewed” records a demo timestamp and, when configured, the shared demo username; it does not verify a clinician identity or acknowledge a medical event. The ECG-derived high/moderate/low experimental bands are display labels and do not create these wearable review suggestions. No push, emergency, or clinical notifications are sent.

## ECG model evaluation

The saved report is a stratified 80/20 recording-level holdout from the public training recordings, plus five-fold recording-level cross-validation. It reports 73.6% holdout accuracy, 0.620 macro F1, and 74.1% AFib sensitivity. The source data does not provide patient identifiers for a patient-level split, and this repository has no independent clinical validation. These are dataset-only metrics, not clinical performance. The numbers are a snapshot from the current saved report and can change when the data or training settings change; use the model card after the latest training run for current values.

## Run locally (Windows PowerShell)

Prerequisites: Python 3.11 or newer, Docker Desktop, and the data files described in the Where to put the data section below.

### Where to put the data

- Put the PhysioNet `REFERENCE.csv` and its matching `.mat` recording files in `data/ecg/`. The record names in `REFERENCE.csv` must match the `.mat` filenames.
- Put the Synthea `patients.csv` and `conditions.csv` exports in `data/synthea/csv/`.

Feature extraction processes about 8,500 ECG recordings and can take several minutes; it prints progress every 500 records. See [DATASETS.md](DATASETS.md) for sources and attribution. The raw data directory is local and excluded from Git.

1. Create and activate a virtual environment, then install dependencies:

    py -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

2. Start PostgreSQL:

    docker compose up -d

3. Copy .env.example to .env if needed and set DATABASE_URL. The default matches the included Compose service.

4. Prepare the data and populate the demo database in this order. Run load_synthea.py once for a fresh database; it inserts profiles and is not an upsert.

    python .\load_synthea.py
    python .\generate_wearable_stream.py
    python .\extract_ecg_features.py
    python .\train_afib_classifier.py
    python .\fuse_patients.py

   Re-running `generate_wearable_stream.py` replaces the generated readings and clears wearable review suggestions. `fuse_patients.py` clears previous ECG windows, assessments, legacy alert rows, and ECG review history, then rebuilds ECG windows and assessments; it does not rebuild wearable readings or create wearable review suggestions. Opening the overview or review queue evaluates current wearable data and stores any new triggered suggestion.

5. Start the app:

    uvicorn main:app --reload

6. Open http://127.0.0.1:8000/. The landing page links to the patient overview.

Set both STROKETWIN_DEMO_USER and STROKETWIN_DEMO_PASSWORD to enable the optional shared demo login for remote presentation. Use HTTPS outside localhost. This is not a production authentication system.

## Data sources and attribution

See DATASETS.md. The source-code license does not relicense PhysioNet data or third-party assets. Verify source data terms before publishing copies of raw datasets in a public repository.

## Repository submission checklist

- [ ] Add team, institution, and member details above.
- [ ] Confirm the PhysioNet and Synthea data terms, and verify raw data remains excluded from Git unless redistribution is permitted.
- [ ] Add the 20-minute prototype walkthrough video.
- [x] Add the final presentation deck.
- [ ] Review the architecture handout and replace any remaining team-specific details.
- [ ] Create the public GitHub repository and confirm a clean install on another computer.

## Boundaries and next research step

The cross-stream watch is a transparent synthetic demo rule, not a trained predictor of stroke, AFib, or another future event. Moving toward an actual outcome forecast requires a paired longitudinal dataset with time-aligned clinical context, wearable signals, and defined outcome labels, followed by patient-level evaluation and external validation. Do not use this prototype to diagnose, triage, change medication, or make treatment decisions.
