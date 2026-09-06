# CampusIQ — Academic Risk & Placement Readiness Predictor

An AI web app for **final-year college students** that:
1. Predicts **academic risk level** (Safe / Warning / Critical) from CGPA, attendance, backlogs, and study habits
2. Explains **why** — the top 3 factors actually driving that result (not a black box)
3. Scores **placement readiness** (0–100) from CGPA, projects, internships, coding practice, and communication
4. Gives **specific recommendations**, not just a number

No school-dropout dataset exists that fits this exact college use case, so the
project includes a script that generates a realistic synthetic dataset with
the same statistical relationships real academic data would have (this is a
completely normal, industry-standard practice when a project needs a dataset
that doesn't already exist).

## Project structure
```
campusiq/
├── generate_dataset.py   # Creates data/students.csv (1500 synthetic students)
├── train_model.py        # Trains risk classifier + placement regressor
├── app.py                # Flask backend + /predict API
├── requirements.txt
├── model/                # risk_model.pkl, placement_model.pkl etc. (created after training)
├── data/                 # students.csv (created after generating)
├── templates/
│   └── index.html        # Dashboard UI (sliders + gauge + risk badge)
└── static/
    ├── style.css          # Control-room dashboard theme
    └── script.js          # Calls /predict, animates gauge and risk badge
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the dataset:
   ```bash
   python generate_dataset.py
   ```
   Creates `data/students.csv` (1500 rows) and prints the risk-level split.

3. Train both models:
   ```bash
   python train_model.py
   ```
   Prints accuracy for the risk classifier and MAE for the placement
   readiness regressor, then saves everything to `model/`.

4. Run the website:
   ```bash
   python app.py
   ```
   Open **http://127.0.0.1:5000**. Move the sliders to describe a student's
   profile and click **Analyze my standing**.

## Pushing to GitHub

```bash
git init
git add .
git commit -m "CampusIQ - academic risk and placement readiness predictor"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

`data/*.csv` and `model/*.pkl` are gitignored since they're generated files —
anyone cloning the repo just runs the two setup scripts above once.

## How it works (for your viva / resume)

**Two models, one feature set:**
- `RandomForestClassifier` → 3-class risk level (Safe / Warning / Critical)
- `RandomForestRegressor` → continuous placement readiness score (0–100)

**Explainability without SHAP:**
For each incoming student, every feature is compared against the *average
profile of "Safe" students* in the training data (z-score), then weighted by
that feature's overall importance in the risk model. The top 3 features
pulling a student away from a safe profile become the "why" shown on the
dashboard — this mirrors how real explainable-AI systems (credit scoring,
churn prediction) work, just without pulling in a heavy extra library.

**Why this is a strong resume project for an AI&DS grad:**
- End-to-end pipeline: synthetic data generation → feature engineering →
  two trained models → explainability → a served product
- Same pattern (risk score + explainability + recommendation) used in real
  fintech/HR-tech systems — interviewers recognize this immediately
- You can speak to *trade-offs*: why Random Forest over logistic regression
  (handles non-linear feature interactions like "low CGPA + backlogs
  compounding"), why a rule-based synthetic dataset was necessary, and the
  precision/recall trade-off on the minority "Critical" class

## Talking points if asked "how would you improve this"
- Replace the z-score explainability with SHAP values for more rigorous,
  per-prediction attribution
- Collect a real (anonymized) dataset from your own department for the final
  report, keep the synthetic generator as the reproducible baseline
- Add a batch CSV upload for a mentor/HOD to screen a whole class at once
- Track a student's risk score over multiple semesters as a time series
