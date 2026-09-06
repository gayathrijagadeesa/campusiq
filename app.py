from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_DIR = "model"
required_files = ["risk_model.pkl", "placement_model.pkl", "features.pkl",
                   "importances.pkl", "safe_stats.pkl", "direction.pkl"]

models_ready = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files)

if models_ready:
    risk_model = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
    placement_model = joblib.load(os.path.join(MODEL_DIR, "placement_model.pkl"))
    FEATURES = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    IMPORTANCES = joblib.load(os.path.join(MODEL_DIR, "importances.pkl"))
    SAFE_STATS = joblib.load(os.path.join(MODEL_DIR, "safe_stats.pkl"))
    DIRECTION = joblib.load(os.path.join(MODEL_DIR, "direction.pkl"))
    print("Models loaded successfully!")
else:
    print("WARNING: Models not found. Run 'python generate_dataset.py' then 'python train_model.py' first.")

FEATURE_LABELS = {
    "current_cgpa": "Current CGPA",
    "cgpa_trend": "CGPA trend (vs last semester)",
    "attendance_pct": "Attendance %",
    "backlogs": "Active backlogs",
    "study_hours_week": "Study hours / week",
    "projects_count": "Projects completed",
    "internships_count": "Internships completed",
    "coding_practice_hours": "Coding practice hours / week",
    "communication_score": "Communication skill (self-rated)",
    "extracurricular_score": "Extracurricular involvement",
}

RECOMMENDATIONS = {
    "current_cgpa": "Prioritize the subjects pulling your CGPA down — focus study hours there first.",
    "cgpa_trend": "Your CGPA dipped vs last semester — a short chat with a mentor can help pinpoint what changed.",
    "attendance_pct": "Attendance below 75% risks exam eligibility — treat it as a hard floor, not a target.",
    "backlogs": "Clear one backlog at a time in the next available arrear exam — it's the single biggest lever here.",
    "study_hours_week": "Even 5 extra focused hours a week compounds fast over a semester.",
    "projects_count": "Ship one small project end-to-end (with a GitHub repo) — it matters more than a perfect one.",
    "internships_count": "Even a short internship or research assistantship strengthens your placement story a lot.",
    "coding_practice_hours": "Consistent daily practice (even 30 min) on DSA beats occasional long sessions.",
    "communication_score": "Mock interviews or a campus toastmasters-style club can move this quickly.",
    "extracurricular_score": "Not urgent for academic risk, but worth building for a well-rounded resume.",
}


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


@app.route("/")
def home():
    return render_template("index.html", features=FEATURE_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    if not models_ready:
        return jsonify({"error": "Models not trained yet. Run generate_dataset.py then train_model.py first."}), 500

    data = request.get_json()

    try:
        values = {f: float(data.get(f, 0)) for f in FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input values."}), 400

    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    # ---- Risk prediction ----
    risk_pred = risk_model.predict(X)[0]
    risk_proba = risk_model.predict_proba(X)[0]
    classes = list(risk_model.classes_)
    confidence = round(max(risk_proba) * 100, 1)

    # ---- Placement readiness ----
    placement_score = round(clamp(float(placement_model.predict(X)[0]), 0, 100), 1)

    # ---- Explainability: which factors are pulling away from "Safe" profile ----
    contributions = []
    for f in FEATURES:
        stats = SAFE_STATS[f]
        z = (stats["mean"] - values[f]) / stats["std"]
        if not DIRECTION[f]:
            z = -z  # for "lower is better" features, flip the sign
        concern = max(0, z) * IMPORTANCES[f]
        contributions.append((f, concern))

    contributions.sort(key=lambda x: -x[1])
    top_factors = [f for f, c in contributions[:3] if c > 0.02]

    if not top_factors:
        top_factors = [contributions[0][0]]

    factors_out = [
        {"feature": FEATURE_LABELS[f], "recommendation": RECOMMENDATIONS[f]}
        for f in top_factors
    ]

    return jsonify({
        "risk_level": risk_pred,
        "confidence": confidence,
        "probabilities": {c: round(p * 100, 1) for c, p in zip(classes, risk_proba)},
        "placement_readiness": placement_score,
        "top_factors": factors_out,
    })


if __name__ == "__main__":
    app.run(debug=True)
