"""
CampusIQ - Model Training
--------------------------
Trains two models:
1. RandomForestClassifier -> Risk level (Safe / Warning / Critical)
2. RandomForestRegressor  -> Placement readiness score (0-100)

Both use the same feature set. Feature importances from the classifier
power the "why this result" explanation on the website.

Run: python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

FEATURES = [
    "current_cgpa", "cgpa_trend", "attendance_pct", "backlogs",
    "study_hours_week", "projects_count", "internships_count",
    "coding_practice_hours", "communication_score", "extracurricular_score"
]

print("Loading data/students.csv ...")
df = pd.read_csv("data/students.csv")

X = df[FEATURES]
y_risk = df["risk_level"]
y_placement = df["placement_readiness"]

X_train, X_test, y_risk_train, y_risk_test = train_test_split(
    X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
)
_, _, y_place_train, y_place_test = train_test_split(
    X, y_placement, test_size=0.2, random_state=42
)

# ---------- Risk Classifier ----------
print("Training Risk Classifier (Random Forest)...")
risk_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
risk_model.fit(X_train, y_risk_train)

risk_pred = risk_model.predict(X_test)
print(f"\nRisk Model Accuracy: {accuracy_score(y_risk_test, risk_pred)*100:.2f}%")
print(classification_report(y_risk_test, risk_pred))

# ---------- Placement Readiness Regressor ----------
print("Training Placement Readiness Regressor...")
placement_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
placement_model.fit(X_train, y_place_train)

place_pred = placement_model.predict(X_test)
mae = mean_absolute_error(y_place_test, place_pred)
print(f"\nPlacement Readiness MAE: {mae:.2f} points (out of 100)")

# ---------- Feature importance (for explainability on the website) ----------
importances = dict(zip(FEATURES, risk_model.feature_importances_))
print("\nFeature importances (risk model):")
for f, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.3f}")

# ---------- Reference stats for explainability ----------
# For each feature, store the mean/std among "Safe" students. When a new
# student comes in, we compare their values against this "safe profile" to
# figure out which factors are pulling them away from Safe (weighted by
# how important that feature is to the model overall).
safe_df = df[df["risk_level"] == "Safe"]
safe_stats = {
    f: {"mean": float(safe_df[f].mean()), "std": float(safe_df[f].std() + 1e-6)}
    for f in FEATURES
}

# Direction: True = higher is better, False = lower is better
DIRECTION = {
    "current_cgpa": True, "cgpa_trend": True, "attendance_pct": True,
    "backlogs": False, "study_hours_week": True, "projects_count": True,
    "internships_count": True, "coding_practice_hours": True,
    "communication_score": True, "extracurricular_score": True,
}

# ---------- Save everything ----------
joblib.dump(risk_model, "model/risk_model.pkl")
joblib.dump(placement_model, "model/placement_model.pkl")
joblib.dump(FEATURES, "model/features.pkl")
joblib.dump(importances, "model/importances.pkl")
joblib.dump(safe_stats, "model/safe_stats.pkl")
joblib.dump(DIRECTION, "model/direction.pkl")

print("\nSaved models to model/. Now run: python app.py")
