"""
CampusIQ - Dataset Generator
-----------------------------
Generates a realistic synthetic dataset of college student records since
there's no single public "college risk + placement" dataset like fake-news
has. This mirrors how real ML teams start a project when clean labeled data
doesn't exist yet: define the features, encode domain rules with noise, and
generate.

Run: python generate_dataset.py
Creates: data/students.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

# ---------- Features ----------
current_cgpa = np.clip(np.random.normal(7.0, 1.3, N), 4.0, 10.0)
previous_cgpa = np.clip(current_cgpa + np.random.normal(0, 0.6, N), 4.0, 10.0)
cgpa_trend = current_cgpa - previous_cgpa  # negative = declining

attendance_pct = np.clip(np.random.normal(78, 14, N), 30, 100)
backlogs = np.clip(np.random.poisson(1.2, N) - (current_cgpa - 6).astype(int), 0, 8)
study_hours_week = np.clip(np.random.normal(12, 6, N), 0, 40)

projects_count = np.clip(np.random.poisson(1.8, N), 0, 8)
internships_count = np.clip(np.random.poisson(0.5, N), 0, 4)
coding_practice_hours = np.clip(np.random.normal(5, 4, N), 0, 25)
communication_score = np.clip(np.random.normal(6.2, 1.8, N), 1, 10)
extracurricular_score = np.clip(np.random.normal(5, 2.3, N), 0, 10)

# ---------- Risk label (rule + noise, mirrors real academic risk factors) ----------
risk_score = (
    (10 - current_cgpa) * 1.4
    + backlogs * 1.8
    + (80 - attendance_pct) * 0.06
    + np.maximum(0, -cgpa_trend) * 2.0
    + (15 - study_hours_week) * 0.08
    + np.random.normal(0, 1.2, N)
)

risk_level = pd.cut(
    risk_score,
    bins=[-np.inf, 6, 12, np.inf],
    labels=["Safe", "Warning", "Critical"]
)

# ---------- Placement readiness score (0-100) ----------
placement_readiness = (
    current_cgpa * 6.5
    + projects_count * 4.5
    + internships_count * 8.0
    + coding_practice_hours * 1.6
    + communication_score * 3.0
    - backlogs * 5.0
    + np.random.normal(0, 5, N)
)
placement_readiness = np.clip(placement_readiness, 0, 100)

df = pd.DataFrame({
    "current_cgpa": np.round(current_cgpa, 2),
    "cgpa_trend": np.round(cgpa_trend, 2),
    "attendance_pct": np.round(attendance_pct, 1),
    "backlogs": backlogs.astype(int),
    "study_hours_week": np.round(study_hours_week, 1),
    "projects_count": projects_count.astype(int),
    "internships_count": internships_count.astype(int),
    "coding_practice_hours": np.round(coding_practice_hours, 1),
    "communication_score": np.round(communication_score, 1),
    "extracurricular_score": np.round(extracurricular_score, 1),
    "risk_level": risk_level.astype(str),
    "placement_readiness": np.round(placement_readiness, 1),
})

df.to_csv("data/students.csv", index=False)
print(f"Generated {len(df)} student records -> data/students.csv")
print(df["risk_level"].value_counts())
print(df.head())
