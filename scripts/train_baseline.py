
import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, "src")

from grievance.data.preprocess import load_and_split
from grievance.models.baseline import build_baseline
from grievance.models.evaluate import per_language_report


# Dataset configuration
DATA_PATH = "data/processed/grievances.csv"
SAMPLE_PATH = "data/samples/grievances_sample.csv"

if not os.path.exists(DATA_PATH):
    DATA_PATH = SAMPLE_PATH
    print("Using sample data:", DATA_PATH)


# Load and split the dataset
train, val, test = load_and_split(DATA_PATH)


# Train the baseline model
model = build_baseline()
model.fit(train["text"], train["department"])


# Generate predictions
test = test.assign(
    pred=model.predict(test["text"])
)


# Evaluate model performance
print(
    classification_report(
        test["department"],
        test["pred"],
        zero_division=0
    )
)


# Save the trained model
os.makedirs("reports/baseline", exist_ok=True)

with open("reports/baseline/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)


# Generate per-language evaluation report
language_report = per_language_report(
    test,
    "department",
    "pred"
)

language_report.to_csv(
    "reports/per_language_report.csv",
    index=False
)
