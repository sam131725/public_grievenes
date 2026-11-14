import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, "src")
from grievance.data.preprocess import load_and_split
from grievance.models.evaluate import per_language_report


path = "data/processed/grievances.csv"
if not os.path.exists(path):
    path = "data/samples/grievances_sample.csv"
    print("using sample data:", path)

_, _, test = load_and_split(path)
with open("reports/baseline/model.pkl", "rb") as file:
    model = pickle.load(file)

test = test.assign(pred=model.predict(test["text"]))
print(classification_report(test["department"], test["pred"], zero_division=0))
os.makedirs("reports", exist_ok=True)
per_language_report(test, "department", "pred").to_csv(
    "reports/per_language_report.csv", index=False
)
