import os, pickle, sys, pandas as pd
sys.path.insert(0, "src")
from sklearn.metrics import classification_report
from grievance.data.preprocess import load_and_split
from grievance.models.baseline import build_baseline
from grievance.models.evaluate import per_language_report

path = "data/processed/grievances.csv"
if not os.path.exists(path):
    path = "data/samples/grievances_sample.csv"; print("using sample data:", path)
train, val, test = load_and_split(path)
m = build_baseline().fit(train["text"], train["department"])
test = test.assign(pred=m.predict(test["text"]))
print(classification_report(test["department"], test["pred"], zero_division=0))
os.makedirs("reports/baseline", exist_ok=True)
pickle.dump(m, open("reports/baseline/model.pkl", "wb"))
per_language_report(test, "department", "pred").to_csv("reports/per_language_report.csv", index=False)
