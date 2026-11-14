"""Fine-tune MuRIL / IndicBERT / mBERT with three heads (department, urgency, sentiment).

Run on a Colab/Kaggle GPU:
  python scripts/train_transformer.py --config configs/muril.yaml
Smoke run on the demo data:
  python scripts/train_transformer.py --config configs/mbert.yaml --data data/samples/grievances_sample.csv --epochs 1

Writes to output_dir: model.pt, config.json, tokenizer + encoder config, metrics.json,
per_language_report.csv, confusion_matrix.csv.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grievance.data.preprocess import load_and_split
from grievance.labels import DEPARTMENTS, SENTIMENT, URGENCY
from grievance.models.evaluate import department_confusion_matrix, per_language_report
from grievance.models.imbalance import balanced_class_weights
from grievance.models.multitask import MultiTaskClassifier

TASKS = {"department": DEPARTMENTS, "urgency": URGENCY, "sentiment": SENTIMENT}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", required=True)
ap.add_argument("--data", help="override data_path from the config")
ap.add_argument("--epochs", type=int, help="override epochs from the config")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
if args.data: cfg["data_path"] = args.data
if args.epochs: cfg["epochs"] = args.epochs

torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(cfg["model_name"])
train, val, test = load_and_split(cfg["data_path"], seed=cfg["seed"])
print(f"device={dev} model={cfg['model_name']} train/val/test={len(train)}/{len(val)}/{len(test)}")


class DS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return r.text, *[labels.index(r[t]) for t, labels in TASKS.items()]


def collate(batch):
    texts, *ys = zip(*batch)
    enc = tok(list(texts), padding=True, truncation=True, max_length=cfg["max_len"], return_tensors="pt")
    return enc, [torch.tensor(y) for y in ys]


def loader(df, shuffle=False):
    return DataLoader(DS(df), cfg["batch_size"], shuffle=shuffle, collate_fn=collate)


def predict(model, dl):
    model.eval()
    preds = {t: [] for t in TASKS}; gold = {t: [] for t in TASKS}
    with torch.no_grad():
        for enc, ys in dl:
            out = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            for (t, _), y in zip(TASKS.items(), ys):
                preds[t] += out[t].argmax(1).cpu().tolist(); gold[t] += y.tolist()
    return preds, gold


def scores(preds, gold):
    return {t: {"accuracy": round(accuracy_score(gold[t], preds[t]), 4),
                "macro_f1": round(f1_score(gold[t], preds[t], average="macro", zero_division=0), 4)}
            for t in TASKS}


model = MultiTaskClassifier(cfg["model_name"]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
train_dl, val_dl, test_dl = loader(train, True), loader(val), loader(test)
sch = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_dl) * cfg["epochs"]), len(train_dl) * cfg["epochs"])

losses = {}
for t, labels in TASKS.items():
    w = None
    if cfg.get("class_weighting", True):
        w = torch.tensor(balanced_class_weights([labels.index(x) for x in train[t]], len(labels)),
                         dtype=torch.float, device=dev)
    losses[t] = torch.nn.CrossEntropyLoss(weight=w)

best_f1, best_state, history = -1.0, None, []
for ep in range(cfg["epochs"]):
    model.train()
    for enc, ys in train_dl:
        out = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        loss = sum(losses[t](out[t], y.to(dev)) for (t, _), y in zip(TASKS.items(), ys))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
    p, g = predict(model, val_dl); s = scores(p, g)
    history.append({"epoch": ep + 1, **{f"val_{t}_f1": s[t]["macro_f1"] for t in TASKS}})
    print(f"epoch {ep+1}: " + "  ".join(f"{t} F1={s[t]['macro_f1']:.3f}" for t in TASKS))
    if s["department"]["macro_f1"] > best_f1:
        best_f1 = s["department"]["macro_f1"]
        best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

model.load_state_dict(best_state)
p, g = predict(model, test_dl)
test_scores = scores(p, g)
print("TEST:", json.dumps(test_scores))

out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
torch.save(best_state, out / "model.pt")
tok.save_pretrained(out)
model.encoder.config.save_pretrained(out)  # lets inference rebuild the encoder offline
# save_pretrained wrote the encoder config to config.json; add our own keys to it
enc_cfg = json.loads((out / "config.json").read_text())
enc_cfg.update({"model_name": cfg["model_name"], "max_len": cfg["max_len"], "seed": cfg["seed"]})
(out / "config.json").write_text(json.dumps(enc_cfg, indent=2))

test_df = test.reset_index(drop=True).assign(pred=[DEPARTMENTS[i] for i in p["department"]])
per_language_report(test_df, "department", "pred").to_csv(out / "per_language_report.csv", index=False)
department_confusion_matrix(test_df, "department", "pred").to_csv(out / "confusion_matrix.csv")
(out / "metrics.json").write_text(json.dumps({
    "model": cfg["model_name"], "split_sizes": [len(train), len(val), len(test)],
    "class_weighting": cfg.get("class_weighting", True), "best_val_department_f1": round(best_f1, 4),
    "test": test_scores, "history": history}, indent=2))
print("saved to", out)
