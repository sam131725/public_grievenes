"""Fine-tune MuRIL / IndicBERT / mBERT with 3 heads. Run on Colab/Kaggle GPU."""
import argparse, os, sys, yaml, torch, numpy as np
sys.path.insert(0, "src")
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from grievance.data.preprocess import load_and_split
from grievance.labels import DEPARTMENTS, URGENCY, SENTIMENT
from grievance.models.multitask import MultiTaskClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
cfg = yaml.safe_load(open(ap.parse_args().config))
torch.manual_seed(cfg["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(cfg["model_name"])
train, val, test = load_and_split(cfg["data_path"])

class DS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return r.text, DEPARTMENTS.index(r.department), URGENCY.index(r.urgency), SENTIMENT.index(r.sentiment)

def collate(b):
    t, d, u, s = zip(*b)
    enc = tok(list(t), padding=True, truncation=True, max_length=cfg["max_len"], return_tensors="pt")
    return enc, torch.tensor(d), torch.tensor(u), torch.tensor(s)

tl = DataLoader(DS(train), cfg["batch_size"], shuffle=True, collate_fn=collate)
vl = DataLoader(DS(val), cfg["batch_size"], collate_fn=collate)
model = MultiTaskClassifier(cfg["model_name"]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
sch = get_linear_schedule_with_warmup(opt, 0, len(tl) * cfg["epochs"])

def w(col, labels):
    if not cfg["class_weighting"]: return None
    cw = compute_class_weight("balanced", classes=np.arange(len(labels)), y=[labels.index(x) for x in train[col]])
    return torch.tensor(cw, dtype=torch.float, device=dev)
ce = {k: torch.nn.CrossEntropyLoss(weight=w(k, l)) for k, l in
      [("department", DEPARTMENTS), ("urgency", URGENCY), ("sentiment", SENTIMENT)]}

for ep in range(cfg["epochs"]):
    model.train()
    for enc, d, u, s in tl:
        out = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        loss = ce["department"](out["department"], d.to(dev)) + ce["urgency"](out["urgency"], u.to(dev)) + ce["sentiment"](out["sentiment"], s.to(dev))
        loss.backward(); opt.step(); sch.step(); opt.zero_grad()
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for enc, d, u, s in vl:
            o = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            P += o["department"].argmax(1).cpu().tolist(); Y += d.tolist()
    print(f"epoch {ep+1} val dept macro-F1 = {f1_score(Y, P, average='macro'):.4f}")

os.makedirs(cfg["output_dir"], exist_ok=True)
torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "model.pt"))
