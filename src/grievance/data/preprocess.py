import re
import pandas as pd
from sklearn.model_selection import train_test_split

_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_WS = re.compile(r"\s+")

def clean_text(t: str) -> str:
    """Remove web links and normalize whitespace in a complaint."""
    t = _URL.sub(" ", str(t))
    return _WS.sub(" ", t).strip()

def load_and_split(path, seed=42, stratify_col="department"):
    """CSV columns: text, language, department, urgency, sentiment."""
    df = pd.read_csv(path)
    if stratify_col not in df.columns:
        raise ValueError(f"Missing stratification column: {stratify_col}")
    df = df.dropna(subset=["text", stratify_col])
    df["text"] = df["text"].map(clean_text)
    df = df.loc[df["text"].ne("")].copy()
    if df.empty:
        raise ValueError("No complaints with nonempty text and labels remain after cleaning.")
    train, tmp = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[stratify_col])
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp[stratify_col])
    return train, val, test
