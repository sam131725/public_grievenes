"""Streamlit dashboard: paste/upload complaints -> department, urgency, sentiment, route.

Run:  streamlit run src/grievance/dashboard/app.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from grievance.models.predictor import load_predictor  # noqa: E402

REPORT_DIR = Path(os.environ.get("GRIEVANCE_REPORT_DIR", ROOT / "reports"))
MAX_ROWS = 5000

st.set_page_config(page_title="Grievance Router", layout="wide")
st.title("Multilingual Public Grievance Classifier & Router")


@st.cache_resource
def get_predictor():
    return load_predictor()


p = get_predictor()
st.sidebar.subheader("Model")
st.sidebar.json(p.info())

tab1, tab2, tab3 = st.tabs(["Paste complaint", "Upload CSV", "Per-language report"])

with tab1:
    text = st.text_area("Complaint text (any supported language)", height=150)
    if st.button("Classify"):
        if not text.strip():
            st.warning("Enter a complaint first.")
        else:
            try:
                res = p.predict(text)
                c1, c2, c3 = st.columns(3)
                c1.metric("Department", res["department"])
                c2.metric("Urgency", res["urgency"])
                c3.metric("Sentiment", res["sentiment"])
                st.success(f"Route to: {res['route']}")
                st.json(res)
            except (RuntimeError, ValueError) as e:
                st.error(str(e))

with tab2:
    f = st.file_uploader("CSV with a 'text' column", type="csv")
    if f:
        df = pd.read_csv(f)
        if "text" not in df.columns:
            st.error(f"CSV needs a 'text' column; found {list(df.columns)}")
        else:
            df = df.dropna(subset=["text"]).head(MAX_ROWS)
            df = df[df["text"].astype(str).str.strip() != ""]
            try:
                preds = pd.DataFrame(p.predict_batch(df["text"].astype(str).tolist()), index=df.index)
                out = pd.concat([df, preds], axis=1)
                st.dataframe(out, use_container_width=True)
                st.bar_chart(out["department"].value_counts())
                st.download_button("Download results", out.to_csv(index=False), "routed.csv")
            except (RuntimeError, ValueError) as e:
                st.error(str(e))

with tab3:
    rep = REPORT_DIR / "per_language_report.csv"
    if rep.exists():
        st.subheader("Per-language accuracy (department)")
        st.dataframe(pd.read_csv(rep), use_container_width=True)
        cm = REPORT_DIR / "confusion_matrix.csv"
        if cm.exists():
            st.subheader("Confusion matrix (rows = actual)")
            st.dataframe(pd.read_csv(cm, index_col=0), use_container_width=True)
    else:
        st.info(f"Run `python scripts/evaluate.py` to generate {rep}")
