import io
import pandas as pd
import streamlit as st
from grievance.models.predictor import Predictor

st.set_page_config(page_title="Grievance Router", layout="wide")
st.title("Multilingual Public Grievance Classifier & Router")

@st.cache_resource
def get_predictor():
    return Predictor()

p = get_predictor()
tab1, tab2, tab3 = st.tabs(["Paste complaint", "Upload CSV", "Per-language report"])

with tab1:
    text = st.text_area("Complaint text (any supported language)", height=150)
    if st.button("Classify") and text.strip():
        st.json(p.predict(text))

with tab2:
    f = st.file_uploader("CSV with a 'text' column", type="csv")
    if f:
        df = pd.read_csv(f)
        out = pd.concat([df, pd.DataFrame([p.predict(t) for t in df["text"]])], axis=1)
        st.dataframe(out)
        st.download_button("Download results", out.to_csv(index=False), "routed.csv")

with tab3:
    try:
        st.dataframe(pd.read_csv("reports/per_language_report.csv"))
    except FileNotFoundError:
        st.info("Run scripts/evaluate.py to generate reports/per_language_report.csv")
