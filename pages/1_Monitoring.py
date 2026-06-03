import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

st.title("Model Monitoring")
st.markdown(
    "Compares model performance and feature distributions between the **training set** (reference) "
    "and the **test set** (current). Run the report to detect data drift or regression quality issues."
)

if st.button("Generate Report", type="primary", use_container_width=True):
    with st.spinner("Running Evidently — this takes ~30 seconds..."):
        from src.monitoring.monitor import generate_report
        path = generate_report()

    st.success("Report ready.")
    html = path.read_text(encoding="utf-8")
    components.html(html, height=900, scrolling=True)
