"""omni-diarize — Streamlit entry point.

    streamlit run app.py

Routes between the two pages in pages/:
  1. 🚀 Run Pipeline       — ingest a YouTube URL or local file and run run_pipeline.py live
  2. 🏛️ Transcript Viewer  — explore, search, play back and export diarized transcripts
"""

from __future__ import annotations

import streamlit as st

from core.config import MODELS, RUN_PAGE, VIEWER_PAGE
from core.transcript import CSS, meta_row

st.set_page_config(page_title="omni-diarize · Dewan Rakyat", page_icon="🏛️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

pages = [
    st.Page(RUN_PAGE, title="Run Pipeline", icon="🚀", default=True),
    st.Page(VIEWER_PAGE, title="Transcript Viewer", icon="🏛️"),
]
nav = st.navigation(pages)

with st.sidebar:
    with st.expander("⚙️ Model stack", expanded=False):
        st.markdown("".join(meta_row(k, v) for k, v in MODELS.items()), unsafe_allow_html=True)

nav.run()
