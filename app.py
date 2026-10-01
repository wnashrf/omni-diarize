"""omni-diarize — Streamlit entry point.

    streamlit run app.py

Routes between the two pages in pages/ via a top navigation bar:
  1. Run Pipeline       — ingest a YouTube URL or local file and run run_pipeline.py live
  2. Transcript Viewer  — explore, search, play back and export diarized transcripts
"""

from __future__ import annotations

import streamlit as st

from core.config import RUN_PAGE, VIEWER_PAGE
from core.transcript import CSS

st.set_page_config(page_title="omni-diarize · Dewan Rakyat", page_icon="🏛️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

pages = [
    st.Page(RUN_PAGE, title="Run Pipeline", icon=":material/play_circle:", default=True),
    st.Page(VIEWER_PAGE, title="Transcript Viewer", icon=":material/article:"),
]
st.navigation(pages, position="top").run()
