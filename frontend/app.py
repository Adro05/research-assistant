"""
Streamlit frontend entrypoint.

This is a project-foundation scaffold. No pages for project management,
paper discovery, document upload, or research chat are implemented yet
(see ARCHITECTURE.md §4). Business logic belongs in the backend, not here.
"""

import streamlit as st

st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("Research Assistant")
st.write("Project scaffold running. Features will be added milestone by milestone.")