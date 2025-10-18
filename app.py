import os, time
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="EdgeLine — Live Odds", layout="wide")
st.title("EdgeLine — Live Odds +EV Workspace")
tab1, tab2, tab3 = st.tabs(["Auto-fetch odds (NCAAF)", "Upload board (CSV)", "Bet tracker"])

with tab1:
    st.subheader("Live Board")
    # ALL of your odds-fetch + table code indented under here

with tab2:
    st.subheader("Upload board (CSV)")
    # uploader code

with tab3:
    st.subheader("Bet tracker")
    # tracker code
