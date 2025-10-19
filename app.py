# ───────────────────── EdgeLine — working Streamlit app.py ─────────────────────
import os, math, time
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Branding helpers ----------
ASSETS = Path("assets")

def _safe_list_assets():
    try:
        return sorted(os.listdir(ASSETS)) if ASSETS.exists() else []
    except Exception:
        return []

def _find_asset(candidates):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = {f.lower(): f for f in _safe_list_assets()}
    for base in candidates:
        base_low = base.lower()
        for ext in exts:
            name = base_low + ext
            if name in files:
                return str(ASSETS / files[name])
        if base_low in files:
            return str(ASSETS / files[base_low])
    return ""

def _pick_favicon():
    return _find_asset(["favicon", "edgeline_logo_black_gold", "edgeline_logo", "logo"])

st.set_page_config(
    page_title="EdgeLine — Predict. Play. Profit.",
    page_icon=_pick_favicon(),
    layout="wide",
)

# ---------- Header styling ----------
def _edgeline_css():
    st.markdown(
        """
        <style>
          :root { --edge-gold:#D4AF37; --edge-green:#7AF57A; --edge-ink:#F2F2F2; --edge-bg:#0B0B0B; }
          .stApp {
            background:
              radial-gradient(1200px 600px at 20% -10%, rgba(212,175,55,.12), transparent 60%),
              radial-gradient(1200px 600px at 90% 10%, rgba(122,245,122,.06), transparent 60%),
              #0B0B0B;
          }
          .edge-tagline {
            display:inline-block; margin:.25rem 0 1rem 0; padding:6px 10px;
            border:1px solid rgba(122,245,122,.25); color:#7AF57A;
            border-radius:999px; font-weight:600; font-size:.9rem; letter-spacing:.04em;
            background:rgba(122,245,122,.07);
          }
          .edge-header h1 { margin:0; padding:0; line-height:1.1; color:#FFD700; }
        </style>
        """, unsafe_allow_html=True
    )

def edge_header():
    _edgeline_css()
    logo = _find_asset(["edgeline_logo_black_gold", "edgeline_logo", "logo"])
    emblem = _find_asset(["favicon", "icon"])

    if logo:
        st.image(logo, width=220)
    elif emblem:
        st.image(emblem, width=140)
    else:
        st.error("No logo found in /assets. Upload edgeline_logo_black_gold.png or favicon.png")

    st.markdown(
        "<div class='edge-header'><h1>EdgeLine</h1>"
        "<div class='edge-tagline'>PREDICT. PLAY. PROFIT.</div></div>",
        unsafe_allow_html=True,
    )

    used = logo or emblem or "(none)"
    listing = ", ".join(sorted(os.listdir(ASSETS))) if ASSETS.exists() else "(assets folder missing)"
    st.caption(f"Logo source: {used}  •  assets/: {listing}")

edge_header()

# ---------- Placeholder main area ----------
st.markdown("### Welcome to EdgeLine — your smart betting edge.")
st.write("Odds data, props, and analysis modules load here.")
