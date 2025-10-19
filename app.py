# EdgeLine — packaged Streamlit app (failsafe)
import os
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="EdgeLine — Predict. Play. Profit.",
    page_icon=None,
    layout="wide",
)

ASSETS = Path("assets")

def list_assets():
    try:
        return sorted(os.listdir(ASSETS)) if ASSETS.exists() else []
    except Exception:
        return []

def find_asset(candidates):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = {f.lower(): f for f in list_assets()}
    for base in candidates:
        b = base.lower()
        for ext in exts:
            name = b + ext
            if name in files:
                return str(ASSETS / files[name])
        if b in files:
            return str(ASSETS / files[b])
    return ""

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
    """,
    unsafe_allow_html=True,
)

logo = find_asset(["edgeline_logo_gradient", "edgeline_logo_black_gold", "edgeline_logo", "logo"])
if logo:
    st.image(logo, width=220)
st.markdown(
    """<div class='edge-header'><h1>EdgeLine</h1>
    <div class='edge-tagline'>PREDICT. PLAY. PROFIT.</div></div>""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Value Bets", "⚙️ Settings"])

with tab1:
    st.subheader("Welcome")
    st.write("Your EdgeLine app is packaged and ready. Upload your logo into /assets to see it here.")

with tab2:
    st.write("Plug your odds/model modules here when ready.")

with tab3:
    st.write("Add secrets in Streamlit Cloud → Settings → Secrets.")
    st.code("THE_ODDS_API_KEY = \"...\"\nOPENAI_API_KEY = \"...\"", language="toml")

st.caption("assets/: " + (", ".join(list_assets()) or "(empty)"))
