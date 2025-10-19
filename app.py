# EdgeLine — failsafe minimal app (branding + no-crash)
import os
from pathlib import Path
import streamlit as st

# --- must be first Streamlit call ---
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
    """Case-insensitive find across common extensions."""
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = {f.lower(): f for f in list_assets()}
    for base in candidates:
        b = base.lower()
        # try base + extension
        for ext in exts:
            name = b + ext
            if name in files:
                return str(ASSETS / files[name])
        # try exact match (already has extension)
        if b in files:
            return str(ASSETS / files[b])
    return ""

# pick favicon if present (won’t crash if missing)
favicon = find_asset(["favicon", "edgeline_logo_black_gold", "edgeline_logo", "logo"])
if favicon:
    # Page icon can only be set in set_page_config, so just show an image header instead.
    pass

# ---- Header UI ----
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

logo = find_asset([
    "edgeline_logo_black_gold",  # our new file
    "edgeline_logo",             # generic fallback
    "edgeline_logo_dark",        # older name
    "logo_dark", "logo"
])

st.container()
if logo:
    st.image(logo, width=220)
elif favicon:
    st.image(favicon, width=140)
else:
    st.warning("Upload your logo into /assets (e.g., edgeline_logo_black_gold.png or favicon.png).")

st.markdown(
    "<div class='edge-header'><h1>EdgeLine</h1>"
    "<div class='edge-tagline'>PREDICT. PLAY. PROFIT.</div></div>",
    unsafe_allow_html=True,
)

# ---- Diagnostics (so you can see what the app sees) ----
used = logo or favicon or "(none)"
st.caption(f"Logo source used: {used}")
st.caption("assets/ contents: " + (", ".join(list_assets()) or "(empty or missing)"))

# ---- Main area placeholder ----
st.markdown("### Welcome to EdgeLine — your smart betting edge.")
st.write("This is a minimal, stable build. Your odds/models can be added back in once the header is correct.")
