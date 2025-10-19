# ───────────── EdgeLine — clean, working base (full file) ─────────────
import os, time, math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Paths / Assets
ASSETS = Path("assets")

# ── Page meta (uses your gold E favicon if present)
st.set_page_config(
    page_title="EdgeLine — Predict. Play. Profit.",
    page_icon=str(ASSETS / "favicon.png") if (ASSETS / "favicon.png").exists() else None,
    layout="wide",
)

# ── Branding CSS + Header
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
          .edge-header h1 { margin:0; padding:0; line-height:1.1 }
        </style>
        """,
        unsafe_allow_html=True,
    )

def edge_header():
    _edgeline_css()
    dark = ASSETS / "edgeline_logo_dark.png"
    light = ASSETS / "edgeline_logo_light.png"
    if dark.exists():
        st.image(str(dark), width=200)
    elif light.exists():
        st.image(str(light), width=200)
    st.markdown(
        "<div class='edge-header'><h1>EdgeLine</h1>"
        "<div class='edge-tagline'>PREDICT. PLAY. PROFIT.</div></div>",
        unsafe_allow_html=True,
    )

# ── Secrets helper (won’t crash if missing)
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

THE_ODDS_API_KEY = get_secret("THE_ODDS_API_KEY")
OPENAI_API_KEY   = get_secret("OPENAI_API_KEY")
API_BASE         = get_secret("API_BASE", "https://edgeline-api.onrender.com")

# ── Safe odds fetch (skips if key missing/unauthorized)
def fetch_example_odds():
    if not THE_ODDS_API_KEY:
        return pd.DataFrame(), "No THE_ODDS_API_KEY set — skipping live fetch."
    url = (
        "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
        "?regions=us&markets=spreads,totals&oddsFormat=american"
        f"&apiKey={THE_ODDS_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 401:
            return pd.DataFrame(), "401 Unauthorized from Odds API — check/upgrade your key."
        r.raise_for_status()
        data = r.json()
        # Flatten a tiny sample for display
        rows = []
        for g in data[:20]:
            home = g.get("home_team"); away = g.get("away_team")
            commence = g.get("commence_time")
            bookmakers = g.get("bookmakers", [])
            if not bookmakers:
                continue
            bm = bookmakers[0]
            mkts = {m["key"]: m for m in bm.get("markets", [])}
            def pick_line(mkt_key, out_key):
                m = mkts.get(mkt_key, {})
                for o in m.get("outcomes", []):
                    if out_key.lower() in o.get("name","").lower():
                        return o.get("point"), o.get("price")
                return None, None
            s_home, s_home_price = pick_line("spreads", home or "")
            t_over, t_over_price = pick_line("totals", "Over")
            rows.append({
                "home": home, "away": away, "time": commence,
                "spread_home": s_home, "spread_home_price": s_home_price,
                "total_over": t_over, "total_over_price": t_over_price,
                "book": bm.get("title")
            })
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Fetch error: {e}"

# ── APP UI
edge_header()

tab_dash, tab_bets, tab_settings = st.tabs(["📊 Dashboard", "🎯 Value Bets", "⚙️ Settings"])

with tab_dash:
    st.subheader("Status")
    msgs = []
    if not THE_ODDS_API_KEY:
        msgs.append("Odds: **No THE_ODDS_API_KEY** found (app will run without live odds).")
    if not OPENAI_API_KEY:
        msgs.append("AI: **No OPENAI_API_KEY** found (write-ups/chat disabled).")
    if msgs:
        for m in msgs:
            st.warning(m)
    else:
        st.success("All keys detected.")

    st.subheader("Live Odds (sample)")
    df, err = fetch_example_odds()
    if err:
        st.info(err)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No rows to show yet.")

with tab_bets:
    st.subheader("Your Board")
    st.write("Upload a board CSV or connect your feed when ready. (This tab will populate with +EV once odds are flowing.)")

with tab_settings:
    st.subheader("Environment")
    col1, col2, col3 = st.columns(3)
    col1.code(f"THE_ODDS_API_KEY set: {bool(THE_ODDS_API_KEY)}")
    col2.code(f"OPENAI_API_KEY set: {bool(OPENAI_API_KEY)}")
    col3.code(f"API_BASE: {API_BASE}")
    st.caption("Edit in Streamlit → ⋮ → Settings → Secrets")
# ──────────────────────────────────────────────────────────────────────
