# EdgeLine — ultra-simple, always-boot app.py
# Works in demo mode (no keys) or live (Odds API key in Streamlit Secrets)

import os, time
import pandas as pd
import requests
import streamlit as st

# ---------- CONFIG ----------
st.set_page_config(page_title="EdgeLine", layout="wide")
st.title("EdgeLine")
st.caption("Predict. Play. Profit.")

# Secrets (safe if missing)
THE_ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", "").strip()
REGION = "us"
MARKETS_DEFAULT = "spreads,totals,h2h"

# ---------- DATA ----------
DEMO_ROWS = [
    {"sport":"CFB","matchup":"Tennessee @ Alabama","market":"spreads","point":-6.5,"price":-110,"book":"DemoBook"},
    {"sport":"CFB","matchup":"Oregon @ Utah","market":"totals","point":47.5,"price":-105,"book":"DemoBook"},
    {"sport":"CFB","matchup":"Duke @ Miami","market":"spreads","point":+14,"price":-108,"book":"DemoBook"},
]

def fetch_odds_or_demo(sport_key: str, markets: str):
    """Try live Odds API. If any error or missing key, return demo rows."""
    if not THE_ODDS_API_KEY:
        return DEMO_ROWS, "demo (no key)"

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": REGION,
        "markets": markets,
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return DEMO_ROWS, f"demo (odds api {r.status_code})"
        js = r.json()
        rows = []
        for g in js[:50]:
            home, away = g.get("home_team",""), g.get("away_team","")
            for bk in g.get("bookmakers", []):
                book = bk.get("title","")
                for mk in bk.get("markets", []):
                    mkt = mk.get("key","")
                    for out in mk.get("outcomes", []):
                        rows.append({
                            "sport": g.get("sport_title","CFB"),
                            "matchup": f"{away} @ {home}",
                            "market": mkt,
                            "point": out.get("point"),
                            "price": out.get("price"),
                            "book": book
                        })
        return rows, "live"
    except Exception:
        return DEMO_ROWS, "demo (fallback)"

# ---------- UI ----------
left, mid = st.columns([1,3])

with left:
    league = st.selectbox(
        "League",
        ["CFB (college football)", "NBA", "NFL", "NHL"],
        index=0
    )
    sport_key = {
        "CFB (college football)": "americanfootball_ncaaf",
        "NFL": "americanfootball_nfl",
        "NBA": "basketball_nba",
        "NHL": "icehockey_nhl",
    }[league]
    mkts = st.multiselect("Markets", ["spreads","totals","h2h"], default=["spreads","totals"])
    markets = ",".join(mkts) if mkts else MARKETS_DEFAULT
    auto = st.toggle("Auto-refresh every 60s", value=False)

with mid:
    st.subheader("Top Value Bets (sample or live)")
    rows, src = fetch_odds_or_demo(sport_key, markets)
    st.caption(f"Source: {src}")
    df = pd.DataFrame(rows)
    if not df.empty:
        # naive highlight: show “best” rows by cheapest (more positive for plus money)
        df["edge_hint"] = df["price"].apply(lambda p: "plus $" if (p and p>0) else "")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No rows yet.")

# optional simple bet tracker (local session only)
st.subheader("Bet Tracker (local session)")
if "bets" not in st.session_state:
    st.session_state.bets = []
c1, c2, c3, c4, c5 = st.columns(5)
with c1: m_match = st.text_input("Matchup", placeholder="Team A @ Team B")
with c2: m_mkt = st.selectbox("Market", ["spread","total","h2h"], index=0)
with c3: m_point = st.text_input("Point", value="")
with c4: m_price = st.text_input("Price (US odds)", value="-110")
with c5:
    if st.button("Add bet"):
        st.session_state.bets.append({"matchup":m_match,"market":m_mkt,"point":m_point,"price":m_price})
if st.session_state.bets:
    st.table(pd.DataFrame(st.session_state.bets))

# footer
st.markdown("<hr style='opacity:.2'/>", unsafe_allow_html=True)
st.caption("EdgeLine — fast demo build. Add your Odds API key in Streamlit Secrets to enable live lines.")
