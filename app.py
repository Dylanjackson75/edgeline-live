import time, requests, pandas as pd, streamlit as st

st.set_page_config(page_title="EdgeLine — Live Odds", layout="wide")
st.title("EdgeLine — Live Odds (No Secrets mode)")

# ---- Inline setup (no Secrets required) ----
with # --- Load Odds API key from Secrets first, then env, then sidebar input ---
import os

ODDS_KEY = st.secrets.get("THE_ODDS_API_KEY", "") or os.getenv("THE_ODDS_API_KEY", "")

key_source = None
if ODDS_KEY:
    key_source = "Secrets" if st.secrets.get("THE_ODDS_API_KEY") else "Env"
else:
    # As a fallback, allow manual entry once; cache in session until refresh.
    with st.sidebar:
        st.subheader("Setup")
        _manual = st.text_input(
            "THE_ODDS_API_KEY",
            value="",
            type="password",
            help="Paste your The Odds API key here (temporary fallback).",
            key="odds_key_manual"
        )
        if _manual:
            st.session_state["odds_key_cached"] = _manual
    ODDS_KEY = st.session_state.get("odds_key_cached", "")
    key_source = "Manual" if ODDS_KEY else None

# Optional status line (small, helpful):
if key_source:
    st.caption(f"Using The Odds API key from **{key_source}**.")
else:
    st.warning("No Odds API key detected. Add it in Secrets or paste it in the sidebar.")

# ----------------- Demo fallback -----------
DEMO_ROWS = pd.DataFrame([
    {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline",
     "market":"spreads","team":"ALABAMA","price":-110,"point":-6.5,"home":"ALABAMA","away":"TENNESSEE"},
    {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline",
     "market":"totals","team":"OVER","price":-105,"point":51.5,"home":"ALABAMA","away":"TENNESSEE"},
    {"game_id":"demo2","time":"2025-10-18T23:30:00Z","book_key":"draftkings","book_name":"DraftKings",
     "market":"h2h","team":"GEORGIA","price":-180,"point":None,"home":"GEORGIA","away":"KENTUCKY"},
])

# ================= Tab 1 ===================
with tab1:
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    markets = c1.multiselect("Markets", ["spreads","totals","h2h",
                                         "player_pass_yds","player_rush_yds","player_rec_yds"],
                             default=["spreads","totals","h2h"])
    regions = c2.selectbox("Region", ["us","us2"], index=0)
    fmt     = c3.selectbox("Odds format", ["american","decimal"], index=0)
    auto    = c4.toggle("Auto-refresh 60s", value=False)
    books_filter = st.text_input("Book filter (comma keys e.g. betonlineag,draftkings,fanduel)", value="")

    if auto:
        st.caption("Auto-refreshing every 60s…")
        if "last_tick" not in st.session_state: st.session_state["last_tick"] = 0.0
        now = time.time()
        if now - st.session_state["last_tick"] >= 60:
            st.session_state["last_tick"] = now
            st.experimental_rerun()

    rows = []
    if ODDS_KEY.strip():
        try:
            with st.spinner("Fetching live board…"):
                data = fetch_odds(",".join(markets), regions, fmt, ODDS_KEY.strip())
            for game in data:
                gid = game.get("id")
                t   = game.get("commence_time")
                home = game.get("home_team"); away = game.get("away_team")
                for bk in game.get("bookmakers", []):
                    bk_key = (bk.get("key") or "").strip()
                    if books_filter:
                        allowed = [b.strip() for b in books_filter.split(",") if b.strip()]
                        if bk_key not in allowed: 
                            continue
                    for m in bk.get("markets", []):
                        mkey = m.get("key")
                        for out in m.get("outcomes", []):
                            rows.append({
                                "game_id": gid, "time": t,
                                "book_key": bk_key, "book_name": bk.get("title"),
                                "market": mkey, "team": out.get("name"),
                                "price": out.get("price"), "point": out.get("point"),
                                "home": home, "away": away
                            })
        except Exception as e:
            st.error(f"Fetch failed ({type(e).__name__}): {e}")
            st.info("Showing demo data instead.")
            rows = DEMO_ROWS.to_dict("records")
    else:
        st.info("No API key provided — showing demo data. Paste your key in the sidebar to fetch live odds.")
        rows = DEMO_ROWS.to_dict("records")

    df = pd.DataFrame(rows)
    st.write("### Board preview")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download current board (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "ncaaf_live_odds.csv", "text/csv")

# ================= Tab 2 ===================
with tab2:
    st.write("Upload a BetOnline board (CSV/TSV).")
    f = st.file_uploader("betonline_lines.csv", type=["csv","tsv","txt"])
    if f is not None:
        try:
            df_up = pd.read_csv(f, sep=None, engine="python")
        except Exception:
            f.seek(0); df_up = pd.read_csv(f)
        st.success("File loaded")
        st.dataframe(df_up.head(200), use_container_width=True)
