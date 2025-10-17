import os, time, asyncio, httpx, pandas as pd, streamlit as st, requests

st.set_page_config(page_title="EdgeLine — Live Odds", layout="wide")
st.title("EdgeLine — Live Odds +EV Workspace")

# ---- Secrets ----
API_BASE = st.secrets.get("API_BASE","")                 # optional backend (Render)
API_KEY  = st.secrets.get("API_KEY","")                  # optional backend key
ODDS_KEY = st.secrets.get("THE_ODDS_API_KEY","")         # The Odds API key

# ---- Optional backend health check ----
status = st.empty()
if API_BASE:
    try:
        r = requests.get(f"{API_BASE}/health",
                         headers={"X-API-Key": API_KEY} if API_KEY else {},
                         timeout=6)
        if r.ok and r.json().get("ok"):
            status.success(f"✅ Connected to API: {API_BASE}")
        else:
            status.warning(f"API reachable but not healthy: {API_BASE}")
    except Exception as e:
        status.info(f"(Optional) Backend not reachable: {e}")
else:
    status.info("Backend optional. Live odds work with The Odds API only.")

tab1, tab2 = st.tabs(["Auto-fetch odds (NCAAF)", "Upload board (CSV)"])

# ===========================
# Tab 1: Auto-fetch (The Odds API)
# ===========================
with tab1:
    st.subheader("NCAAF Odds — US books (The Odds API)")
    if not ODDS_KEY:
        st.error("Add THE_ODDS_API_KEY in Secrets to enable auto-fetch.")
        st.stop()

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    markets = c1.multiselect(
        "Markets",
        ["spreads","totals","h2h","player_pass_yds","player_rush_yds","player_rec_yds"],
        default=["spreads","totals","h2h"],
        help="Core markets supported; props coverage varies by book."
    )
    regions = c2.selectbox("Region(s)", ["us","us2"], index=0)
    odds_fmt = c3.selectbox("Odds format", ["american","decimal"], index=0)
    auto_refresh = c4.toggle("Auto-refresh (60s)", value=False)

    book_filter = st.text_input(
        "Filter to sportsbooks (comma-separated keys, e.g., betonlineag,draftkings,fanduel) — leave blank for all",
        value=""
    )

    # Cache results briefly to avoid rate-limit thrash
    @st.cache_data(ttl=60, show_spinner=False)
    def fetch_odds(mkts: str, regions: str, fmt: str, key: str):
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
        params = {"regions": regions, "markets": mkts, "oddsFormat": fmt, "apiKey": key}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    # Auto-refresh (simple timer)
    if auto_refresh:
        st.caption("Auto-refreshing every 60 seconds…")
        st.experimental_rerun  # noop reference to keep linters happy
        st.session_state.setdefault("last_tick", 0.0)
        now = time.time()
        if now - st.session_state["last_tick"] >= 60:
            st.session_state["last_tick"] = now
            st.experimental_rerun()

    # Fetch & flatten
    try:
        with st.spinner("Fetching live board…"):
            data = fetch_odds(",".join(markets), regions, odds_fmt, ODDS_KEY)
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        st.stop()

    # Flatten JSON → rows
    rows = []
    try:
        for game in data:
            gid = game.get("id")
            commence = game.get("commence_time")
            home = game.get("home_team"); away = game.get("away_team")
            for bk in game.get("bookmakers", []):
                bk_key = (bk.get("key") or "").strip()
                bk_title = bk.get("title")
                if book_filter:
                    allowed = [b.strip() for b in book_filter.split(",") if b.strip()]
                    if bk_key not in allowed:
                        continue
                for market in bk.get("markets", []):
                    mkey = market.get("key")
                    for out in market.get("outcomes", []):
                        rows.append({
                            "game_id": gid,
                            "time": commence,
                            "book_key": bk_key,
                            "book_name": bk_title,
                            "market": mkey,
                            "team": out.get("name"),
                            "price": out.get("price"),
                            "point": out.get("point"),
                            "home": home,
                            "away": away,
                        })
    except Exception as e:
        st.error(f"Parse failed: {e}")
        st.stop()

    if not rows:
        st.warning("No rows returned. Try removing the book filter or change markets.")
    df = pd.DataFrame(rows)
    st.write("### Board preview")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download current board (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="ncaaf_live_odds.csv",
        mime="text/csv"
    )

# ===========================
# Tab 2: Manual CSV upload (backup)
# ===========================
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
