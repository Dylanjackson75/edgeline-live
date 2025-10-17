import time, os, requests, pandas as pd, streamlit as st

st.set_page_config(page_title="EdgeLine — Live Odds", layout="wide")
st.title("EdgeLine — Live Odds +EV Workspace")

# ---- Optional backend health (won't block odds) ----
API_BASE = st.secrets.get("API_BASE","")
API_KEY  = st.secrets.get("API_KEY","")
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
    status.info("Backend optional. Live odds will still work.")

# ---- Odds API key resolution: Secrets ➜ Env ➜ Sidebar fallback ----
ODDS_KEY = st.secrets.get("THE_ODDS_API_KEY", "") or os.getenv("THE_ODDS_API_KEY", "")
key_source = None
if ODDS_KEY:
    key_source = "Secrets" if "THE_ODDS_API_KEY" in st.secrets else "Env"
else:
    with st.sidebar:
        st.subheader("Setup (temporary)")
        _manual = st.text_input(
            "THE_ODDS_API_KEY",
            value="",
            type="password",
            help="Paste your The Odds API key (only needed if Secrets/env not set).",
            key="odds_key_manual"
        )
        if _manual:
            st.session_state["odds_key_cached"] = _manual
    ODDS_KEY = st.session_state.get("odds_key_cached", "")
    key_source = "Manual" if ODDS_KEY else None

if key_source:
    st.caption(f"Using The Odds API key from **{key_source}**.")
else:
    st.warning("No Odds API key detected. Add it in Secrets or paste it in the sidebar.")

# ----------------- Helper -----------------
def fetch_odds(markets_csv: str, regions: str, fmt: str, key: str):
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
    params = {"regions": regions, "markets": markets_csv, "oddsFormat": fmt, "apiKey": key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

DEMO_ROWS = pd.DataFrame([
    {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline",
     "market":"spreads","team":"ALABAMA","price":-110,"point":-6.5,"home":"ALABAMA","away":"TENNESSEE"},
    {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline",
     "market":"totals","team":"OVER","price":-105,"point":51.5,"home":"ALABAMA","away":"TENNESSEE"},
    {"game_id":"demo2","time":"2025-10-18T23:30:00Z","book_key":"draftkings","book_name":"DraftKings",
     "market":"h2h","team":"GEORGIA","price":-180,"point":None,"home":"GEORGIA","away":"KENTUCKY"},
])

tab1, tab2, tab3 = st.tabs(["Auto-fetch odds (NCAAF)", "Upload board (CSV)", "Bet tracker"])
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
        st.info("No API key provided — showing demo data. Add key in Secrets or the sidebar to fetch live odds.")
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

# ================= Tab 3: Bet tracker ===================
with tab3:
    st.subheader("Your Bets")
    # init storage
    if "bets" not in st.session_state:
        st.session_state["bets"] = []

    c1,c2,c3,c4,c5 = st.columns([1.6,1,1,1,1])
    with c1: game = st.text_input("Game (e.g., TENN @ ALA)")
    with c2: market = st.selectbox("Market", ["FT Spread","FT Total","1H Spread","1H Total","ML"])
    with c3: pick = st.text_input("Pick (e.g., ALA -6.5 / OVER / ALA ML)")
    with c4: odds = st.number_input("Odds (American)", value=-110, step=1)
    with c5: stake = st.number_input("Stake ($)", value=50.0, step=5.0)

    if st.button("Add bet"):
        if game and pick:
            st.session_state["bets"].append({
                "game": game, "market": market, "pick": pick,
                "odds": int(odds), "stake": float(stake),
                "result": "open", "payout": 0.0
            })
        else:
            st.warning("Add at least a Game and Pick before adding.")

    import pandas as _pd
    if st.session_state["bets"]:
        dfb = _pd.DataFrame(st.session_state["bets"])
        st.dataframe(dfb, use_container_width=True)

        st.markdown("### Grade a result")
        i = st.number_input("Row #", min_value=0, max_value=len(dfb)-1, value=0)
        res = st.selectbox("Result", ["open","win","loss","push"], index=0)
        if st.button("Update result"):
            bet = st.session_state["bets"][i]
            bet["result"] = res
            # compute payout (profit/loss excluding returned stake)
            odd = bet["odds"]; stak = bet["stake"]
            if res == "win":
                dec = 1 + (100/abs(odd) if odd < 0 else odd/100)
                bet["payout"] = round(stak*dec - stak, 2)
            elif res == "loss":
                bet["payout"] = -stak
            elif res == "push":
                bet["payout"] = 0.0

        # summary
        dfb = _pd.DataFrame(st.session_state["bets"])
        closed = dfb[dfb["result"].isin(["win","loss","push"])]
        wins = (closed["result"]=="win").sum()
        losses = (closed["result"]=="loss").sum()
        pushes = (closed["result"]=="push").sum()
        profit = float(closed["payout"].sum()) if not closed.empty else 0.0
        risked = float(closed.loc[closed["result"]!="push","stake"].sum()) if not closed.empty else 0.0
        roi = (profit / risked * 100) if risked > 0 else 0.0

        cA,cB,cC = st.columns(3)
        cA.metric("Record", f"{wins}-{losses}-{pushes}")
        cB.metric("Profit", f"${profit:,.2f}")
        cC.metric("ROI", f"{roi:.2f}%")

        st.download_button(
            "Export bets (CSV)",
            dfb.to_csv(index=False).encode("utf-8"),
            "bets.csv", "text/csv"
        )
    else:
        st.info("No bets yet. Add one above.")
