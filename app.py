# app.py  — minimal sanity check
import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

def main():
    st.set_page_config(page_title="EdgeLine - Live Odds", layout="wide")
    st.title("EdgeLine — Live Odds +EV Workspace")
    st.success("✅ Imports OK. Streamlit booted. If you can see this, your env and header are fine.")

if __name__ == "__main__":
    main()
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
    st.subheader("NCAAF Odds — Live Board")

    # --- Controls (keep these as you had) ---
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    markets = c1.multiselect(
        "Markets",
        ["spreads","totals","h2h","player_pass_yds","player_rush_yds","player_rec_yds"],
        default=["spreads","totals","h2h"]
    )
    regions = c2.selectbox("Region", ["us","us2"], index=0)
    odds_fmt = c3.selectbox("Odds format", ["american","decimal"], index=0)
    auto_refresh = c4.toggle("Auto-refresh (60s)", value=False)

    # Optional preset for books
    KNOWN_BOOKS = {
        "betonlineag":"BetOnline","draftkings":"DraftKings","fanduel":"FanDuel",
        "caesars":"Caesars","betmgm":"BetMGM","pointsbetus":"PointsBet",
        "betrivers":"BetRivers","pinnacle":"Pinnacle","williamhill_us":"William Hill",
    }

    # ---- Fetch data (you already have fetch_odds + ODDS_KEY above) ----
    rows = []
    if ODDS_KEY.strip():
        try:
            with st.spinner("Fetching live board…"):
                data = fetch_odds(",".join(markets), regions, odds_fmt, ODDS_KEY.strip())
            for game in data:
                gid = game.get("id")
                t   = game.get("commence_time")
                home = game.get("home_team"); away = game.get("away_team")
                for bk in game.get("bookmakers", []):
                    bk_key = (bk.get("key") or "").strip()
                    for m in bk.get("markets", []):
                        mkey = m.get("key")
                        for out in m.get("outcomes", []):
                            rows.append({
                                "game_id": gid, "time": t,
                                "book_key": bk_key, "book_name": KNOWN_BOOKS.get(bk_key, bk.get("title")),
                                "market": mkey, "team": out.get("name"),
                                "price": out.get("price"), "point": out.get("point"),
                                "home": home, "away": away,
                            })
        except requests.HTTPError as e:
            st.error(f"Odds API error: {e.response.status_code} {e.response.text[:160]}")
            st.stop()
        except requests.RequestException as e:
            st.error(f"Network error contacting Odds API: {e}")
            st.stop()
    else:
        st.info("No Odds API key detected — showing demo rows.")
        rows = [
            {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline",
             "market":"spreads","team":"ALABAMA","price":-110,"point":-6.5,"home":"ALABAMA","away":"TENNESSEE"},
            {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline",
             "market":"totals","team":"OVER","price":-105,"point":51.5,"home":"ALABAMA","away":"TENNESSEE"},
            {"game_id":"demo2","time":"2025-10-18T23:30:00Z","book_key":"draftkings","book_name":"DraftKings",
             "market":"h2h","team":"GEORGIA","price":-180,"point":None,"home":"GEORGIA","away":"KENTUCKY"},
        ]

    df = pd.DataFrame(rows)

    # ---- Books preset + multiselect filter ----
    available_books = sorted(df.get("book_key", pd.Series([])).dropna().unique().tolist())
    pretty = lambda k: KNOWN_BOOKS.get(k, k)

    cB1, cB2 = st.columns([1,2])
    preset = cB1.selectbox("Books preset", ["(All)", "BetOnline only", "DK only", "FD only", "DK + FD + BOL"])
    preselect = []
    if preset == "BetOnline only": preselect = ["betonlineag"]
    elif preset == "DK only":      preselect = ["draftkings"]
    elif preset == "FD only":      preselect = ["fanduel"]
    elif preset == "DK + FD + BOL": preselect = ["draftkings","fanduel","betonlineag"]

    selected_books = cB2.multiselect(
        "Or pick specific books",
        options=available_books,
        default=preselect if preselect else available_books,
        format_func=pretty
    )
    if selected_books:
        df = df[df["book_key"].isin(selected_books)]

    # ---- Search by team/matchup ----
    q = st.text_input("Search (team or matchup)", value="")
    if q:
        qlow = q.strip().lower()
        df = df[
            df["home"].fillna("").str.lower().str.contains(qlow)
            | df["away"].fillna("").str.lower().str.contains(qlow)
            | (df["home"].fillna("") + " @ " + df["away"].fillna("")).str.lower().str.contains(qlow)
        ]

    # ---- Normalize numeric cols ----
    for c in ("price","point"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Best price flag per game/market/team ----
    def pick_best(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["_rank"] = g["price"].abs()
        idx = g["_rank"].idxmin()
        g["_best"] = False
        if pd.notna(idx): g.loc[idx, "_best"] = True
        return g
    if not df.empty:
        df = df.groupby(["game_id","market","team"], group_keys=False).apply(pick_best)
        df["_flag"] = np.where(df.get("_best", False), "⭐ best", "")

    st.write("### Board preview")
    st.dataframe(df.drop(columns=["_rank"], errors="ignore"), use_container_width=True)

    # ---- Optional compact pivot view ----
    compact = st.toggle("Compact view (per game x book)", value=False)
    if compact and not df.empty:
        base_cols = ["time","home","away","book_key","market","team","point","price"]
        d2 = df[base_cols].copy()

        def label_row(r):
            team = r["team"]
            if r["market"] == "totals":
                pt = "" if pd.isna(r["point"]) else f"{r['point']:g}"
                price = "" if pd.isna(r["price"]) else f" ({int(r['price'])})"
                return f"{team.upper()} {pt}{price}"
            else:
                pt = "" if pd.isna(r["point"]) else f"{r['point']:+g}"
                price = "" if pd.isna(r["price"]) else f" ({int(r['price'])})"
                return f"{team} {pt}{price}"

        d2["label"] = d2.apply(label_row, axis=1)
        pv = d2.pivot_table(
            index=["time","home","away","book_key"],
            columns="market",
            values="label",
            aggfunc=lambda x: " / ".join(sorted(set(x)))
        ).reset_index()
        st.dataframe(pv.fillna(""), use_container_width=True)
        st.download_button(
            "Download compact CSV",
            pv.to_csv(index=False).encode("utf-8"),
            "compact_odds.csv",
            "text/csv"
        )

    # ---- Auto-refresh every 60s ----
    if auto_refresh:
        st.caption("Auto-refreshing every 60 seconds…")
        if "last_tick" not in st.session_state: st.session_state["last_tick"] = 0.0
        now = time.time()
        if now - st.session_state["last_tick"] >= 60:
            st.session_state["last_tick"] = now
            st.experimental_rerun()

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
