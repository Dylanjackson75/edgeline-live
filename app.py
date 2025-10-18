import os, time, math
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -------------------- Setup -------------------- #
st.set_page_config(page_title="EdgeLine — Predict. Play. Profit.", layout="wide")

ODDS_KEY = st.secrets.get("THE_ODDS_API_KEY", "") or os.getenv("THE_ODDS_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

HAS_AI = bool(OPENAI_KEY)
if HAS_AI:
    try:
        from openai import OpenAI
        oai = OpenAI(api_key=OPENAI_KEY)
    except Exception:
        HAS_AI = False  # degrade gracefully

# -------------------- Branding -------------------- #
st.markdown(
    "<h1 style='margin-bottom:0'>EdgeLine</h1>"
    "<div style='color:#00FF88;font-weight:600;margin-top:2px'>Predict. Play. Profit.</div>",
    unsafe_allow_html=True,
)
st.caption("Live odds, value ranking, props, write-ups, bet tracking, and chat — all in one place.")
SPORTS = {
    "CFB":  {"odds_key": "americanfootball_ncaaf", "default_markets": ["spreads","totals","h2h"]},
    "NCAAB":{"odds_key": "basketball_ncaab",        "default_markets": ["spreads","totals","h2h","spreads:1st_half","totals:1st_half","spreads:1st_quarter","totals:1st_quarter"]},
    "NBA":  {"odds_key": "basketball_nba",          "default_markets": ["spreads","totals","h2h","spreads:1st_half","totals:1st_half","spreads:1st_quarter","totals:1st_quarter"]},
    "NHL":  {"odds_key": "icehockey_nhl",           "default_markets": ["spreads","totals","h2h","h2h_3way"]},  # 3-way (regulation) too
    "UFC":  {"odds_key": "mma_mixed_martial_arts",  "default_markets": ["h2h"]}  # can add method/round props later
}

# map “pretty” market tags -> The Odds API market keys
MARKET_ALIASES = {
    "FT Spread": "spreads",
    "FT Total":  "totals",
    "Moneyline": "h2h",
    "1H Spread": "spreads:1st_half",
    "1H Total":  "totals:1st_half",
    "1Q Spread": "spreads:1st_quarter",
    "1Q Total":  "totals:1st_quarter",
    "3-Way ML":  "h2h_3way"
}
# -------------------- Helpers -------------------- #
KNOWN_BOOKS = {
    "betonlineag": "BetOnline",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "caesars": "Caesars",
    "betmgm": "BetMGM",
    "pointsbetus": "PointsBet",
    "betrivers": "BetRivers",
    "pinnacle": "Pinnacle",
    "williamhill_us": "William Hill",
}

KEY_NUMBERS = {3,7,10,14}

def fetch_odds(markets_csv: str, regions: str, fmt: str, key: str):
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
    params = {"regions": regions, "markets": markets_csv, "oddsFormat": fmt, "apiKey": key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def american_to_implied(price: float) -> float:
    if pd.isna(price): return np.nan
    p = float(price)
    return abs(p)/(abs(p)+100) if p<0 else 100/(p+100)

def remove_vig_two_way(p1, p2):
    s = p1 + p2
    if s == 0 or any(pd.isna([p1,p2])): return (np.nan, np.nan)
    return (p1/s, p2/s)

def label_compact(r: pd.Series) -> str:
    team, price, point, mkt = r.get("team"), r.get("price"), r.get("point"), r.get("market")
    if mkt == "totals":
        pt = "" if pd.isna(point) else f"{point:g}"
        pr = "" if pd.isna(price) else f" ({int(price)})"
        return f"{str(team).upper()} {pt}{pr}"
    pt = "" if pd.isna(point) else f"{point:+g}"
    pr = "" if pd.isna(price) else f" ({int(price)})"
    return f"{team} {pt}{pr}"

def key_number_cross(prev_point, new_point):
    try:
        a, b = abs(float(prev_point)), abs(float(new_point))
    except Exception:
        return False
    return any(min(a,b) < k <= max(a,b) for k in KEY_NUMBERS)

def best_flag_group(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g["_rank"] = g["price"].abs()
    idx = g["_rank"].idxmin()
    g["_best"] = False
    if pd.notna(idx): g.loc[idx, "_best"] = True
    return g

def median_fair_price(df_gbt: pd.DataFrame) -> float:
    """Use median price across books as a crude 'consensus' fair for this outcome."""
    if df_gbt.empty: return np.nan
    return float(np.nanmedian(pd.to_numeric(df_gbt["price"], errors="coerce").values))

# -------------------- Sidebar -------------------- #
with st.sidebar:
    st.header("Controls")

    sport = st.selectbox("Sport / League", list(SPORTS.keys()), index=0)
    sport_info = SPORTS[sport]

    # Default markets per sport (editable)
    default_markets = sport_info["default_markets"]
    markets = st.multiselect(
        "Markets",
        options=sorted(set(default_markets + ["spreads","totals","h2h","spreads:1st_half","totals:1st_half","spreads:1st_quarter","totals:1st_quarter","h2h_3way"])),
        default=default_markets
    )

    regions = st.selectbox("Region", ["us","us2"], 0)
    odds_fmt = st.selectbox("Odds format", ["american","decimal"], 0)
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)

    books_preset = st.selectbox("Books preset", ["(All)","BetOnline only","DK only","FD only","DK+FD+BOL"], 0)
    query = st.text_input("Search team/matchup/fighter", "")

    st.divider()
    st.caption("API keys")
    st.success("The Odds API key loaded" if ODDS_KEY else "Add THE_ODDS_API_KEY in Secrets to fetch live odds.")
    if HAS_AI: st.success("OpenAI key loaded (AI write-ups/chat on)")

# -------------------- Tabs -------------------- #
tab_dashboard, tab_props, tab_writeups, tab_upload, tab_tracker, tab_chat = st.tabs(
    ["Top Value Bets","Props","Write-Ups","Upload Board","Bet Tracker","Chat"]
)

# Keep session storage
if "last_board" not in st.session_state:
    st.session_state["last_board"] = pd.DataFrame()
if "writeups" not in st.session_state:
    st.session_state["writeups"] = {}  # key: matchup -> text
if "bets" not in st.session_state:
    st.session_state["bets"] = []      # list of dicts

# =========================================================
# TAB 1 — TOP VALUE BETS (Live + movement + best price)
# =========================================================
with tab_dashboard:
    st.subheader("Top Value Bets (live)")

 rows = []
if ODDS_KEY:
    try:
        with st.spinner(f"Fetching live odds for {sport}…"):
            data = fetch_odds_for_sport(
                sport_info["odds_key"], ",".join(markets), regions, odds_fmt, ODDS_KEY
            )
        for game in data:
            gid = game.get("id")
            t = game.get("commence_time")
            home, away = game.get("home_team"), game.get("away_team")
            # UFC: some feeds use 'home_team'/'away_team' loosely; still fine for listing fights.
            for bk in game.get("bookmakers", []):
                bk_key = (bk.get("key") or "").strip()
                bk_name = KNOWN_BOOKS.get(bk_key, bk.get("title"))
                for m in bk.get("markets", []):
                    mkey = m.get("key")  # e.g., spreads, totals, h2h, h2h_3way, spreads:1st_half
                    for out in m.get("outcomes", []):
                        rows.append({
                            "sport": sport, "game_id": gid, "time": t,
                            "book_key": bk_key, "book_name": bk_name,
                            "market": mkey, "team": out.get("name"),
                            "price": out.get("price"), "point": out.get("point"),
                            "home": home, "away": away
                        })
    except requests.RequestException as e:
        st.error(f"Odds fetch failed: {e}")
else:
    st.info(f"Demo mode — no THE_ODDS_API_KEY. Showing {sport} placeholders.")
    # (optional) keep your demo rows here if you like
            {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline","market":"spreads","team":"ALABAMA","price":-110,"point":-6.5,"home":"ALABAMA","away":"TENNESSEE"},
            {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"draftkings","book_name":"DraftKings","market":"spreads","team":"ALABAMA","price":-105,"point":-6.0,"home":"ALABAMA","away":"TENNESSEE"},
            {"game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"fanduel","book_name":"FanDuel","market":"totals","team":"OVER","price":-102,"point":51.5,"home":"ALABAMA","away":"TENNESSEE"},
        ]

    df = pd.DataFrame(rows)
    for c in ("price","point"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Books filter
    available_books = sorted(df.get("book_key", pd.Series([])).dropna().unique().tolist())
    preselect = available_books
    if books_preset == "BetOnline only":
        preselect = ["betonlineag"]
    elif books_preset == "DK only":
        preselect = ["draftkings"]
    elif books_preset == "FD only":
        preselect = ["fanduel"]
    elif books_preset == "DK+FD+BOL":
        preselect = ["draftkings","fanduel","betonlineag"]
    selected_books = st.multiselect("Books", options=available_books, default=preselect, format_func=lambda k: KNOWN_BOOKS.get(k,k))
    if selected_books:
        df = df[df["book_key"].isin(selected_books)]

    # Search filter
    if query:
        q = query.strip().lower()
        mask = (
            df["home"].fillna("").str.lower().str.contains(q)
            | df["away"].fillna("").str.lower().str.contains(q)
            | (df["home"].fillna("") + " @ " + df["away"].fillna("")).str.lower().str.contains(q)
        )
        df = df[mask]

    # Best price ⭐ per game/market/team
    if not df.empty:
        df = df.groupby(["game_id","market","team"], group_keys=False).apply(best_flag_group)
        df["_flag"] = np.where(df["_best"], "⭐ best", "")

    # Movement deltas vs last snapshot
    def compute_deltas(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
        if prev_df is None or prev_df.empty or curr_df.empty:
            curr_df["_move"] = ""; curr_df["_steam"] = False; curr_df["_crossed_key"] = False
            return curr_df
        keys = ["game_id","book_key","market","team"]
        prev = prev_df.copy(); curr = curr_df.copy()
        for c in ("price","point"):
            prev[c] = pd.to_numeric(prev.get(c), errors="coerce")
            curr[c] = pd.to_numeric(curr.get(c), errors="coerce")
        merged = curr.merge(
            prev[keys+["price","point"]].rename(columns={"price":"price_prev","point":"point_prev"}),
            on=keys, how="left"
        )
        merged["d_price"] = merged["price"] - merged["price_prev"]
        merged["d_point"] = merged["point"] - merged["point_prev"]
        merged["_steam"] = ((merged["d_point"].abs() >= 1.0) | (merged["d_price"].abs() >= 20)).fillna(False)
        merged["_crossed_key"] = merged.apply(lambda r: key_number_cross(r.get("point_prev"), r.get("point")), axis=1)
        def fmt(r):
            parts=[]; 
            if pd.notna(r.get("d_point")) and r["d_point"]!=0: parts.append(f"Δpt {r['d_point']:+g}")
            if pd.notna(r.get("d_price")) and r["d_price"]!=0: parts.append(f"Δ¢ {int(r['d_price']):+d}")
            return " / ".join(parts)
        merged["_move"] = merged.apply(fmt, axis=1)
        return merged

    curr = compute_deltas(st.session_state["last_board"], df.copy())
    st.session_state["last_board"] = df.copy()

    # Rank by best/steam/key/move magnitude
    if not curr.empty:
        curr["_move_score"] = curr["d_point"].abs().fillna(0)*10 + curr["d_price"].abs().fillna(0)/10
        curr = curr.sort_values(["_best","_crossed_key","_steam","_move_score"], ascending=[False,False,False,False])

    # Quick consensus-based edge (median price across books as proxy)
    # For each (game, market, team), compute median price and "edge cents" vs each book price
    if not curr.empty:
        edges=[]
        for (gid,mkt,tm), g in curr.groupby(["game_id","market","team"]):
            fair = median_fair_price(g)
            for _, r in g.iterrows():
                edge_cents = np.nan
                if pd.notna(fair) and pd.notna(r["price"]):
                    edge_cents = int(r["price"] - fair)
                edges.append({**r.to_dict(), "fair_price_consensus": fair, "edge_cents_vs_consensus": edge_cents})
        curr = pd.DataFrame(edges)

    st.caption("Badges: ⭐ Best price · 🔥 Steam move · 🎯 Crossed key number")
    st.dataframe(
        curr[["time","home","away","book_name","market","team","point","price","fair_price_consensus","edge_cents_vs_consensus","_move","_flag","_steam","_crossed_key"]],
        use_container_width=True, hide_index=True
    )

    # Compact view
    if st.toggle("Compact view", value=False):
        d2 = curr[["time","home","away","book_key","market","team","point","price"]].copy()
        d2["label"] = d2.apply(label_compact, axis=1)
        pv = (
            d2.pivot_table(
                index=["time","home","away","book_key"],
                columns="market",
                values="label",
                aggfunc=lambda x: " / ".join(sorted(set(x))),
            ).reset_index().fillna("")
        )
        st.dataframe(pv, use_container_width=True)
        st.download_button("Download compact CSV", pv.to_csv(index=False).encode("utf-8"), "compact_odds.csv", "text/csv")

    if auto_refresh:
        if "last_tick" not in st.session_state: st.session_state["last_tick"] = 0.0
        if time.time() - st.session_state["last_tick"] >= 60:
            st.session_state["last_tick"] = time.time()
            st.experimental_rerun()

# =================================
# TAB 2 — PROPS (no-vig + edges)
# =================================
with tab_props:
    st.subheader("Player Props — No-Vig Splits & Edges")
    st.caption("Select a game or search a player; we use book prices to compute no-vig probabilities.")
    want = st.multiselect("Prop markets", ["player_pass_yds","player_rush_yds","player_rec_yds","player_pass_tds","player_receptions"], default=["player_pass_yds","player_rush_yds","player_rec_yds"])
    props_df = pd.DataFrame()

    if ODDS_KEY and want:
        try:
            with st.spinner("Fetching props…"):
                raw = fetch_odds(",".join(want), regions, odds_fmt, ODDS_KEY)
            rows=[]
            for game in raw:
                gid = game.get("id"); t = game.get("commence_time")
                home, away = game.get("home_team"), game.get("away_team")
                for bk in game.get("bookmakers", []):
                    bk_key = (bk.get("key") or "").strip()
                    bk_name = KNOWN_BOOKS.get(bk_key, bk.get("title"))
                    for m in bk.get("markets", []):
                        mkey = m.get("key")
                        for out in m.get("outcomes", []):
                            rows.append({
                                "game_id": gid, "time": t, "book_key": bk_key, "book_name": bk_name,
                                "market": mkey, "participant": out.get("name"),
                                "price": out.get("price"), "point": out.get("point"),
                                "home": home, "away": away
                            })
            props_df = pd.DataFrame(rows)
        except requests.RequestException as e:
            st.error(f"Props fetch failed: {e}")
    else:
        st.info("Add THE_ODDS_API_KEY in Secrets and select markets to fetch props.")

    if not props_df.empty:
        props_df["price"] = pd.to_numeric(props_df["price"], errors="coerce")
        props_df["point"] = pd.to_numeric(props_df["point"], errors="coerce")
        # Build no-vig on paired O/U rows by (game, market, participant, point)
        # The Odds API props outcomes do not always label OVER/UNDER; we use two-way pairs by identical key+point.
        # For display simplicity, show raw prices and implieds; advanced pairing can be added later per book format.
        props_df["implied"] = props_df["price"].apply(american_to_implied)
        st.dataframe(props_df[["time","home","away","book_name","market","participant","point","price","implied"]], use_container_width=True)
        st.download_button("Download props CSV", props_df.to_csv(index=False).encode("utf-8"), "props.csv", "text/csv")

# =================================
# TAB 3 — WRITE-UPS (manual + AI)
# =================================
with tab_writeups:
    st.subheader("Game Write-Ups")
    st.caption("Keep notes or click Generate to draft an AI write-up for the matchup.")
    game_title = st.text_input("Matchup (e.g., LSU @ Florida)")
    colA, colB = st.columns([3,1])
    with colA:
        text = st.text_area("Write-up", st.session_state["writeups"].get(game_title, ""), height=220)
    with colB:
        if HAS_AI and st.button("Generate with AI"):
            prompt = f"Write a concise, data-aware betting analysis for the college football matchup: {game_title}. Structure with: EdgeLine Model Lean, Why it matters (3 bullets), Risk flags (tempo, injuries, weather, key numbers), Suggested stake guidance."
            try:
                resp = oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"You are EdgeLine's betting analyst."},{"role":"user","content":prompt}],
                    temperature=0.7,
                    max_tokens=300
                )
                ai_txt = resp.choices[0].message.content.strip()
                text = ai_txt
            except Exception as e:
                st.error(f"AI error: {e}")
        if st.button("Save"):
            if game_title:
                st.session_state["writeups"][game_title] = text
                st.success("Saved.")
    if st.session_state["writeups"]:
        st.markdown("#### Saved Write-Ups")
        for k,v in st.session_state["writeups"].items():
            with st.expander(k):
                st.write(v)

# =================================
# TAB 4 — UPLOAD BOARD
# =================================
with tab_upload:
    st.subheader("Upload Board (CSV/TSV)")
    up = st.file_uploader("Upload BetOnline or compatible board", type=["csv","tsv","txt"])
    if up:
        try:
            df_up = pd.read_csv(up, sep=None, engine="python")
        except Exception:
            up.seek(0)
            df_up = pd.read_csv(up)
        st.success("Loaded.")
        st.dataframe(df_up.head(250), use_container_width=True)
        st.download_button("Download copy", df_up.to_csv(index=False).encode("utf-8"), "board_copy.csv", "text/csv")

# =================================
# TAB 5 — BET TRACKER
# =================================
with tab_tracker:
    st.subheader("Bet Tracker")
    c1,c2,c3,c4,c5,c6 = st.columns([1.6,1,1,1,1,1])
    with c1:
        gm = st.text_input("Game (e.g., LSU @ Florida)")
    with c2:
        market = st.selectbox("Market", ["FT Spread","FT Total","1H Spread","1H Total","ML"])
    with c3:
        pick = st.text_input("Pick (e.g., LSU -3.5 / OVER 55.5 / LSU ML)")
    with c4:
        odds = st.number_input("Odds (American)", value=-110, step=1)
    with c5:
        stake = st.number_input("Stake ($)", value=50.0, step=5.0)
    with c6:
        cl = st.text_input("Closing line (optional)", value="")
    if st.button("Add bet"):
        if gm and pick:
            st.session_state["bets"].append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "game": gm, "market": market, "pick": pick,
                "odds": int(odds), "stake": float(stake),
                "closing_line": cl, "result":"open", "payout":0.0
            })
        else:
            st.warning("Enter Game and Pick.")
