import os, time
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -------------------- App header -------------------- #
st.set_page_config(page_title="EdgeLine — Live Odds", layout="wide")
st.title("EdgeLine — Live Odds +EV Workspace")
st.image(
    "https://raw.githubusercontent.com/dylanjackson75-sketch/edgeline-live/main/assets/edgeline_logo.png",
    width=200,
)
# -------------------- Secrets / keys -------------------- #
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
            help="Paste your The Odds API key if Secrets/Env not set.",
            key="odds_key_manual",
        )
        if _manual:
            st.session_state["odds_key_cached"] = _manual
    ODDS_KEY = st.session_state.get("odds_key_cached", "")
    key_source = "Manual" if ODDS_KEY else None

if key_source:
    st.caption(f"Using The Odds API key from **{key_source}**.")
else:
    st.warning("No Odds API key detected. Add it in Streamlit **⋯ → Edit secrets** or paste it in the sidebar.")

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

def fetch_odds(markets_csv: str, regions: str, fmt: str, key: str):
    """Fetch odds from The Odds API for NCAAF."""
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
    params = {"regions": regions, "markets": markets_csv, "oddsFormat": fmt, "apiKey": key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def label_row_for_compact(r: pd.Series) -> str:
    # Build a compact label like "ALA -6.5 (-110)" or "OVER 51.5 (-105)"
    team = r.get("team")
    price = r.get("price")
    point = r.get("point")
    if r.get("market") == "totals":
        pt = "" if pd.isna(point) else f"{point:g}"
        pr = "" if pd.isna(price) else f" ({int(price)})"
        return f"{str(team).upper()} {pt}{pr}"
    else:
        pt = "" if pd.isna(point) else f"{point:+g}"
        pr = "" if pd.isna(price) else f" ({int(price)})"
        return f"{team} {pt}{pr}"

def best_flag_group(group: pd.DataFrame) -> pd.DataFrame:
    # Mark the outcome with the smallest absolute price (best deal) per game/market/team
    g = group.copy()
    g["_rank"] = g["price"].abs()
    idx = g["_rank"].idxmin()
    g["_best"] = False
    if pd.notna(idx):
        g.loc[idx, "_best"] = True
    return g

# -------------------- Tabs -------------------- #
tab1, tab2, tab3 = st.tabs(["Auto-fetch odds (NCAAF)", "Upload board (CSV)", "Bet tracker"])

# ========================= TAB 1 ========================= #
with tab1:
    st.subheader("Live Board")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    markets = c1.multiselect(
        "Markets",
        ["spreads", "totals", "h2h", "player_pass_yds", "player_rush_yds", "player_rec_yds"],
        default=["spreads", "totals", "h2h"],
    )
    regions = c2.selectbox("Region", ["us", "us2"], index=0)
    odds_fmt = c3.selectbox("Odds format", ["american", "decimal"], index=0)
    auto_refresh = c4.toggle("Auto-refresh (60s)", value=False)

    rows = []
    if ODDS_KEY.strip():
        try:
            with st.spinner("Fetching live board…"):
                data = fetch_odds(",".join(markets), regions, odds_fmt, ODDS_KEY.strip())
            for game in data:
                gid = game.get("id")
                t = game.get("commence_time")
                home = game.get("home_team")
                away = game.get("away_team")
                for bk in game.get("bookmakers", []):
                    bk_key = (bk.get("key") or "").strip()
                    bk_name = KNOWN_BOOKS.get(bk_key, bk.get("title"))
                    for m in bk.get("markets", []):
                        mkey = m.get("key")
                        for out in m.get("outcomes", []):
                            rows.append(
                                {
                                    "game_id": gid,
                                    "time": t,
                                    "book_key": bk_key,
                                    "book_name": bk_name,
                                    "market": mkey,
                                    "team": out.get("name"),
                                    "price": out.get("price"),
                                    "point": out.get("point"),
                                    "home": home,
                                    "away": away,
                                }
                            )
        except requests.HTTPError as e:
            st.error(f"Odds API error: {e.response.status_code} {e.response.text[:200]}")
            st.stop()
        except requests.RequestException as e:
            st.error(f"Network error contacting Odds API: {e}")
            st.stop()
    else:
        st.info("No API key provided — showing demo rows. Add THE_ODDS_API_KEY in **⋯ → Edit secrets** to fetch live odds.")
        rows = [
            {
                "game_id": "demo1",
                "time": "2025-10-18T20:00:00Z",
                "book_key": "betonlineag",
                "book_name": "BetOnline",
                "market": "spreads",
                "team": "ALABAMA",
                "price": -110,
                "point": -6.5,
                "home": "ALABAMA",
                "away": "TENNESSEE",
            },
            {
                "game_id": "demo1",
                "time": "2025-10-18T20:00:00Z",
                "book_key": "betonlineag",
                "book_name": "BetOnline",
                "market": "totals",
                "team": "OVER",
                "price": -105,
                "point": 51.5,
                "home": "ALABAMA",
                "away": "TENNESSEE",
            },
            {
                "game_id": "demo2",
                "time": "2025-10-18T23:30:00Z",
                "book_key": "draftkings",
                "book_name": "DraftKings",
                "market": "h2h",
                "team": "GEORGIA",
                "price": -180,
                "point": None,
                "home": "GEORGIA",
                "away": "KENTUCKY",
            },
        ]

    df = pd.DataFrame(rows)

    # ---- Books preset + multiselect ----
    available_books = sorted(df.get("book_key", pd.Series([])).dropna().unique().tolist())
    pretty = lambda k: KNOWN_BOOKS.get(k, k)

    cB1, cB2 = st.columns([1, 2])
    preset = cB1.selectbox("Books preset", ["(All)", "BetOnline only", "DK only", "FD only", "DK + FD + BOL"])
    preselect = []
    if preset == "BetOnline only":
        preselect = ["betonlineag"]
    elif preset == "DK only":
        preselect = ["draftkings"]
    elif preset == "FD only":
        preselect = ["fanduel"]
    elif preset == "DK + FD + BOL":
        preselect = ["draftkings", "fanduel", "betonlineag"]

    selected_books = cB2.multiselect(
        "Or pick specific books",
        options=available_books,
        default=preselect if preselect else available_books,
        format_func=pretty,
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

    # ---- Normalize numeric columns ----
    for c in ("price", "point"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- ⭐ Best price flag per game/market/team ----
    if not df.empty:
        df = df.groupby(["game_id", "market", "team"], group_keys=False).apply(best_flag_group)
        df["_flag"] = np.where(df.get("_best", False), "⭐ best", "")

    st.write("### Board preview")
    st.dataframe(df.drop(columns=["_rank"], errors="ignore"), use_container_width=True)

    # ---- Compact pivot view ----
    compact = st.toggle("Compact view (per game x book)", value=False)
    if compact and not df.empty:
        base_cols = ["time", "home", "away", "book_key", "market", "team", "point", "price"]
        d2 = df[base_cols].copy()
        d2["label"] = d2.apply(label_row_for_compact, axis=1)
        pv = (
            d2.pivot_table(
                index=["time", "home", "away", "book_key"],
                columns="market",
                values="label",
                aggfunc=lambda x: " / ".join(sorted(set(x))),
            )
            .reset_index()
            .fillna("")
        )
        st.dataframe(pv, use_container_width=True)
        st.download_button("Download compact CSV", pv.to_csv(index=False).encode("utf-8"), "compact_odds.csv", "text/csv")

    # ---- Auto-refresh every 60s ----
    if auto_refresh:
        st.caption("Auto-refreshing every 60 seconds…")
        if "last_tick" not in st.session_state:
            st.session_state["last_tick"] = 0.0
        now = time.time()
        if now - st.session_state["last_tick"] >= 60:
            st.session_state["last_tick"] = now
            st.experimental_rerun()

# ---- Live movement monitoring & auto-adjust (drop-in) ----
import time
import pandas as pd
import numpy as np
import streamlit as st

KEY_NUMBERS = {3, 7, 10, 14}

def key_number_edge(old_point, new_point):
    try:
        old_abs, new_abs = abs(float(old_point)), abs(float(new_point))
    except Exception:
        return False
    # Flag if we crossed any key number between old -> new
    crossed = any(min(old_abs, new_abs) < k <= max(old_abs, new_abs) for k in KEY_NUMBERS)
    return crossed

def compute_deltas(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
    if prev_df is None or prev_df.empty or curr_df is None or curr_df.empty:
        curr_df["_move"] = ""
        curr_df["_steam"] = False
        curr_df["_crossed_key"] = False
        return curr_df

    keys = ["game_id", "book_key", "market", "team"]
    prev = prev_df.copy()
    curr = curr_df.copy()
    for c in ("price","point"):
        if c in prev.columns: prev[c] = pd.to_numeric(prev[c], errors="coerce")
        if c in curr.columns: curr[c] = pd.to_numeric(curr[c], errors="coerce")

    merged = curr.merge(
        prev[keys + ["price","point"]].rename(columns={"price":"price_prev","point":"point_prev"}),
        on=keys, how="left"
    )
    merged["d_price"] = merged["price"] - merged["price_prev"]
    merged["d_point"] = merged["point"] - merged["point_prev"]
    # Steam heuristics
    merged["_steam"] = (
        (merged["d_point"].abs() >= 1.0) |
        (merged["d_price"].abs() >= 20)
    ).fillna(False)

    merged["_crossed_key"] = merged.apply(
        lambda r: key_number_edge(r.get("point_prev"), r.get("point")), axis=1
    )
    # Pretty move badge
    def fmt_move(r):
        p = r.get("d_point")
        c = r.get("d_price")
        parts = []
        if pd.notna(p) and p != 0: parts.append(f"Δpt {p:+g}")
        if pd.notna(c) and c != 0: parts.append(f"Δ¢ {int(c):+d}")
        return " / ".join(parts)
    merged["_move"] = merged.apply(fmt_move, axis=1)
    return merged

# Keep last snapshot in session
if "last_board" not in st.session_state:
    st.session_state["last_board"] = pd.DataFrame()

# After you build `df` with fresh odds (your existing code), add:
curr = df.copy()
curr = compute_deltas(st.session_state["last_board"], curr)

# Re-rank: prioritize best price, then biggest favorable move, then time
if not curr.empty:
    # Favorable means price improved (more + for dogs, more - for favs) or spread moved to our side
    # Here we just sort by magnitude of move and best-price flag
    curr["_move_score"] = curr["d_point"].abs().fillna(0)*10 + curr["d_price"].abs().fillna(0)/10
    curr = curr.sort_values(["_best","_crossed_key","_steam","_move_score"], ascending=[False, False, False, False])

# UI badges
st.write("### Live moves")
st.caption("Badges: ⭐ Best price · 🔥 Steam move · 🎯 Crossed key number (3/7/10/14)")
if not curr.empty:
    view = curr.copy()
    view["_badges"] = (
        np.where(view.get("_best", False), "⭐", "")
        + np.where(view.get("_steam", False), "🔥", "")
        + np.where(view.get("_crossed_key", False), "🎯", "")
    )
    st.dataframe(
        view[["time","home","away","book_name","market","team","point","price","_move","_badges"]],
        use_container_width=True,
        hide_index=True
    )

# Auto-adjust picks (toggle)
auto_adjust = st.toggle("Auto-adjust picks on movement", value=True,
                        help="Recompute and drop plays when edge/Kelly falls below thresholds after a move.")

EDGE_MIN_KELLY = st.slider("Min Kelly to keep a play", 0.0, 1.0, 0.0025, 0.0005, help="0.25% default")
if auto_adjust and not curr.empty:
    # If you already compute kelly_* columns elsewhere, filter by them here.
    kelly_cols = [c for c in curr.columns if c.startswith("kelly_")]
    if kelly_cols:
        curr = curr[curr[kelly_cols].max(axis=1) >= EDGE_MIN_KELLY]

# Persist snapshot for next diff
st.session_state["last_board"] = df.copy()

# ========================= TAB 2 ========================= #
with tab2:
    st.subheader("Upload board (CSV/TSV)")
    st.write("Upload a BetOnline board export or any compatible CSV/TSV.")
    f = st.file_uploader("betonline_lines.csv", type=["csv", "tsv", "txt"])
    if f is not None:
        try:
            df_up = pd.read_csv(f, sep=None, engine="python")
        except Exception:
            f.seek(0)
            df_up = pd.read_csv(f)
        st.success("File loaded")
        st.dataframe(df_up.head(200), use_container_width=True)
        st.download_button("Download copy (CSV)", df_up.to_csv(index=False).encode("utf-8"), "board_copy.csv", "text/csv")

# ========================= TAB 3 ========================= #
with tab3:
    st.subheader("Bet tracker")

    if "bets" not in st.session_state:
        st.session_state["bets"] = []

    c1, c2, c3, c4, c5 = st.columns([1.6, 1, 1, 1, 1])
    with c1:
        game = st.text_input("Game (e.g., TENN @ ALA)")
    with c2:
        market = st.selectbox("Market", ["FT Spread", "FT Total", "1H Spread", "1H Total", "ML"])
    with c3:
        pick = st.text_input("Pick (e.g., ALA -6.5 / OVER / ALA ML)")
    with c4:
        odds = st.number_input("Odds (American)", value=-110, step=1)
    with c5:
        stake = st.number_input("Stake ($)", value=50.0, step=5.0)

    if st.button("Add bet"):
        if game and pick:
            st.session_state["bets"].append(
                {
                    "game": game,
                    "market": market,
                    "pick": pick,
                    "odds": int(odds),
                    "stake": float(stake),
                    "result": "open",
                    "payout": 0.0,
                }
            )
        else:
            st.warning("Add at least a Game and Pick before adding.")

    if st.session_state["bets"]:
        dfb = pd.DataFrame(st.session_state["bets"])
        st.dataframe(dfb, use_container_width=True)

        st.markdown("### Grade a result")
        i = st.number_input("Row #", min_value=0, max_value=len(dfb) - 1, value=0)
        res = st.selectbox("Result", ["open", "win", "loss", "push"], index=0)
        if st.button("Update result"):
            bet = st.session_state["bets"][i]
            bet["result"] = res
            odd = bet["odds"]
            stak = bet["stake"]
            if res == "win":
                dec = 1 + (100 / abs(odd) if odd < 0 else odd / 100)
                bet["payout"] = round(stak * dec - stak, 2)
            elif res == "loss":
                bet["payout"] = -stak
            elif res == "push":
                bet["payout"] = 0.0

        # Summary
        dfb = pd.DataFrame(st.session_state["bets"])
        closed = dfb[dfb["result"].isin(["win", "loss", "push"])]
        wins = (closed["result"] == "win").sum()
        losses = (closed["result"] == "loss").sum()
        pushes = (closed["result"] == "push").sum()
        profit = float(closed["payout"].sum()) if not closed.empty else 0.0
        risked = float(closed.loc[closed["result"] != "push", "stake"].sum()) if not closed.empty else 0.0
        roi = (profit / risked * 100) if risked > 0 else 0.0

        cA, cB, cC = st.columns(3)
        cA.metric("Record", f"{wins}-{losses}-{pushes}")
        cB.metric("Profit", f"${profit:,.2f}")
        cC.metric("ROI", f"{roi:.2f}%")

        st.download_button("Export bets (CSV)", dfb.to_csv(index=False).encode("utf-8"), "bets.csv", "text/csv")
    else:
        st.info("No bets yet. Add one above.")
