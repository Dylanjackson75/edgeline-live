import os, time, math
import pandas as pd
import numpy as np
import requests
import streamlit as st
from math import erf, sqrt
# --- EdgeLine branding ---
import base64, pathlib, streamlit as st

ASSETS = pathlib.Path("assets")

st.set_page_config(
    page_title="EdgeLine — Predict. Play. Profit.",
    page_icon=str(ASSETS / "favicon.png"),
    layout="wide",
)

def _inject_css():
    css = f"""
    <style>
      :root {{
        --edge-gold: #D4AF37;
        --edge-green: #7AF57A;
        --edge-black: #0B0B0B;
        --edge-ink:   #F2F2F2;
      }}
      /* Gradient page background */
      .stApp {{
        background: radial-gradient(1200px 600px at 20% -10%, rgba(212,175,55,0.12), transparent 60%),
                    radial-gradient(1200px 600px at 90% 10%, rgba(122,245,122,0.06), transparent 60%),
                    #0B0B0B;
      }}
      /* Cards/panels polish */
      .st-emotion-cache-1r4qj8v, .st-emotion-cache-1dp5vir, .st-emotion-cache-1r6slb0 {{
        border: 1px solid rgba(212,175,55,0.15);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 10px 30px rgba(0,0,0,0.35);
        border-radius: 14px;
      }}
      /* H1/H2 styling */
      h1, .stMarkdown h1 {{ letter-spacing: 0.2px; }}
      h1 strong, h2 strong {{ color: var(--edge-gold); }}
      /* Tagline pill */
      .edge-tagline {{
        display:inline-block; margin-top:4px; padding:6px 10px;
        border:1px solid rgba(122,245,122,0.25); color:#7AF57A; border-radius:999px;
        font-weight:600; font-size:0.9rem; letter-spacing:0.04em;
        background: rgba(122,245,122,0.07);
      }}
      /* Buttons */
      .stButton>button {{
        border-radius: 12px;
        border: 1px solid rgba(212,175,55,0.35);
        background: linear-gradient(180deg, rgba(212,175,55,0.18), rgba(212,175,55,0.10));
        color: var(--edge-ink);
      }}
      .stButton>button:hover {{
        border-color: var(--edge-gold);
        box-shadow: 0 0 0 2px rgba(212,175,55,0.25) inset;
      }}
      /* Links */
      a, .stMarkdown a {{ color: var(--edge-green) !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def edge_header():
    logo_path = ASSETS / "edgeline_logo_dark.png"
    if logo_path.exists():
        st.image(str(logo_path), width=180)
    st.markdown(
        "<h1 style='margin:0'>EdgeLine</h1>"
        "<div class='edge-tagline'>PREDICT. PLAY. PROFIT.</div>",
        unsafe_allow_html=True
    )

_inject_css()

# =========================
# App & Secrets
# =========================
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

# =========================
# Branding (header)
# =========================
st.markdown(
    "<h1 style='margin-bottom:0'>EdgeLine</h1>"
    "<div style='color:#00FF88;font-weight:600;margin-top:2px'>Predict. Play. Profit.</div>",
    unsafe_allow_html=True,
)
st.caption("Multi-sport odds, movement, fair numbers, EV/Kelly, props, write-ups, bet tracking, and chat.")

# =========================
# Sport keys & market aliases
# =========================
SPORTS = {
    "CFB":  {"odds_key": "americanfootball_ncaaf", "default_markets": ["spreads","totals","h2h"]},
    "NCAAB":{"odds_key": "basketball_ncaab",        "default_markets": ["spreads","totals","h2h","spreads:1st_half","totals:1st_half","spreads:1st_quarter","totals:1st_quarter"]},
    "NBA":  {"odds_key": "basketball_nba",          "default_markets": ["spreads","totals","h2h","spreads:1st_half","totals:1st_half","spreads:1st_quarter","totals:1st_quarter"]},
    "NHL":  {"odds_key": "icehockey_nhl",           "default_markets": ["spreads","totals","h2h","h2h_3way"]},
    "UFC":  {"odds_key": "mma_mixed_martial_arts",  "default_markets": ["h2h"]},
}
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

# =========================
# Helper functions
# =========================
def fetch_odds_for_sport(sport_key: str, markets_csv: str, regions: str, fmt: str, key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"regions": regions, "markets": markets_csv, "oddsFormat": fmt, "apiKey": key}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def american_to_decimal(odds: float) -> float:
    return 1 + (100/abs(odds) if odds < 0 else odds/100)

def american_to_implied(price: float) -> float:
    if pd.isna(price): return np.nan
    p = float(price)
    return abs(p)/(abs(p)+100) if p<0 else 100/(p+100)

def remove_vig_two_way(p1, p2):
    s = p1 + p2
    if s == 0 or any(pd.isna([p1,p2])): return (np.nan, np.nan)
    return (p1/s, p2/s)

def kelly_fraction_from_prob(p: float, american_odds: float) -> float:
    if p is None or np.isnan(p): return 0.0
    b = american_to_decimal(american_odds) - 1.0
    q = 1 - p
    return max(0.0, (b*p - q) / b) if b > 0 else 0.0

def prob_cover(fair_line: float, book_line: float, sigma: float) -> float:
    z = (book_line - fair_line) / sigma
    return 0.5*(1 + erf(z/sqrt(2)))

def prob_over(fair_total: float, book_total: float, sigma: float) -> float:
    z = (fair_total - book_total) / sigma
    return 0.5*(1 + erf(z/sqrt(2)))

def key_number_cross(prev_point, new_point):
    try:
        a, b = abs(float(prev_point)), abs(float(new_point))
    except Exception:
        return False
    return any(min(a,b) < k <= max(a,b) for k in KEY_NUMBERS)

def best_flag_group(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    # "best" = most favorable absolute price (closest to even) as quick proxy
    g["_rank"] = g["price"].abs()
    idx = g["_rank"].idxmin()
    g["_best"] = False
    if pd.notna(idx):
        g.loc[idx, "_best"] = True
    return g

def label_compact(r: pd.Series) -> str:
    team, price, point, mkt = r.get("team"), r.get("price"), r.get("point"), r.get("market")
    if mkt and "totals" in mkt:
        pt = "" if pd.isna(point) else f"{point:g}"
        pr = "" if pd.isna(price) else f" ({int(price)})"
        return f"{str(team).upper()} {pt}{pr}"
    pt = "" if pd.isna(point) else f"{point:+g}"
    pr = "" if pd.isna(price) else f" ({int(price)})"
    return f"{team} {pt}{pr}"

def median_fair_price(df_gbt: pd.DataFrame) -> float:
    if df_gbt.empty: return np.nan
    return float(np.nanmedian(pd.to_numeric(df_gbt["price"], errors="coerce").values))

# =========================
# Import engines
# =========================
try:
    from bb_engine import project_game as bb_project
except Exception:
    def bb_project(date, home, away, level="NBA"):
        base_total = 227.0 if level=="NBA" else 142.0
        return 0.0, base_total

try:
    from nhl_engine import project_game_nhl
except Exception:
    def project_game_nhl(home, away, base_draw_rate=0.22):
        return {"p_home_ml":0.5,"p_away_ml":0.5,"p_home_reg":0.39,"p_draw_reg":0.22,"p_away_reg":0.39}

try:
    from mma_engine import fight_prob, default_prob
except Exception:
    def fight_prob(*args, **kwargs): return 0.5
    def default_prob(): return 0.5

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("Controls")

    sport = st.selectbox("Sport / League", list(SPORTS.keys()), index=0)
    sport_info = SPORTS[sport]

    default_markets = sport_info["default_markets"]
    all_markets = sorted(set(default_markets + [
        "spreads","totals","h2h",
        "spreads:1st_half","totals:1st_half",
        "spreads:1st_quarter","totals:1st_quarter",
        "h2h_3way"
    ]))
    markets = st.multiselect("Markets", options=all_markets, default=default_markets)

    regions = st.selectbox("Region", ["us","us2"], 0)
    odds_fmt = st.selectbox("Odds format", ["american","decimal"], 0)
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)

    books_preset = st.selectbox("Books preset", ["(All)","BetOnline only","DK only","FD only","DK+FD+BOL"], 0)
    query = st.text_input("Search team/matchup/fighter", "")

    st.divider()
    # Model/EV controls
    spread_sigma = st.slider("Spread Sigma (pts)", 6.0, 18.0, 13.0, 0.5)
    total_sigma  = st.slider("Total Sigma (pts)", 6.0, 18.0, 11.0, 0.5)
    draw_rate    = st.slider("NHL Draw Rate (reg)", 0.10, 0.30, 0.22, 0.01)
    kelly_scale  = st.slider("Kelly Scale", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.caption("API keys")
    st.success("The Odds API key loaded" if ODDS_KEY else "Add THE_ODDS_API_KEY in Secrets to fetch live odds.")
    if HAS_AI: st.success("OpenAI key loaded (AI write-ups/chat on)")
    else: st.info("AI write-ups/chat disabled (no OPENAI_API_KEY).")

# =========================
# Tabs
# =========================
tab_dashboard, tab_props, tab_writeups, tab_upload, tab_tracker, tab_chat = st.tabs(
    ["Top Value Bets","Props","Write-Ups","Upload Board","Bet Tracker","Chat"]
)

# Session storage
if "last_board" not in st.session_state:
    st.session_state["last_board"] = pd.DataFrame()
if "writeups" not in st.session_state:
    st.session_state["writeups"] = {}
if "bets" not in st.session_state:
    st.session_state["bets"] = []

# =========================================================
# TAB 1 — Top Value Bets (Live + movement + best price + EV/Kelly)
# =========================================================
st.info("Multi-sport odds, movement, fair numbers, EV/Kelly, props, write-ups, bet tracking, and chat.", icon="✨")
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
                for bk in game.get("bookmakers", []):
                    bk_key = (bk.get("key") or "").strip()
                    bk_name = KNOWN_BOOKS.get(bk_key, bk.get("title"))
                    for m in bk.get("markets", []):
                        mkey = m.get("key")  # e.g., spreads, totals, h2h, h2h_3way, spreads:1st_half ...
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
        st.info(f"Demo mode — no THE_ODDS_API_KEY.")
        rows = [
            {"sport":"NBA","game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"betonlineag","book_name":"BetOnline","market":"spreads","team":"LAKERS","price":-110,"point":-4.5,"home":"LAKERS","away":"SUNS"},
            {"sport":"NBA","game_id":"demo1","time":"2025-10-18T20:00:00Z","book_key":"draftkings","book_name":"DraftKings","market":"totals","team":"OVER","price":-105,"point":227.5,"home":"LAKERS","away":"SUNS"},
            {"sport":"NHL","game_id":"demo2","time":"2025-10-18T22:00:00Z","book_key":"fanduel","book_name":"FanDuel","market":"h2h_3way","team":"DRAW","price":+320,"point":np.nan,"home":"RANGERS","away":"BRUINS"},
        ]

    df = pd.DataFrame(rows)
    for c in ("price","point"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter by book preset
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
            | df["team"].fillna("").str.lower().str.contains(q)
        )
        df = df[mask]

    # Best price ⭐ per game/market/team
    if not df.empty:
        df = df.groupby(["game_id","market","team"], group_keys=False).apply(best_flag_group)
        df["_flag"] = np.where(df["_best"], "⭐ best", "")

    # Movement deltas vs. last snapshot
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

    # ----------------------------
    # Compute per-game "fair" numbers by sport
    # ----------------------------
    fairs = {}
    if not curr.empty:
        games = curr[["game_id","time","home","away"]].dropna().drop_duplicates()
        for _, g in games.iterrows():
            gid, t, home, away = g["game_id"], g["time"], str(g["home"]), str(g["away"])
            if sport in ("NBA","NCAAB"):
                level = "NBA" if sport=="NBA" else "NCAAB"
                fs, ft = bb_project(date=str(t)[:10], home=home, away=away, level=level)
                fairs[gid] = {"fair_spread_home": fs, "fair_total": ft}
            elif sport == "NHL":
                nhl = project_game_nhl(home, away, base_draw_rate=draw_rate)
                fairs[gid] = {**nhl}
            elif sport == "UFC":
                # Fallback 50/50 until ratings are supplied
                fairs[gid] = {"p_home_win": 0.5, "p_away_win": 0.5}
            else:
                fairs[gid] = {}
        fair_df = pd.DataFrame([{"game_id": k, **v} for k,v in fairs.items()])
        curr = curr.merge(fair_df, on="game_id", how="left")

    # ----------------------------
    # Per-row probability & Kelly
    # ----------------------------
    def row_prob(r):
        mk = r.get("market")
        price = r.get("price")
        point = r.get("point")

        if sport in ("NBA","NCAAB"):
            if mk and mk.startswith("spreads") and pd.notna(point):
                fair = r.get("fair_spread_home")
                if pd.isna(fair): return np.nan
                is_home_side = str(r.get("team","")).upper() == str(r.get("home","")).upper()
                p = prob_cover(fair, point, spread_sigma)
                return p if is_home_side else (1-p)
            if mk and mk.startswith("totals") and pd.notna(point):
                fairT = r.get("fair_total")
                if pd.isna(fairT): return np.nan
                if str(r.get("team","")).upper().startswith("OVER"):
                    return prob_over(fairT, point, total_sigma)
                else:
                    return 1 - prob_over(fairT, point, total_sigma)
            if mk == "h2h":
                fair = r.get("fair_spread_home")
                if pd.isna(fair): return np.nan
                p_home = 0.5*(1 + erf((-fair)/ (spread_sigma*sqrt(2))))
                is_home_ml = str(r.get("team","")).upper() == str(r.get("home","")).upper()
                return p_home if is_home_ml else 1-p_home

        if sport == "NHL":
            if mk == "h2h":
                is_home_ml = str(r.get("team","")).upper() == str(r.get("home","")).upper()
                p_home = r.get("p_home_ml")
                if pd.isna(p_home): return np.nan
                return p_home if is_home_ml else 1-p_home
            if mk == "h2h_3way":
                name = str(r.get("team","")).upper()
                if "DRAW" in name or "TIE" in name:
                    return r.get("p_draw_reg")
                is_home = name == str(r.get("home","")).upper()
                return r.get("p_home_reg") if is_home else r.get("p_away_reg")
            if mk and mk.startswith("totals") and pd.notna(point):
                fairT = 6.2  # simple baseline; replace with xG total model
                if str(r.get("team","")).upper().startswith("OVER"):
                    return prob_over(fairT, point, total_sigma)
                else:
                    return 1 - prob_over(fairT, point, total_sigma)

        if sport == "UFC":
            if mk == "h2h":
                is_home = str(r.get("team","")).upper() == str(r.get("home","")).upper()
                p_home = r.get("p_home_win")
                if pd.isna(p_home): return np.nan
                return p_home if is_home else 1 - p_home

        return np.nan

    def row_kelly(r):
        p = r.get("_p"); odds = r.get("price")
        if pd.isna(p) or pd.isna(odds): return 0.0
        return kelly_fraction_from_prob(float(p), float(odds)) * kelly_scale

    if not curr.empty:
        curr["_p"] = curr.apply(row_prob, axis=1)
        curr["_kelly"] = curr.apply(row_kelly, axis=1).clip(lower=0)
        curr = curr.sort_values(["_kelly","_crossed_key","_steam","_move_score"], ascending=[False,False,False,False])

    # Consensus-based edge (median book price per outcome -> cents vs current)
    if not curr.empty:
        edges=[]
        for (gid,mkt,tm), g in curr.groupby(["game_id","market","team"]):
            fair_price = median_fair_price(g)
            for _, r in g.iterrows():
                edge_cents = np.nan
                if pd.notna(fair_price) and pd.notna(r["price"]):
                    edge_cents = int(r["price"] - fair_price)
                edges.append({**r.to_dict(), "fair_price_consensus": fair_price, "edge_cents_vs_consensus": edge_cents})
        curr = pd.DataFrame(edges)

    st.caption("Badges: ⭐ Best price · 🔥 Steam move · 🎯 Crossed key number (spreads only)")
    st.dataframe(
        curr[[
            "time","home","away","book_name","market","team","point","price",
            "fair_price_consensus","edge_cents_vs_consensus","_p","_kelly",
            "_move","_flag","_steam","_crossed_key"
        ]].rename(columns={"_p":"prob","_kelly":"kelly"}),
        use_container_width=True, hide_index=True
    )

    # Compact view + CSV
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

    # Auto refresh
    if auto_refresh:
        if "last_tick" not in st.session_state: st.session_state["last_tick"] = 0.0
        if time.time() - st.session_state["last_tick"] >= 60:
            st.session_state["last_tick"] = time.time()
            st.experimental_rerun()

# =================================
# TAB 2 — Props (simple no-vig view; extend per sport feed)
# =================================
with tab_props:
    st.subheader("Player Props — No-Vig Splits & Edges")
    st.caption("Select prop markets supported by your feed. (Feeds vary by sport.)")

    # Example football props keys — update if your feed supports hoops props: player_points, player_rebounds, etc.
    supported = ["player_pass_yds","player_rush_yds","player_rec_yds","player_pass_tds","player_receptions"]
    want = st.multiselect("Prop markets", supported, default=["player_pass_yds","player_rush_yds"])

    props_df = pd.DataFrame()
    if ODDS_KEY and want:
        try:
            with st.spinner("Fetching props…"):
                data_props = fetch_odds_for_sport(SPORTS.get("CFB","CFB")["odds_key"], ",".join(want), regions, odds_fmt, ODDS_KEY) if sport=="CFB" else fetch_odds_for_sport(SPORTS[sport]["odds_key"], ",".join(want), regions, odds_fmt, ODDS_KEY)
            rows=[]
            for game in data_props:
                gid = game.get("id"); t = game.get("commence_time")
                home, away = game.get("home_team"), game.get("away_team")
                for bk in game.get("bookmakers", []):
                    bk_key = (bk.get("key") or "").strip()
                    bk_name = KNOWN_BOOKS.get(bk_key, bk.get("title"))
                    for m in bk.get("markets", []):
                        mkey = m.get("key")
                        for out in m.get("outcomes", []):
                            rows.append({
                                "sport": sport, "game_id": gid, "time": t, "book_key": bk_key, "book_name": bk_name,
                                "market": mkey, "participant": out.get("name"),
                                "price": out.get("price"), "point": out.get("point"),
                                "home": home, "away": away
                            })
            props_df = pd.DataFrame(rows)
        except requests.RequestException as e:
            st.error(f"Props fetch failed: {e}")
    else:
        st.info("Add THE_ODDS_API_KEY in Secrets and select one or more prop markets.")

    if not props_df.empty:
        props_df["price"] = pd.to_numeric(props_df["price"], errors="coerce")
        props_df["point"] = pd.to_numeric(props_df["point"], errors="coerce")
        props_df["implied"] = props_df["price"].apply(american_to_implied)
        st.dataframe(
            props_df[["time","home","away","book_name","market","participant","point","price","implied"]],
            use_container_width=True, hide_index=True
        )
        st.download_button("Download props CSV", props_df.to_csv(index=False).encode("utf-8"), "props.csv", "text/csv")

# =================================
# TAB 3 — Write-Ups (manual + AI)
# =================================
with tab_writeups:
    st.subheader("Game / Event Write-Ups")
    st.caption("Keep notes or click Generate to draft an AI write-up for the selected sport.")

    style = {"CFB":"college football","NCAAB":"college basketball","NBA":"NBA basketball","NHL":"NHL hockey","UFC":"UFC mixed martial arts"}[sport]
    game_title = st.text_input(f"{style.title()} Matchup (e.g., LSU @ Florida / Oilers @ Stars / Gaethje vs Poirier)")

    colA, colB = st.columns([3,1])
    with colA:
        text = st.text_area("Write-up", st.session_state["writeups"].get((sport,game_title), ""), height=260)
    with colB:
        if HAS_AI and st.button("Generate with AI"):
            prompt = (
                f"Write a concise, data-aware betting analysis for a {style} matchup: {game_title}. "
                f"Include sections: EdgeLine Model Lean; Why it matters (3 bullets with stats or matchup edges); "
                f"Risk flags (tempo/injuries/travel/key numbers as relevant); "
                f"Suggested stake guidance (Kelly-lite). Keep it sharp and readable."
            )
            try:
                resp = oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"You are EdgeLine's multi-sport betting analyst."},
                              {"role":"user","content":prompt}],
                    temperature=0.7,
                    max_tokens=320
                )
                text = resp.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"AI error: {e}")
        if st.button("Save"):
            if game_title:
                st.session_state["writeups"][(sport,game_title)] = text
                st.success("Saved.")

    if st.session_state["writeups"]:
        st.markdown("#### Saved Write-Ups")
        for (sp, k), v in list(st.session_state["writeups"].items()):
            if sp != sport: continue
            with st.expander(k):
                st.write(v)

# =================================
# TAB 4 — Upload Board (CSV/TSV preview)
# =================================
with tab_upload:
    st.subheader("Upload Board (CSV/TSV)")
    up = st.file_uploader("Upload a board (BetOnline or compatible export)", type=["csv","tsv","txt"])
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
# TAB 5 — Bet Tracker
# =================================
with tab_tracker:
    st.subheader("Bet Tracker")
    c0,c1,c2,c3,c4,c5,c6,c7 = st.columns([1,1.6,1,1,1,1,1,1])
    with c0:
        bt_sport = st.selectbox("Sport", list(SPORTS.keys()), index=list(SPORTS.keys()).index(sport))
    with c1:
        gm = st.text_input("Event (e.g., LSU @ Florida / Oilers @ Stars / Gaethje vs Poirier)")
    with c2:
        market = st.selectbox("Market", ["FT Spread","FT Total","1H Spread","1H Total","1Q Spread","1Q Total","ML","3-Way ML"])
    with c3:
        period = st.text_input("Period (e.g., 1H / 1Q / Reg / OT incl.)", value="")
    with c4:
        pick = st.text_input("Pick (e.g., LSU -3.5 / OVER 55.5 / Oilers Reg / Gaethje ML)")
    with c5:
        odds = st.number_input("Odds (American)", value=-110, step=1)
    with c6:
        stake = st.number_input("Stake ($)", value=50.0, step=5.0)
    with c7:
        cl = st.text_input("Closing line (optional)", value="")
    if st.button("Add bet"):
        if gm and pick:
            st.session_state["bets"].append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sport": bt_sport,
                "event": gm, "market": market, "period": period,
                "pick": pick, "odds": int(odds), "stake": float(stake),
                "closing_line": cl, "result":"open", "payout":0.0
            })
            st.success("Bet added.")
        else:
            st.warning("Enter Event and Pick.")

    if st.session_state["bets"]:
        bt = pd.DataFrame(st.session_state["bets"])
        st.dataframe(bt, use_container_width=True, hide_index=True)
        st.download_button("Download tracker CSV", bt.to_csv(index=False).encode("utf-8"), "bet_tracker.csv", "text/csv")

# =================================
# TAB 6 — In-App Chat
# =================================
with tab_chat:
    st.subheader("EdgeLine Chat (multi-sport)")
    q = st.text_input("Ask anything (e.g., What's my edge on LSU vs Florida 1H? Best NHL 3-way spots?)")

    if HAS_AI and st.button("Ask"):
        sys = (
            "You are EdgeLine, an analyst that explains betting edges with clarity. "
            "Use concise bullets and practical guidance. When uncertain, state what would resolve it."
        )
        user = f"Sport: {sport}. Question: {q}. Consider live prices, key numbers (if applicable), and risk flags."
        try:
            r = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=0.5, max_tokens=300
            )
            st.write(r.choices[0].message.content)
        except Exception as e:
            st.error(f"AI error: {e}")
    elif not HAS_AI:
        st.info("Add OPENAI_API_KEY to enable chat.")
