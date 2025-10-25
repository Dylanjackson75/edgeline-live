import requests
import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt

# =============================
# CONFIG
# =============================

ODDS_API_KEY = "YOUR_ODDS_API_KEY_HERE"  # <-- put your live odds API key
CFBD_API_KEY = "YOUR_CFBD_API_KEY_HERE"  # <-- put your collegefootballdata.com key
SPORT_KEY = "americanfootball_ncaaf"     # Odds API sport code for college football
BASE_HFA = 2.0                           # base home field edge in points

# Home stadium lat/lon + altitude flag for situational edge.
# Add more teams over time. If you don't know a team, leave it out;
# they'll just get no travel calc / no altitude bump.
STADIUMS = {
    "Alabama":         {"lat": 33.2083, "lon": -87.5504, "altitude_flag": 0},
    "South Carolina":  {"lat": 34.0007, "lon": -81.0348, "altitude_flag": 0},
    "Washington":      {"lat": 47.6500, "lon": -122.3000, "altitude_flag": 0},
    "Illinois":        {"lat": 40.1020, "lon": -88.2272, "altitude_flag": 0},
    "Virginia Tech":   {"lat": 37.2219, "lon": -80.4189, "altitude_flag": 0},
    "California":      {"lat": 37.8715, "lon": -122.2730, "altitude_flag": 0},
    # Add LSU, Texas A&M, Michigan, etc.
}

# =============================
# HELPERS
# =============================

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3959  # Earth radius in miles
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = (
        np.sin(dlat/2.0)**2
        + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

def travel_tax(miles):
    # Adds to home edge. Cross-country = more edge.
    # Caps at +3.
    if pd.isna(miles):
        return 0.0
    return min((miles / 1000.0) * 0.8, 3.0)

def fetch_live_odds():
    """
    Pulls current spreads/totals for all NCAAF games from the Odds API.
    Assumes you already used this service in your repo for MMA/NHL, etc.
    We just point it at SPORT_KEY.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "apiKey": ODDS_API_KEY
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    rows = []
    for g in data:
        home = g["home_team"]
        away = g["away_team"]
        kickoff = g["commence_time"]  # ISO timestamp

        spread_pts = None
        total_pts = None

        # We'll just take the first bookmaker (you can average later)
        if g.get("bookmakers"):
            first_book = g["bookmakers"][0]["markets"]
            for m in first_book:
                if m["key"] == "spreads":
                    # Find home team line
                    for o in m["outcomes"]:
                        if o["name"] == home:
                            # e.g. -11.5 for Alabama
                            spread_pts = float(o["point"])
                if m["key"] == "totals":
                    # Just grab the number (same for both sides)
                    total_pts = float(m["outcomes"][0]["point"])

        rows.append({
            "home_team": home,
            "away_team": away,
            "game_datetime": kickoff,
            "market_spread_home": spread_pts,  # home negative = home favored
            "market_total": total_pts
        })

    return pd.DataFrame(rows)

def fetch_team_metrics():
    """
    Pull team efficiency from collegefootballdata.com:
    - off_epa (offense efficiency per play)
    - def_epa (defense efficiency allowed)
    - qb_grade (passing EPA proxy for QB quality)
    Also attach stadium lat/lon and altitude flag.
    """
    url = "https://api.collegefootballdata.com/metrics/ppa"
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()

    rows = []
    for t in data:
        team = t["team"]
        off_epa = t.get("offense",{}).get("overall",{}).get("ppa")
        def_epa = t.get("defense",{}).get("overall",{}).get("ppa")
        qb_eff = t.get("offense",{}).get("passing",{}).get("ppa")  # QB proxy
        stadium = STADIUMS.get(team, {})
        rows.append({
            "team": team,
            "off_epa": off_epa,
            "def_epa": def_epa,
            "qb_grade": qb_eff,
            "lat": stadium.get("lat"),
            "lon": stadium.get("lon"),
            "altitude_flag": stadium.get("altitude_flag", 0),
        })
    return pd.DataFrame(rows)

def build_feature_table(odds_df, metrics_df):
    """
    Combine odds with team metrics.
    Compute travel miles (away -> home).
    Compute situational HFA (base HFA + travel tax + altitude).
    """
    df = odds_df.merge(
        metrics_df.add_prefix("home_"),
        left_on="home_team",
        right_on="home_team",
        how="left"
    ).merge(
        metrics_df.add_prefix("away_"),
        left_on="away_team",
        right_on="away_team",
        how="left"
    )

    # travel miles calculation
    df["travel_miles"] = df.apply(
        lambda r: haversine_miles(
            r["away_lat"], r["away_lon"],
            r["home_lat"], r["home_lon"]
        ) if pd.notna(r["away_lat"]) and pd.notna(r["home_lat"]) else np.nan,
        axis=1
    )

    df["situational_hfa"] = (
        BASE_HFA
        + df["travel_miles"].apply(travel_tax)
        + df["home_altitude_flag"].fillna(0) * 1.0
    )

    return df

def model_scores(df):
    """
    This is the “engine”.
    It creates our spread for the home team, compares it to the book,
    and classifies the edge.
    """

    # Our internal number for home team spread:
    #  - home offense vs away defense
    #  - away offense vs home defense (subtract)
    #  - QB gap (weighted)
    #  - situational HFA (rest/travel/altitude baked in)
    df["our_spread_home"] = (
        (df["home_off_epa"] - df["away_def_epa"]) -
        (df["away_off_epa"] - df["home_def_epa"]) +
        (df["home_qb_grade"] - df["away_qb_grade"]) * 2.0 +
        df["situational_hfa"]
    )

    # Edge in points vs the market line
    df["spread_edge_pts"] = df["our_spread_home"] - df["market_spread_home"]

    # classify confidence band
    def classify(edge):
        e = abs(edge)
        if e >= 4:
            return "EDGE PLAY"
        if e >= 2:
            return "LEAN"
        return "PASS"

    df["label"] = df["spread_edge_pts"].apply(classify)

    # final view for guide
    picks = df[[
        "away_team","home_team","game_datetime",
        "market_spread_home","our_spread_home","spread_edge_pts",
        "label"
    ]].sort_values(by="spread_edge_pts", ascending=False)

    return picks

def score_games():
    """
    Public function we call from app.py or CLI.
    Returns a DataFrame with EDGE PLAY / LEAN labels.
    """
    odds_df = fetch_live_odds()
    metrics_df = fetch_team_metrics()
    features = build_feature_table(odds_df, metrics_df)
    picks = model_scores(features)
    return picks

if __name__ == "__main__":
    out = score_games()
    print("=== EDGE PLAYS / LEANS ===")
    print(out[out["label"] != "PASS"].to_string(index=False))
