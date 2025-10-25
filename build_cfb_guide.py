import pandas as pd
from cfb_engine import score_games

def make_sections(df):
    edge_df = df[df["label"] == "EDGE PLAY"]
    lean_df = df[df["label"] == "LEAN"]
    pass_df = df[df["label"] == "PASS"]

    guide_text = []

    guide_text.append("SECTION 1. EDGE PLAYS (Hammer These)")
    for _, row in edge_df.iterrows():
        guide_text.append(
            f"- {row['home_team']} {row['market_spread_home']} vs {row['away_team']} "
            f"(Our line {row['our_spread_home']:.1f}, edge {row['spread_edge_pts']:.1f} pts)"
        )

    guide_text.append("\nSECTION 2. LEANS (Smaller Unit / Parlay Glue)")
    for _, row in lean_df.iterrows():
        guide_text.append(
            f"- {row['home_team']} {row['market_spread_home']} vs {row['away_team']} "
            f"(Our line {row['our_spread_home']:.1f}, edge {row['spread_edge_pts']:.1f} pts)"
        )

    guide_text.append("\nSECTION 3. PASS / STAY DISCIPLINED")
    for _, row in pass_df.iterrows():
        guide_text.append(
            f"- {row['away_team']} @ {row['home_team']} (No playable edge)"
        )

    full = "\n".join(guide_text)
    return full

if __name__ == "__main__":
    df = score_games()
    guide = make_sections(df)

    # save text for PDF / socials / email blast
    with open("cfb_week_guide.txt", "w") as f:
        f.write(guide)

    print("\n=== EDGE LINE WEEKLY COLLEGE FOOTBALL GUIDE ===\n")
    print(guide)
    print("\nSaved cfb_week_guide.txt")
