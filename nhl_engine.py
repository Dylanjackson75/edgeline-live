# nhl_engine.py
import numpy as np

def sigmoid(x): 
    return 1.0/(1.0+np.exp(-x))

def project_game_nhl(home:str, away:str, goalie_adj_home:float=0.0, goalie_adj_away:float=0.0,
                     base_draw_rate:float=0.22) -> dict:
    """Estimate probabilities for NHL home/away/draw using a logistic model."""
    theta_home = 0.0 + goalie_adj_home
    theta_away = 0.0 + goalie_adj_away
    edge = theta_home - theta_away
    p_home_reg = sigmoid(edge) * (1.0 - base_draw_rate)
    p_away_reg = (1.0 - base_draw_rate) - p_home_reg
    p_draw_reg = base_draw_rate
    p_home_ml = p_home_reg + 0.5*p_draw_reg
    p_away_ml = p_away_reg + 0.5*p_draw_reg
    return {
        "p_home_ml": float(p_home_ml),
        "p_away_ml": float(p_away_ml),
        "p_home_reg": float(p_home_reg),
        "p_draw_reg": float(p_draw_reg),
        "p_away_reg": float(p_away_reg),
    }
