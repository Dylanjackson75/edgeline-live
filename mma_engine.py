# mma_engine.py
import numpy as np

def fight_prob(thetaA:float, thetaB:float, reachA:float, reachB:float, ageA:float, ageB:float,
               stanceA:str="O", stanceB:str="O") -> float:
    """
    Returns p(A beats B). theta* are overall ratings (Elo/BT). Provide 1500 baseline if unknown.
    Simple physical priors: reach + age; southpaw slight adj.
    """
    x = (thetaA - thetaB)/100.0 + 0.02*(reachA - reachB) + 0.01*(ageB - ageA)
    if stanceA != stanceB:
        x += 0.05
    return float(1.0/(1.0 + np.exp(-x)))

def default_prob():  # fallback 50/50
    return 0.5
