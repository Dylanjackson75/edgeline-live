# mma_engine.py
import numpy as np

def fight_prob(thetaA:float, thetaB:float, reachA:float, reachB:float, ageA:float, ageB:float,
               stanceA:str="O", stanceB:str="O") -> float:
    """Simple logistic model: skill gap + reach + age + stance advantage."""
    x = (thetaA - thetaB)/100.0 + 0.02*(reachA - reachB) + 0.01*(ageB - ageA)
    if stanceA != stanceB:
        x += 0.05
    return float(1.0/(1.0 + np.exp(-x)))

def default_prob():
    return 0.5
