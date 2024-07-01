import numpy as np
from scipy.integrate import odeint


def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):
    S0 = N - I0 - E0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, E0, I0, R0
    args = (beta, gamma, sigma)
    solution = odeint(seir_model, y0, t, args)
    S, E, I, R = solution.T
    return S, E, I, R
