import numpy as np
from scipy.integrate import odeint


def seird_model(y, t, N, beta, gamma, sigma, mu):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def run_seird_model(N, E0, I0, R0, D0, beta, gamma, sigma, mu, days):
    # Initial conditions vector
    y0 = N - E0 - I0 - R0 - D0
    # A grid of time points (in days)
    t = np.linspace(0, days, days)
    # Integrate the SEIRD equations over the time grid, t.
    ret = odeint(seird_model, y0, t, args=(N, beta, gamma, sigma, mu))
    S, E, I, R, D = ret.T
    return S, E, I, R, D
