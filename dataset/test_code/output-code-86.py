import numpy as np
from scipy.integrate import odeint


def seird_model(y, t, params):
    S, E, I, R, D = y
    beta, sigma, gamma, mu = params
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def run_seird_model(S0, E0, I0, R0, D0, beta, sigma, gamma, mu, days):
    N = S0 + E0 + I0 + R0 + D0
    y0 = [S0, E0, I0, R0, D0]
    params = [beta, sigma, gamma, mu]
    t = np.linspace(0, days, days)
    
    solution = odeint(seird_model, y0, t, args=(params,))
    S, E, I, R, D = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3], solution[:, 4]
    
    return S, E, I, R, D
