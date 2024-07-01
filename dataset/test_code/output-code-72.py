import numpy as np


def seirhd_model(y, t, beta, gamma, delta, alpha, rho):
    S, E, I, R, H, D = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dHdt = alpha * rho * I
    dDdt = alpha * gamma * I
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]


def simulate_seirhd_model(S0, E0, I0, R0, H0, D0, beta, gamma, delta, alpha, rho, t):
    y0 = [S0, E0, I0, R0, H0, D0]
    params = (beta, gamma, delta, alpha, rho)
    result = odeint(seirhd_model, y0, t, args=params)
    return result[:, 0], result[:, 1], result[:, 2], result[:, 3], result[:, 4], result[:, 5]
