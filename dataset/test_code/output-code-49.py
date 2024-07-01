import numpy as np


def seirhd_model(t, y, params):
    S, E, I, R, H, D = y
    beta, sigma, gamma, delta, alpha = params
    N = S + E + I + R + H + D
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I - delta * I - alpha * I
    dR = gamma * I
    dH = delta * I
    dD = alpha * I
    return np.array([dS, dE, dI, dR, dH, dD])


def run_seirhd_model(initial_conditions, params, t_max, step_size):
    t = np.arange(0, t_max + step_size, step_size)
    result = np.zeros((len(t), len(initial_conditions)))
    result[0] = initial_conditions

    for i in range(1, len(t)):
        result[i] = rk4_step(seirhd_model, result[i-1], params, t[i-1], step_size)

    return result
