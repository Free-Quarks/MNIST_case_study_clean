import numpy as np


def seirhd_model(t, y, N, beta, sigma, gamma, delta, alpha):
    S, E, I, R, H, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (1 - alpha) * gamma * I - alpha * delta * I
    dRdt = (1 - alpha) * gamma * I
    dHdt = alpha * delta * I
    dDdt = alpha * gamma * I
    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt


def run_seirhd_model(N, initial_conditions, parameters, t):
    beta, sigma, gamma, delta, alpha = parameters
    S0, E0, I0, R0, H0, D0 = initial_conditions
    y0 = S0, E0, I0, R0, H0, D0

    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        tspan = [t[i - 1], t[i]]
        res = odeint(seirhd_model, y0, tspan, args=(N, beta, sigma, gamma, delta, alpha))
        y[i] = res[1]
        y0 = y[i]

    S, E, I, R, H, D = y.T

    return S, E, I, R, H, D
