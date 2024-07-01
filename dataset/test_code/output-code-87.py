import numpy as np
import matplotlib.pyplot as plt

def SEIRD_model(t, y, beta, gamma, delta, alpha):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - (gamma + alpha) * I
    dRdt = gamma * I
    dDdt = alpha * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def simulate_SEIRD_model(S0, E0, I0, R0, D0, beta, gamma, delta, alpha, t_max, dt):
    t = np.arange(0, t_max, dt)
    y0 = [S0, E0, I0, R0, D0]

    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = SEIRD_model(t[i-1], y[i-1], beta, gamma, delta, alpha)
        k2 = SEIRD_model(t[i-1] + dt/2, y[i-1] + (dt/2)*k1, beta, gamma, delta, alpha)
        y[i] = y[i-1] + dt*k2

    S, E, I, R, D = y.T

    return t, S, E, I, R, D
