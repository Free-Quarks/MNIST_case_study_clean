import numpy as np

def seir_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_seir_model(beta, sigma, gamma, S0, E0, I0, R0, t_max, nt):
    t = np.linspace(0, t_max, nt)
    y0 = S0, E0, I0, R0
    y = np.zeros((nt, 4))
    y[0] = y0
    dt = t[1] - t[0]

    for i in range(nt-1):
        k1 = seir_model(t[i], y[i], beta, sigma, gamma)
        k2 = seir_model(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k1, beta, sigma, gamma)
        k3 = seir_model(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k2, beta, sigma, gamma)
        k4 = seir_model(t[i] + dt, y[i] + dt * k3, beta, sigma, gamma)
        y[i+1] = y[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y

