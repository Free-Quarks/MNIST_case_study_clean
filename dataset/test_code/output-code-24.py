import numpy as np


def SEIRHD_model(t, y, N, beta, gamma, sigma, mu, alpha):
    S, E, I, R, H, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I + alpha * H
    dHdt = mu * I - alpha * H
    dDdt = alpha * H
    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt



def RK4_integration(y0, t0, t1, dt, N, beta, gamma, sigma, mu, alpha):
    t = np.arange(t0, t1+dt, dt)
    n = len(t)
    y = np.zeros((n, 6))
    y[0] = y0
    for i in range(n-1):
        k1 = dt * SEIRHD_model(t[i], y[i], N, beta, gamma, sigma, mu, alpha)
        k2 = dt * SEIRHD_model(t[i] + dt/2, y[i] + k1/2, N, beta, gamma, sigma, mu, alpha)
        k3 = dt * SEIRHD_model(t[i] + dt/2, y[i] + k2/2, N, beta, gamma, sigma, mu, alpha)
        k4 = dt * SEIRHD_model(t[i] + dt, y[i] + k3, N, beta, gamma, sigma, mu, alpha)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y
