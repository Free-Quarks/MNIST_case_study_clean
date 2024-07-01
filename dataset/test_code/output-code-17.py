import numpy as np


def sidarthe_model(y, t, beta, gamma, mu, delta, alpha, rho, sigma):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + delta * A) / N
    dIdt = beta * S * (I + delta * A) / N - (gamma + mu) * I
    dDdt = mu * I
    dAdt = delta * (gamma * I - alpha * A)
    dRdt = gamma * I + alpha * A
    dTdt = rho * mu * I
    dHdt = sigma * (gamma * I - alpha * A)
    dEdt = rho * mu * I
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def rk2_step(y, t, dt, beta, gamma, mu, delta, alpha, rho, sigma):
    k1 = dt * sidarthe_model(y, t, beta, gamma, mu, delta, alpha, rho, sigma)
    k2 = dt * sidarthe_model(y + 0.5 * k1, t + 0.5 * dt, beta, gamma, mu, delta, alpha, rho, sigma)
    y_new = y + k2
    return y_new


def sidarthe_rk2(t0, y0, dt, beta, gamma, mu, delta, alpha, rho, sigma):
    t = np.arange(t0, t0 + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = rk2_step(y[i-1], t[i-1], dt, beta, gamma, mu, delta, alpha, rho, sigma)
    return t, y

