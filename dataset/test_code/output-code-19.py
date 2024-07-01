import numpy as np


def sidarthe_model(t, y, p):
    S, I, D, A, R, T, H, E = y
    beta, gamma, mu, alpha, delta, rho, eta, theta = p
    N = S + I + D + A + R + T + H + E
    
    dSdt = -beta * S * (I + alpha * A) / N
    dIdt = beta * S * (I + alpha * A) / N - (gamma + mu) * I
    dDdt = delta * rho * I - (eta + mu) * D
    dAdt = (1 - delta) * rho * I - (theta + mu) * A
    dRdt = gamma * I - mu * R
    dTdt = eta * D - mu * T
    dHdt = theta * A - mu * H
    dEdt = mu * (I + D + A + R + T + H) - mu * E
    
    return np.array([dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt])

