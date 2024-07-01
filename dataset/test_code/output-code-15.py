import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(t, y, beta, sigma, tau, theta, p, q, r, s, d, gamma, xi):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dS_dt = -beta * (I + theta * A) * S / N
    dI_dt = beta * (I + theta * A) * S / N - sigma * I - tau * I - p * xi * I
    dD_dt = p * (1 - xi) * I
    dA_dt = sigma * I - (q + gamma + d) * A
    dR_dt = gamma * A
    dT_dt = tau * I - (r + s) * T
    dH_dt = q * A + r * T
    dE_dt = s * T
    return [dS_dt, dI_dt, dD_dt, dA_dt, dR_dt, dT_dt, dH_dt, dE_dt]


def simulate_sidarthe_model(S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0, beta, sigma, tau, theta, p, q, r, s, d, gamma, xi, t_end, t_step):
    t = np.arange(0, t_end, t_step)
    y_0 = [S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0]
    y = np.zeros((len(t), len(y_0)))
    y[0] = y_0
    for i in range(1, len(t)):
        dy_dt = sidarthe_model(t[i-1], y[i-1], beta, sigma, tau, theta, p, q, r, s, d, gamma, xi)
        y[i] = y[i-1] + t_step * dy_dt
    return t, y
