import numpy as np

def SIR_model(x, t, beta, gamma):
    S, I, R = x
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def RK4_step(x, t, dt, beta, gamma):
    k1 = dt * SIR_model(x, t, beta, gamma)
    k2 = dt * SIR_model(x + 0.5 * k1, t + 0.5 * dt, beta, gamma)
    k3 = dt * SIR_model(x + 0.5 * k2, t + 0.5 * dt, beta, gamma)
    k4 = dt * SIR_model(x + k3, t + dt, beta, gamma)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t_new = t + dt
    return x_new, t_new

def simulate_SIR_model(S0, I0, R0, beta, gamma, T, dt):
    t = np.arange(0, T+dt, dt)
    x = np.zeros((len(t), 3))
    x[0] = S0, I0, R0
    for i in range(1, len(t)):
        x[i], t[i] = RK4_step(x[i-1], t[i-1], dt, beta, gamma)
    return x[:, 0], x[:, 1], x[:, 2]
