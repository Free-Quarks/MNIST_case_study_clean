import numpy as np


def seirhd_model(y, t, beta, sigma, gamma, delta):
    S, E, I, R, H, D = y
    N = S + E + I + R + H + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + delta) * I
    dRdt = gamma * I
    dHdt = delta * I
    dDdt = delta * I
    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt


def rk3_step(y, t, dt, model_function, *args):
    k1 = model_function(y, t, *args)
    k2 = model_function(y + dt/2 * k1, t + dt/2, *args)
    k3 = model_function(y - dt * k1 + 2 * dt * k2, t + dt, *args)
    y_next = y + dt/6 * (k1 + 4 * k2 + k3)
    return y_next


def simulate_seirhd_model(initial_conditions, beta, sigma, gamma, delta, dt, n_steps):
    t = np.arange(n_steps) * dt
    y = np.zeros((n_steps, len(initial_conditions)))
    y[0] = initial_conditions
    for i in range(1, n_steps):
        y[i] = rk3_step(y[i-1], t[i-1], dt, seirhd_model, beta, sigma, gamma, delta)
    return y


initial_conditions = [1000, 1, 0, 0, 0, 0]
beta = 0.2
sigma = 0.5
gamma = 0.1
delta = 0.05
dt = 0.01
n_steps = 1000

y = simulate_seirhd_model(initial_conditions, beta, sigma, gamma, delta, dt, n_steps)

