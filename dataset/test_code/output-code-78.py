import numpy as np

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dydt = np.zeros(3)
    dydt[0] = -beta * S * I
    dydt[1] = beta * S * I - gamma * I
    dydt[2] = gamma * I
    return dydt


def rk3_step(t, y, h, beta, gamma):
    k1 = h * sir_model(t, y, beta, gamma)
    k2 = h * sir_model(t + 0.5 * h, y + 0.5 * k1, beta, gamma)
    k3 = h * sir_model(t + h, y - k1 + 2 * k2, beta, gamma)
    return y + (1 / 6) * (k1 + 4 * k2 + k3)


def simulate_sir_model(S0, I0, R0, beta, gamma, num_steps, h):
    t = np.linspace(0, num_steps * h, num_steps + 1)
    y = np.zeros((num_steps + 1, 3))
    y[0] = S0, I0, R0

    for i in range(num_steps):
        y[i+1] = rk3_step(t[i], y[i], h, beta, gamma)

    return t, y[:, 0], y[:, 1], y[:, 2]
