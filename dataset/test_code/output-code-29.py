import numpy as np


def SIR_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])


def RK4_solver(model, y0, t, args):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = model(t[i], y[i], *args)
        k2 = model(t[i] + 0.5 * h, y[i] + 0.5 * h * k1, *args)
        k3 = model(t[i] + 0.5 * h, y[i] + 0.5 * h * k2, *args)
        k4 = model(t[i] + h, y[i] + h * k3, *args)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# Example usage
beta = 0.2
gamma = 0.1
y0 = np.array([0.99, 0.01, 0.0])
t = np.linspace(0, 100, 1000)

y = RK4_solver(SIR_model, y0, t, (beta, gamma))

print(y)
