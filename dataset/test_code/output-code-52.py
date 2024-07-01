import numpy as np

def SIR_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def RK2_solver(model, t0, y0, t_end, h, *args):
    t = np.arange(t0, t_end+h, h)
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0

    for i in range(len(t)-1):
        k1 = model(t[i], y[i, :], *args)
        k2 = model(t[i] + h/2, y[i, :] + h/2 * k1, *args)
        y[i+1, :] = y[i, :] + h * k2

    return t, y


# Example usage
beta = 0.5
gamma = 0.1

# Initial conditions
S0 = 990
I0 = 10
R0 = 0
y0 = [S0, I0, R0]

# Time parameters
t0 = 0
t_end = 100
h = 0.1

# Solve the SIR model using RK2
t, y = RK2_solver(SIR_model, t0, y0, t_end, h, beta, gamma)

