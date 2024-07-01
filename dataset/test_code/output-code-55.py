import numpy as np
import matplotlib.pyplot as plt

def euler(f, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        y[i+1] = y[i] + h*f(t[i], y[i])
    return y


def seir_model(t, y):
    S, E, I, R = y
    N = S + E + I + R
    beta = 0.4
    gamma = 0.1
    sigma = 0.2
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


# Example usage

# Define time points
t = np.linspace(0, 100, 1000)

# Initial conditions
y0 = [1000, 1, 0, 0]

# Solve SEIR model
y = euler(seir_model, y0, t)

# Plotting
plt.plot(t, y[:, 0], label='S')
plt.plot(t, y[:, 1], label='E')
plt.plot(t, y[:, 2], label='I')
plt.plot(t, y[:, 3], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
