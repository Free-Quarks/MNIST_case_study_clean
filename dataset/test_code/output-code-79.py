import numpy as np
import matplotlib.pyplot as plt

# Function that defines the dynamics of the SIR model
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# RK4 method

def rk4(t0, tf, y0, h, beta, gamma):
    t = np.arange(t0, tf + h, h)
    n = len(t)
    y = np.zeros((n, 3))
    y[0] = y0
    for i in range(n - 1):
        k1 = h * np.array(sir_model(t[i], y[i], beta, gamma))
        k2 = h * np.array(sir_model(t[i] + h / 2, y[i] + k1 / 2, beta, gamma))
        k3 = h * np.array(sir_model(t[i] + h / 2, y[i] + k2 / 2, beta, gamma))
        k4 = h * np.array(sir_model(t[i] + h, y[i] + k3, beta, gamma))
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, y

# Parameters
beta = 0.2
gamma = 0.1

# Initial conditions
y0 = [0.99, 0.01, 0]

# Solve the SIR model using RK4
t, y = rk4(0, 100, y0, 0.1, beta, gamma)

# Plot the results
plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Infected')
plt.plot(t, y[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
