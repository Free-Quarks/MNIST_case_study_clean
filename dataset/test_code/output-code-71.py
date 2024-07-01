import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the model

# Function that returns dy/dt

def model(y, t):
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    H = y[4]
    D = y[5]
    N = S + E + I + R + H + D
    alpha = 0.2
    beta = 0.5
    gamma = 0.1
    delta = 0.05
    mu = 0.01
    dSdt = -alpha * S * I / N
    dEdt = alpha * S * I / N - beta * E
    dIdt = beta * E - gamma * I - delta * I
    dRdt = gamma * I
    dHdt = delta * I - mu * H
    dDdt = mu * H
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]

# Initial conditions
S0 = 990
E0 = 10
I0 = 0
R0 = 0
H0 = 0
D0 = 0
y0 = [S0, E0, I0, R0, H0, D0]

# Time grid
t = np.linspace(0, 100, 100)

# Solve the ODE
y = odeint(model, y0, t)

# Plot the results
plt.plot(t, y[:, 0], 'b', label='Susceptible')
plt.plot(t, y[:, 1], 'g', label='Exposed')
plt.plot(t, y[:, 2], 'r', label='Infected')
plt.plot(t, y[:, 3], 'y', label='Recovered')
plt.plot(t, y[:, 4], 'm', label='Hospitalized')
plt.plot(t, y[:, 5], 'c', label='Dead')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SEIRHD Model')
plt.legend(loc='best')
plt.show()
