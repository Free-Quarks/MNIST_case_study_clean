import numpy as np
from scipy.integrate import odeint

# Define the SIDARTHE model

# Parameters
gamma = 1/8  # Recovery rate
mu = 0.03  # Mortality rate
sigma = 1/5  # Incubation rate
eta = 1/3  # Post incubation rate
alpha = 0.2  # Hospitalization rate
theta = 0.1  # Hospitalization death rate
rho = 0.15  # ICU rate
omega = 0.3  # ICU death rate

# Model equations
def sidarthe(y, t):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -sigma * S * (I + A) / N
    dIdt = sigma * S * (I + A) / N - (eta + gamma) * I
    dDdt = mu * gamma * I - (alpha + theta) * D
    dAdt = (1 - mu) * gamma * I - (alpha + rho) * A
    dRdt = (1 - alpha) * (1 - theta) * D + (1 - alpha) * (1 - rho) * A - gamma * R
    dTdt = alpha * (1 - theta) * D + alpha * (1 - rho) * A - gamma * T
    dHdt = alpha * theta * D + alpha * rho * A - (omega + gamma) * H
    dEdt = eta * I - H
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]

# Initial conditions
y0 = [9999, 1, 0, 0, 0, 0, 0, 0]  # Initial number of individuals in each compartment

# Time vector
t = np.linspace(0, 50, 100)  # Grid of time points (in days)

# Integrate the SIDARTHE equations
sol = odeint(sidarthe, y0, t)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='D')
plt.plot(t, sol[:, 3], label='A')
plt.plot(t, sol[:, 4], label='R')
plt.plot(t, sol[:, 5], label='T')
plt.plot(t, sol[:, 6], label='H')
plt.plot(t, sol[:, 7], label='E')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
