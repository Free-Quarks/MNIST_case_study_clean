import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seirhd_model(y, t, beta, sigma, gamma, mu, rho):
    S, E, I, R, H, D = y
    N = S + E + I + R + H + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (1 - rho) * gamma * I - rho * mu * I
    dRdt = (1 - rho) * gamma * I
    dHdt = rho * mu * I
    dDdt = rho * (1 - mu) * I
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]

# Initial conditions
S0 = 999
E0 = 1
I0 = 0
R0 = 0
H0 = 0
D0 = 0

# Parameters
beta = 0.2
sigma = 1 / 5.2
gamma = 1 / 2.3
mu = 0.01
rho = 0.1

# Time vector
t = np.linspace(0, 100, 100)

# Solve the SEIRHD model
y = odeint(seirhd_model, [S0, E0, I0, R0, H0, D0], t, args=(beta, sigma, gamma, mu, rho))

# Plot the results
plt.plot(t, y[:, 0], label='S')
plt.plot(t, y[:, 1], label='E')
plt.plot(t, y[:, 2], label='I')
plt.plot(t, y[:, 3], label='R')
plt.plot(t, y[:, 4], label='H')
plt.plot(t, y[:, 5], label='D')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

