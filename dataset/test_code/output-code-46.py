import numpy as np
from scipy.integrate import odeint

# Define the SEIRHD model

def seirhd_model(y, t, N, beta, sigma, gamma, delta, mu):
    S, E, I, R, H, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + delta + mu) * I
    dRdt = gamma * I
    dHdt = delta * I
    dDdt = mu * I
    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt

# Initial conditions

N = 1000000  # total population
E0 = 10      # initial exposed individuals
I0 = 1       # initial infected individuals
R0 = 0       # initial recovered individuals
H0 = 0       # initial hospitalized individuals
D0 = 0       # initial deceased individuals
S0 = N - E0 - I0 - R0 - H0 - D0  # initial susceptible individuals

# Parameters

beta = 0.2   # transmission rate
sigma = 1/5  # incubation rate
gamma = 1/10 # recovery rate
delta = 1/2  # hospitalization rate
mu = 1/20   # fatality rate

# Time vector

t = np.linspace(0, 365, 365)

# Integrate the SEIRHD equations

y = S0, E0, I0, R0, H0, D0
result = odeint(seirhd_model, y, t, args=(N, beta, sigma, gamma, delta, mu))
S, E, I, R, H, D = result.T

# Plot the results

import matplotlib.pyplot as plt

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SEIRHD Model')
plt.legend()
plt.grid(True)
plt.show()
