import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Define the SEIRD model

# Function that contains the model dynamics

def seird_model(y, t, N, beta, gamma, sigma, mu):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (1 - mu) * gamma * I - mu * gamma * I
    dRdt = (1 - mu) * gamma * I
    dDdt = mu * gamma * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


# Parameters
N = 1000
beta = 0.2
gamma = 0.1
sigma = 0.01
mu = 0.02

# Initial conditions
S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0

# Time vector
t = np.linspace(0, 100, 100)

# Solve the SEIRD model
y0 = S0, E0, I0, R0, D0
sol = odeint(seird_model, y0, t, args=(N, beta, gamma, sigma, mu))
S, E, I, R, D = sol.T

# Plotting
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.grid(True)
plt.show()
