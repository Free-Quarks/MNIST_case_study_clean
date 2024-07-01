import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that contains the model dynamics

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions
y0 = [0.99, 0.01, 0]

# Time points
t = np.linspace(0, 100, 1000)

# Parameters
beta = 0.2
gamma = 0.1

# Solve the differential equation
solution = odeint(sir_model, y0, t, args=(beta, gamma))

# Plot the results
plt.plot(t, solution[:, 0], label='Susceptible')
plt.plot(t, solution[:, 1], label='Infected')
plt.plot(t, solution[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model')
plt.legend()
plt.show()
