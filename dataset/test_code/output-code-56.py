import numpy as np
from scipy.integrate import odeint

# Define the SEIR model function
def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Set initial conditions
y0 = [999, 1, 0, 0]  # S, E, I, R

# Set model parameters
beta = 0.3
sigma = 0.1
gamma = 0.05

# Set time points
t = np.linspace(0, 100, 100)

# Solve the SEIR model
solution = odeint(seir_model, y0, t, args=(beta, gamma, sigma))

# Extract the variables
S = solution[:, 0]
E = solution[:, 1]
I = solution[:, 2]
R = solution[:, 3]

# Plot the variables
import matplotlib.pyplot as plt

plt.plot(t, S, label='S')
plt.plot(t, E, label='E')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
