import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Initial conditions
S0 = 990
E0 = 10
I0 = 0
R0 = 0
y0 = S0, E0, I0, R0

# Parameters
beta = 0.2
gamma = 0.1
sigma = 0.1

# Time vector
t = np.linspace(0, 100, 1000)

# Solve the SEIR model
sol = odeint(seir_model, y0, t, args=(beta, gamma, sigma))

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(t, sol[:, 0], label='Susceptible')
plt.plot(t, sol[:, 1], label='Exposed')
plt.plot(t, sol[:, 2], label='Infected')
plt.plot(t, sol[:, 3], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.grid(True)
plt.show()
