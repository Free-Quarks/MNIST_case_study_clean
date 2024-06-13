import numpy as np
import matplotlib.pyplot as plt

# Function that contains the model dynamics

def SIR_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Parameters
beta = 0.2
gamma = 0.1

# Initial conditions
S0 = 0.99
I0 = 0.01
R0 = 0

# Time vector
t = np.linspace(0, 100, 1000)

# Solve the ODE
from scipy.integrate import solve_ivp
solution = solve_ivp(SIR_model, [0, 100], [S0, I0, R0], args=(beta, gamma), t_eval=t)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='S')
plt.plot(solution.t, solution.y[1], label='I')
plt.plot(solution.t, solution.y[2], label='R')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model')
plt.legend()
plt.show()
