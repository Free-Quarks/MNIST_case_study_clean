import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Model parameters
gamma = 0.1  # Recovery rate
sigma = 0.2  # Infection rate

# Initial conditions
S0 = 99  # Susceptible population
E0 = 1  # Exposed population
I0 = 0  # Infected population
R0 = 0  # Recovered population

def seir_model(t, y):
    S, E, I, R = y
    dSdt = -sigma * S * I
    dEdt = sigma * S * I - gamma * E
    dIdt = gamma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Solve the SEIR model
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 1001)
y0 = [S0, E0, I0, R0]
solution = solve_ivp(seir_model, t_span, y0, t_eval=t_eval, method='RK45')

# Plot the results
plt.plot(solution.t, solution.y[0], label='Susceptible')
plt.plot(solution.t, solution.y[1], label='Exposed')
plt.plot(solution.t, solution.y[2], label='Infected')
plt.plot(solution.t, solution.y[3], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

