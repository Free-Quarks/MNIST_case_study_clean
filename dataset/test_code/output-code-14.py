import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the SEIRD model
def seird_model(t, y, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

# Parameters
beta = 0.5
sigma = 0.1
gamma = 0.2
mu = 0.05

# Initial conditions
S0 = 990
E0 = 10
I0 = 0
R0 = 0
D0 = 0

# Time vector
t_span = (0, 100)

# Solve the SEIRD model
solution = solve_ivp(fun=seird_model, t_span=t_span, y0=[S0, E0, I0, R0, D0], method='RK45', args=(beta, sigma, gamma, mu))

# Plot the results
plt.plot(solution.t, solution.y[0], label='Susceptible')
plt.plot(solution.t, solution.y[1], label='Exposed')
plt.plot(solution.t, solution.y[2], label='Infected')
plt.plot(solution.t, solution.y[3], label='Recovered')
plt.plot(solution.t, solution.y[4], label='Deaths')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
