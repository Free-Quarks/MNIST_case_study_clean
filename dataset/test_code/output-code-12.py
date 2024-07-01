import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Model parameters
beta = 0.3
sigma = 0.1
gamma = 0.05
mu = 0.01
N = 1000

# Initial conditions
S0 = N - 1
E0 = 1
I0 = 0
R0 = 0
D0 = 0

# SEIRD model

def seird_model(t, y):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return dSdt, dEdt, dIdt, dRdt, dDdt

# Solve the system of ODEs
solution = solve_ivp(seird_model, [0, 200], [S0, E0, I0, R0, D0], method='RK45', t_eval=np.arange(0, 200, 0.1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='Susceptible')
plt.plot(solution.t, solution.y[1], label='Exposed')
plt.plot(solution.t, solution.y[2], label='Infected')
plt.plot(solution.t, solution.y[3], label='Recovered')
plt.plot(solution.t, solution.y[4], label='Deaths')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.show()
