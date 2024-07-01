import numpy as np

def sidarthe_model(t, y, N, beta, gamma, delta, epsilon, theta, rho, sigma):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * ((I + delta * A) / N)
    dIdt = beta * S * ((I + delta * A) / N) - (gamma + theta) * I
    dDdt = delta * beta * S * ((I + delta * A) / N) - (gamma + rho) * D
    dAdt = theta * I - (epsilon + sigma) * A
    dRdt = gamma * (I + D) + sigma * A
    dTdt = rho * D
    dHdt = epsilon * A
    dEdt = delta * beta * S * ((I + delta * A) / N)
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt

# Define parameters
N = 100000
beta = 0.2
gamma = 0.1
theta = 0.1
rho = 0.05
sigma = 0.01
delta = 0.02
epsilon = 0.01

# Define initial conditions
S0, I0, D0, A0, R0, T0, H0, E0 = N - 1, 1, 0, 0, 0, 0, 0, 0
y0 = S0, I0, D0, A0, R0, T0, H0, E0

# Define time vector
t = np.linspace(0, 50, 1000)

# Solve the ODEs
from scipy.integrate import solve_ivp
sol = solve_ivp(sidarthe_model, [t[0], t[-1]], y0, args=(N, beta, gamma, delta, epsilon, theta, rho, sigma), method='RK45', t_eval=t)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0], label='S')
plt.plot(sol.t, sol.y[1], label='I')
plt.plot(sol.t, sol.y[2], label='D')
plt.plot(sol.t, sol.y[3], label='A')
plt.plot(sol.t, sol.y[4], label='R')
plt.plot(sol.t, sol.y[5], label='T')
plt.plot(sol.t, sol.y[6], label='H')
plt.plot(sol.t, sol.y[7], label='E')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
