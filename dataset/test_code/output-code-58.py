import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def seir_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def simulate_seir_model(S0, E0, I0, R0, beta, sigma, gamma, t_start, t_end, num_points):
    t = np.linspace(t_start, t_end, num_points)
    y0 = [S0, E0, I0, R0]
    sol = solve_ivp(lambda t, y: seir_model(t, y, beta, sigma, gamma), [t_start, t_end], y0, t_eval=t)
    return sol.t, sol.y


# Example usage
S0 = 9999
E0 = 1
I0 = 0
R0 = 0
beta = 0.25
sigma = 0.1
gamma = 0.05
t_start = 0
t_end = 100
t_num_points = 1000

t, y = simulate_seir_model(S0, E0, I0, R0, beta, sigma, gamma, t_start, t_end, t_num_points)

plt.plot(t, y[0], label='S')
plt.plot(t, y[1], label='E')
plt.plot(t, y[2], label='I')
plt.plot(t, y[3], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
