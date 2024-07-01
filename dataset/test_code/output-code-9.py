import numpy as np


def seir_model(t, y, params):
    S, E, I, R = y
    beta, gamma, sigma = params
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def rk4(t0, y0, dt, params, model_function):
    k1 = np.asarray(model_function(t0, y0, params))
    k2 = np.asarray(model_function(t0 + dt / 2, y0 + dt * k1 / 2, params))
    k3 = np.asarray(model_function(t0 + dt / 2, y0 + dt * k2 / 2, params))
    k4 = np.asarray(model_function(t0 + dt, y0 + dt * k3, params))
    y1 = y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y1


# Example usage

# Parameters
beta = 0.2
sigma = 0.1
gamma = 0.05

# Initial conditions
S0 = 990
E0 = 10
I0 = 0
R0 = 0
y0 = [S0, E0, I0, R0]

# Time grid
t0 = 0
t_max = 100
dt = 1
num_steps = int(t_max / dt)
t = np.linspace(t0, t_max, num_steps + 1)

# Initialize result arrays
S = np.zeros(num_steps + 1)
E = np.zeros(num_steps + 1)
I = np.zeros(num_steps + 1)
R = np.zeros(num_steps + 1)

# Assign initial conditions
S[0] = S0
E[0] = E0
I[0] = I0
R[0] = R0

# Solve the system using RK4
for i in range(num_steps):
    t_i = t[i]
    y_i = [S[i], E[i], I[i], R[i]]
    y_next = rk4(t_i, y_i, dt, [beta, gamma, sigma], seir_model)
    S[i + 1], E[i + 1], I[i + 1], R[i + 1] = y_next

# Plotting the results
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
