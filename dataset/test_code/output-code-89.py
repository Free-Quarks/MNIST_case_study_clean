import numpy as np

# Define the SEIRD model

# Parameters
beta = 0.2
sigma = 1/5.2
gamma = 1/14
mu = 1/30

# Initial conditions
N = 10000
I0 = 1
E0 = 0
R0 = 0
D0 = 0
S0 = N - I0 - R0 - D0 - E0

# Time vector
T = 365

# Define the differential equations

def seird_model(y, t):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return dSdt, dEdt, dIdt, dRdt, dDdt

# Solve the differential equations using RK4

def rk4_solver(model, y0, t0, T, dt):
    y = np.zeros((int(T / dt) + 1, len(y0)))
    y[0] = y0
    t = t0
    i = 1
    while t < T:
        k1 = dt * model(y[i-1], t)
        k2 = dt * model(y[i-1] + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * model(y[i-1] + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * model(y[i-1] + k3, t + dt)
        y[i] = y[i-1] + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        i += 1
    return y

# Run the simulation and plot the results

y0 = S0, E0, I0, R0, D0
t0 = 0
dt = 1
y = rk4_solver(seird_model, y0, t0, T, dt)

import matplotlib.pyplot as plt

plt.plot(y[:, 0], label='Susceptible')
plt.plot(y[:, 1], label='Exposed')
plt.plot(y[:, 2], label='Infected')
plt.plot(y[:, 3], label='Recovered')
plt.plot(y[:, 4], label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.show()
