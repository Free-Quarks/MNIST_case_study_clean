import numpy as np
import matplotlib.pyplot as plt


# Define the model dynamics

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Define the Runge-Kutta 3rd order method

def rk3_step(model, t, y, dt, *args):
    k1 = np.array(model(t, y, *args))
    k2 = np.array(model(t + 0.5 * dt, y + 0.5 * dt * k1, *args))
    k3 = np.array(model(t + dt, y - dt * k1 + 2 * dt * k2, *args))
    return y + dt / 6 * (k1 + 4 * k2 + k3)


# Set initial conditions

S0 = 1000
I0 = 1
R0 = 0
N = S0 + I0 + R0
y0 = [S0, I0, R0]


# Set model parameters

beta = 0.3
# infection rate

gamma = 0.1
# recovery rate


# Set simulation parameters

t0 = 0
# initial time

t_end = 100
# end time

dt = 0.1
# time step


# Perform the simulation

t_values = np.arange(t0, t_end, dt)
y_values = np.zeros((len(t_values), 3))
y_values[0] = y0

for i in range(1, len(t_values)):
    y_values[i] = rk3_step(sir_model, t_values[i-1], y_values[i-1], dt, beta, gamma)


# Plot the results

plt.figure()
plt.plot(t_values, y_values[:, 0], label='S')
plt.plot(t_values, y_values[:, 1], label='I')
plt.plot(t_values, y_values[:, 2], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
