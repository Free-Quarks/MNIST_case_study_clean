import numpy as np
import matplotlib.pyplot as plt


# Define the SIR model function

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Define the RK2 method


def rk2_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + h, y + h * k1, *args)
    return y + 0.5 * h * (k1 + k2)


# Define the main simulation function


def simulate_sir_model(t, initial_conditions, beta, gamma, h):
    num_steps = int(np.ceil(t / h))
    y = np.zeros((num_steps + 1, 3))
    y[0, :] = initial_conditions
    for i in range(num_steps):
        t_i = i * h
        y[i + 1, :] = rk2_step(sir_model, t_i, y[i, :], h, beta, gamma)
    return y


# Set the parameters

beta = 0.2
# Infection rate

gamma = 0.1
# Recovery rate

h = 0.1
# Step size


# Set the initial conditions

initial_conditions = [1000, 1, 0]
# Number of susceptibles, infected, and recovered individuals at t = 0


# Simulate the SIR model

t = 30
# Total simulation time

y = simulate_sir_model(t, initial_conditions, beta, gamma, h)


# Plot the results

plt.plot(np.arange(0, t + h, h), y[:, 0], label='Susceptible')
plt.plot(np.arange(0, t + h, h), y[:, 1], label='Infected')
plt.plot(np.arange(0, t + h, h), y[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()

