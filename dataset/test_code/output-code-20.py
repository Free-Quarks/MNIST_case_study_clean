import numpy as np
import matplotlib.pyplot as plt

def seirhd_model(params, t):
    S, E, I, R, H, D = params
    N = S + E + I + R + H + D
    beta, sigma, gamma, eta, mu = 0.2, 0.1, 0.1, 0.05, 0.01
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (1 - eta) * gamma * I - eta * mu * I
    dRdt = (1 - eta) * gamma * I
    dHdt = (1 - eta) * mu * I
    dDdt = eta * mu * I
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]


def euler_integration(model, initial_conditions, t):
    num_compartments = len(initial_conditions)
    num_time_steps = len(t)
    compartments = np.zeros((num_time_steps, num_compartments))
    compartments[0] = initial_conditions

    for i in range(1, num_time_steps):
        dt = t[i] - t[i-1]
        compartments[i] = compartments[i-1] + dt * model(compartments[i-1], t[i-1])

    return compartments


# Example usage
t = np.arange(0, 100, 1)
initial_conditions = [1000, 1, 0, 0, 0, 0]

compartments = euler_integration(seirhd_model, initial_conditions, t)

plt.plot(t, compartments[:, 0], label='S')
plt.plot(t, compartments[:, 1], label='E')
plt.plot(t, compartments[:, 2], label='I')
plt.plot(t, compartments[:, 3], label='R')
plt.plot(t, compartments[:, 4], label='H')
plt.plot(t, compartments[:, 5], label='D')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
