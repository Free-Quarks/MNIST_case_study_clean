import numpy as np
import matplotlib.pyplot as plt


def model_func(t, y, beta, gamma, sigma, rho, alpha):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dS = -beta * S * (I + rho * A) / N
    dI = (beta * S * (I + rho * A) / N) - (gamma + sigma) * I
    dD = gamma * I
    dA = sigma * I - alpha * A
    dR = (1 - alpha) * sigma * I
    dT = rho * sigma * I
    dH = alpha * sigma * I
    dE = rho * (1 - sigma) * I
    dy = [dS, dI, dD, dA, dR, dT, dH, dE]
    return dy


def RK2(model_func, t0, y0, h, beta, gamma, sigma, rho, alpha, num_steps):
    t = np.zeros(num_steps)
    y = np.zeros((num_steps, len(y0)))
    y[0, :] = y0
    for i in range(1, num_steps):
        t[i] = t[i-1] + h
        k1 = model_func(t[i-1], y[i-1, :], beta, gamma, sigma, rho, alpha)
        k2 = model_func(t[i-1] + h/2, y[i-1, :] + (h/2) * k1, beta, gamma, sigma, rho, alpha)
        y[i, :] = y[i-1, :] + h * k2
    return t, y


# Example usage

# Define initial conditions
S0 = 1000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0
y0 = [S0, I0, D0, A0, R0, T0, H0, E0]

# Define model parameters
beta = 0.3
gamma = 0.1
sigma = 0.1
rho = 0.8
alpha = 0.5

# Define simulation parameters
t0 = 0
h = 0.1
num_steps = 1000

# Run simulation
t, y = RK2(model_func, t0, y0, h, beta, gamma, sigma, rho, alpha, num_steps)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 1], label='Infected')
plt.plot(t, y[:, 2], label='Deaths')
plt.plot(t, y[:, 3], label='Asymptomatic')
plt.plot(t, y[:, 4], label='Recovered')
plt.plot(t, y[:, 5], label='Tested')
plt.plot(t, y[:, 6], label='Hospitalized')
plt.plot(t, y[:, 7], label='Exposed')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
