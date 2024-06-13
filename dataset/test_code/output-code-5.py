import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, gamma, sigma, N, I0, E0, R0, t_max):
    # Initialize arrays to store the compartments
    S = np.zeros(t_max)
    E = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)

    # Set initial conditions
    S[0] = N - I0 - E0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # Euler's method to solve the SEIR equations
    for t in range(1, t_max):
        dSdt = -beta * S[t-1] * I[t-1] / N
        dEdt = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dIdt = sigma * E[t-1] - gamma * I[t-1]
        dRdt = gamma * I[t-1]

        S[t] = S[t-1] + dSdt
        E[t] = E[t-1] + dEdt
        I[t] = I[t-1] + dIdt
        R[t] = R[t-1] + dRdt

    return S, E, I, R


# Set parameters
beta = 0.2
gamma = 0.1
sigma = 0.5
N = 1000
I0 = 10
E0 = 5
R0 = 0
t_max = 100

# Run the model
S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, t_max)

# Plot the results
plt.plot(range(t_max), S, label='Susceptible')
plt.plot(range(t_max), E, label='Exposed')
plt.plot(range(t_max), I, label='Infected')
plt.plot(range(t_max), R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
