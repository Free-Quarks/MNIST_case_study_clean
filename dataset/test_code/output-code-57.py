import numpy as np


def seir_model(beta, sigma, gamma, N, initial_conditions, t_end, dt):
    S_0, E_0, I_0, R_0 = initial_conditions
    num_steps = int(t_end / dt)

    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    t = np.linspace(0, t_end, num_steps)

    S[0] = S_0
    E[0] = E_0
    I[0] = I_0
    R[0] = R_0

    for i in range(1, num_steps):
        dS_dt = -beta * S[i-1] * I[i-1] / N
        dE_dt = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI_dt = sigma * E[i-1] - gamma * I[i-1]
        dR_dt = gamma * I[i-1]

        S[i] = S[i-1] + dt * dS_dt
        E[i] = E[i-1] + dt * dE_dt
        I[i] = I[i-1] + dt * dI_dt
        R[i] = R[i-1] + dt * dR_dt

    return S, E, I, R


# Example usage
beta = 0.3
sigma = 0.2
gamma = 0.1
N = 1000
initial_conditions = (999, 1, 0, 0)
t_end = 100
dt = 0.1

S, E, I, R = seir_model(beta, sigma, gamma, N, initial_conditions, t_end, dt)
