import numpy as np
import matplotlib.pyplot as plt


# Parameters
beta = 0.2
gamma = 0.1
N = 1000
T = 100


# Function to define the model dynamics

def SIR_model(S, I, R, beta, gamma):
    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    return dS_dt, dI_dt, dR_dt


# Function to solve the model using RK3

def solve_SIR_model(beta, gamma, N, T):
    dt = 0.1
    num_steps = int(T / dt)
    S = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)

    S[0] = N - 1
    I[0] = 1

    for i in range(num_steps):
        dS_1, dI_1, dR_1 = SIR_model(S[i], I[i], R[i], beta, gamma)
        S_half = S[i] + 0.5 * dt * dS_1
        I_half = I[i] + 0.5 * dt * dI_1
        R_half = R[i] + 0.5 * dt * dR_1

        dS_2, dI_2, dR_2 = SIR_model(S_half, I_half, R_half, beta, gamma)
        S[i+1] = S[i] + dt * (2/3 * dS_1 + 1/3 * dS_2)
        I[i+1] = I[i] + dt * (2/3 * dI_1 + 1/3 * dI_2)
        R[i+1] = R[i] + dt * (2/3 * dR_1 + 1/3 * dR_2)

    return S, I, R


# Solve the model
S, I, R = solve_SIR_model(beta, gamma, N, T)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time Steps')
plt.ylabel('Number of Individuals')
plt.title('SIR Model using RK3')
plt.legend()
plt.grid(True)
plt.show()
