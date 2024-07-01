import numpy as np
import matplotlib.pyplot as plt


def SIR_model_beta_gamma(S, I, R, beta, gamma):
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def euler_integration(S, I, R, beta, gamma, dt, num_steps):
    S_values = [S]
    I_values = [I]
    R_values = [R]
    for _ in range(num_steps):
        dSdt, dIdt, dRdt = SIR_model_beta_gamma(S, I, R, beta, gamma)
        S += dSdt * dt
        I += dIdt * dt
        R += dRdt * dt
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)
    return S_values, I_values, R_values


S = 999
I = 1
R = 0
beta = 0.2
gamma = 0.1
dt = 0.1
num_steps = 100

S_values, I_values, R_values = euler_integration(S, I, R, beta, gamma, dt, num_steps)

plt.plot(S_values, label='Susceptible')
plt.plot(I_values, label='Infected')
plt.plot(R_values, label='Recovered')
plt.xlabel('Time step')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('SIR Model using Euler Integration')
plt.show()
