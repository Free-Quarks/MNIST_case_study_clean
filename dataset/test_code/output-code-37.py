import numpy as np
import matplotlib.pyplot as plt


# Function that defines the SEIRD model dynamics

def seird_model(t, y, beta, gamma, alpha, delta):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


# Function to solve the SEIRD model using RK2

def solve_seird_model(t, y0, beta, gamma, alpha, delta, dt):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = seird_model(t[i-1], y[i-1], beta, gamma, alpha, delta)
        k2 = seird_model(t[i-1] + dt, y[i-1] + dt * k1, beta, gamma, alpha, delta)
        y[i] = y[i-1] + dt * (k1 + k2) / 2
    return y


# Example usage

t_end = 100
dt = 0.1
t = np.arange(0, t_end, dt)
y0 = [1000, 1, 0, 0, 0]
beta = 0.2
alpha = 0.1
gamma = 0.1
delta = 0.01

y = solve_seird_model(t, y0, beta, gamma, alpha, delta, dt)

plt.plot(t, y[:, 0], label='S')
plt.plot(t, y[:, 1], label='E')
plt.plot(t, y[:, 2], label='I')
plt.plot(t, y[:, 3], label='R')
plt.plot(t, y[:, 4], label='D')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.show()
