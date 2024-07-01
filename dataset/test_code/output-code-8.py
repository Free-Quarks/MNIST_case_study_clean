import numpy as np
import matplotlib.pyplot as plt

# Function to define the SEIR model dynamics

def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Function to solve the SEIR model using 3rd order Runge-Kutta

def solve_seir_model(beta, gamma, sigma, S_0, E_0, I_0, R_0, t_max, dt):
    t = np.arange(0, t_max + dt, dt)
    N = S_0 + E_0 + I_0 + R_0
    y = [S_0, E_0, I_0, R_0]
    result = []
    for i in range(len(t) - 1):
        h = t[i+1] - t[i]
        k1 = seir_model(y, t[i], beta, gamma, sigma)
        k2 = seir_model(y + h/2 * k1, t[i] + h/2, beta, gamma, sigma)
        k3 = seir_model(y - h*k1 + 2*h*k2, t[i+1], beta, gamma, sigma)
        y = y + h/6 * (k1 + 4*k2 + k3)
        result.append(y)
    return np.array(result)

# Initial conditions

S_0 = 990
E_0 = 10
I_0 = 0
R_0 = 0

# Model parameters

beta = 0.3
sigma = 0.1
gamma = 0.05

t_max = 100
dt = 0.1

# Solve the SEIR model

result = solve_seir_model(beta, gamma, sigma, S_0, E_0, I_0, R_0, t_max, dt)

# Plot the results

plt.plot(result[:,0], label='Susceptible')
plt.plot(result[:,1], label='Exposed')
plt.plot(result[:,2], label='Infected')
plt.plot(result[:,3], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
