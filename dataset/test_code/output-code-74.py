import numpy as np


def seirhd_model(t, y, params):
    S, E, I, R, H, D = y
    beta, sigma, gamma, delta, mu = params
    N = S + E + I + R + H + D

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - delta * I
    dRdt = gamma * I
    dHdt = delta * I - mu * H
    dDdt = mu * H

    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]



# Parameters
beta = 0.2
sigma = 0.5
gamma = 0.1
mu = 0.01
delta = 0.05

# Initial conditions
S0 = 990
E0 = 10
I0 = 0
R0 = 0
H0 = 0
D0 = 0

# Time vector
t = np.linspace(0, 100, 1000)

# Parameters vector
params = [beta, sigma, gamma, delta, mu]

# Initial conditions vector
initial_conditions = [S0, E0, I0, R0, H0, D0]

# Solve the model
solution = odeint(seirhd_model, initial_conditions, t, args=(params,))

# Plotting the results
plt.plot(t, solution[:, 0], label='Susceptible')
plt.plot(t, solution[:, 1], label='Exposed')
plt.plot(t, solution[:, 2], label='Infected')
plt.plot(t, solution[:, 3], label='Recovered')
plt.plot(t, solution[:, 4], label='Hospitalized')
plt.plot(t, solution[:, 5], label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRHD Model')
plt.legend()
plt.show()
