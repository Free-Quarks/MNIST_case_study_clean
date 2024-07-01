import numpy as np
from scipy.integrate import odeint

def seirhd_model(y, t, params):
    S, E, I, R, H, D = y
    N = S + E + I + R + H + D
    beta, sigma, gamma, alpha, delta = params
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - alpha * I
    dRdt = gamma * I
    dHdt = alpha * I - delta * H
    dDdt = delta * H
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]

# Define initial conditions
S0 = 999
E0 = 1
I0 = 0
R0 = 0
H0 = 0
D0 = 0

# Define model parameters
beta = 0.2
sigma = 1/5.2
gamma = 1/2
alpha = 1/6
delta = 1/8
params = [beta, sigma, gamma, alpha, delta]

# Define time points
t = np.linspace(0, 100, 100)

# Solve the SEIRHD model
y0 = [S0, E0, I0, R0, H0, D0]
y = odeint(seirhd_model, y0, t, args=(params,))

# Extract the compartments
S = y[:, 0]
E = y[:, 1]
I = y[:, 2]
R = y[:, 3]
H = y[:, 4]
D = y[:, 5]

# Plotting the results
import matplotlib.pyplot as plt

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRHD Model')
plt.legend()
plt.show()
