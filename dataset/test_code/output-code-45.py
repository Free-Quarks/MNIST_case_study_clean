import numpy as np
import matplotlib.pyplot as plt

# Model parameters
beta = 0.2
sigma = 0.1
gamma = 0.05
rho = 0.01
alpha = 0.005

# Initial conditions
S0 = 990
E0 = 10
I0 = 0
R0 = 0
H0 = 0
D0 = 0

# Time vector
t = np.linspace(0, 200, 201)

# Function to define the model dynamics

def seirhd_model(y, t):
    S, E, I, R, H, D = y
    dSdt = -beta * S * I - rho * S
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I - alpha * I
    dRdt = gamma * I
    dHdt = alpha * I
    dDdt = rho * S
    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt

# Solve the differential equations
y0 = S0, E0, I0, R0, H0, D0
solution = odeint(seirhd_model, y0, t)

# Plot the results
plt.plot(t, solution[:, 0], label='Susceptible')
plt.plot(t, solution[:, 1], label='Exposed')
plt.plot(t, solution[:, 2], label='Infected')
plt.plot(t, solution[:, 3], label='Recovered')
plt.plot(t, solution[:, 4], label='Hospitalized')
plt.plot(t, solution[:, 5], label='Deaths')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRHD Model')
plt.legend()
plt.grid()
plt.show()
