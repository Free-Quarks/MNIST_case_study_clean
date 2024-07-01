import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sidarthe_model(y, t, beta, gamma, delta, alpha, lam, rho, theta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    
    dSdt = -beta * S * (I + delta * A) / N
    dIdt = beta * S * (I + delta * A) / N - (1 - delta) * gamma * I - delta * alpha * I
    dDdt = delta * alpha * I
    dAdt = (1 - delta) * gamma * I - rho * A
    dRdt = rho * A
    dTdt = theta * A
    dHdt = lam * T
    dEdt = (1 - theta) * A
    
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


# Parameters
beta = 0.2
gamma = 0.1
alpha = 0.05
lam = 0.001
rho = 0.01
theta = 0.02
delta = 0.02

# Initial conditions
S0 = 999
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

# Time vector
t = np.linspace(0, 100, 1000)

# Initial conditions vector
y0 = [S0, I0, D0, A0, R0, T0, H0, E0]

# Integrate the ODEs
sol = odeint(sidarthe_model, y0, t, args=(beta, gamma, delta, alpha, lam, rho, theta))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='D')
plt.plot(t, sol[:, 3], label='A')
plt.plot(t, sol[:, 4], label='R')
plt.plot(t, sol[:, 5], label='T')
plt.plot(t, sol[:, 6], label='H')
plt.plot(t, sol[:, 7], label='E')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
