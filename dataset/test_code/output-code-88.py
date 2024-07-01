import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def seird_model(t, y, beta, sigma, gamma, delta):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - (gamma + delta) * I
    dR = gamma * I
    dD = delta * I
    return [dS, dE, dI, dR, dD]


def simulate_seird_model(N, beta, sigma, gamma, delta, E0, I0, R0, D0, t_max, dt):
    S0 = N - E0 - I0 - R0 - D0
    y0 = [S0, E0, I0, R0, D0]
    t_eval = np.arange(0, t_max + dt, dt)
    sol = RK45(lambda t, y: seird_model(t, y, beta, sigma, gamma, delta), 0, y0, t_max, first_step=dt)
    results = np.zeros((len(t_eval), len(y0)))
    for i, t in enumerate(t_eval):
        results[i] = sol.y
        sol.step()
    return t_eval, results


N = 100000
beta = 0.3
sigma = 1 / 5.2
gamma = 1 / 10
delta = 1 / 20
E0 = 10
I0 = 1
R0 = 0
D0 = 0
t_max = 100
dt = 0.1
t, y = simulate_seird_model(N, beta, sigma, gamma, delta, E0, I0, R0, D0, t_max, dt)

plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Exposed')
plt.plot(t, y[:, 2], label='Infected')
plt.plot(t, y[:, 3], label='Recovered')
plt.plot(t, y[:, 4], label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.show()
