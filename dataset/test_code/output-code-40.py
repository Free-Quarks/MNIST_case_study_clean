import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(y, t, params):
    S, I, D, A, R, T, H, E = y
    beta, alpha, delta, rho, theta, kappa, epsilon, gamma, mu = params
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + A) / N
    dIdt = beta * S * (I + A) / N - (1 - alpha) * delta * I - alpha * rho * I - alpha * (1 - rho) * I - epsilon * kappa * I
    dDdt = epsilon * kappa * I
    dAdt = alpha * (1 - rho) * I - theta * A
    dRdt = alpha * rho * I + theta * A + gamma * (1 - mu) * R
    dTdt = gamma * mu * R
    dHdt = (1 - alpha) * delta * I
    dEdt = (1 - alpha) * (1 - delta) * I
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def sidarthe_simulation(S0, I0, D0, A0, R0, T0, H0, E0, beta, alpha, delta, rho, theta, kappa, epsilon, gamma, mu, days):
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]
    params = [beta, alpha, delta, rho, theta, kappa, epsilon, gamma, mu]
    t = np.arange(days)
    y = np.zeros((days, len(y0)))
    y[0] = y0
    for i in range(1, days):
        tspan = [t[i-1], t[i]]
        y[i] = y[i-1] + np.asarray(euler(lambda y, t: sidarthe_model(y, t, params), y[i-1], tspan, 1))
    return t, y


S0 = 60000000
I0 = 100
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

beta = 0.5
alpha = 1
rho = 0.3
delta = 0.1
theta = 0.1
kappa = 0.1
epsilon = 0.1
gamma = 0.2
mu = 0.1
days = 100


t, y = sidarthe_simulation(S0, I0, D0, A0, R0, T0, H0, E0, beta, alpha, delta, rho, theta, kappa, epsilon, gamma, mu, days)


plt.plot(t, y[:, 1], 'r', label='Infected')
plt.plot(t, y[:, 2], 'g', label='Dead')
plt.plot(t, y[:, 3], 'b', label='Asymptomatic')
plt.plot(t, y[:, 4], 'y', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
