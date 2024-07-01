import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, t, params):
    S, I, D, A, R, T, H, E = N
    alpha, beta, gamma, delta, epsi, theta, rho, mu = params
    dSdt = -alpha * S * I / N.sum()
    dIdt = alpha * S * I / N.sum() - beta * I
    dDdt = delta * (epsi * A + theta * H + rho * I + mu * T) * I / N.sum()
    dAdt = (1 - delta) * (epsi * A + theta * H + rho * I + mu * T) * I / N.sum()
    dRdt = (1 - delta) * (epsi * A + theta * H + rho * I + mu * T) * I / N.sum()
    dTdt = epsi * A * I / N.sum()
    dHdt = theta * H * I / N.sum()
    dEdt = rho * I * I / N.sum()
    return np.array([dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt])


def euler_integration(model_function, N, t, params):
    dt = t[1] - t[0]
    S, I, D, A, R, T, H, E = N
    n = len(t)
    result = np.zeros((n, len(N)))
    result[0] = N
    for i in range(1, n):
        result[i] = result[i-1] + dt * model_function(result[i-1], t[i], params)
    return result[:,0], result[:,1], result[:,2], result[:,3], result[:,4], result[:,5], result[:,6], result[:,7]


# Parameters
alpha = 0.3
beta = 0.4
gamma = 0
mu = 0.2
delta = 0.1
epsi = 0.2
theta = 0.15
rho = 0.05


# Initial conditions
S0 = 999
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0


# Total population, N.
N = np.array([S0, I0, D0, A0, R0, T0, H0, E0])


# Time vector
t = np.linspace(0, 49, 50)


# Euler Integration
S, I, D, A, R, T, H, E = euler_integration(sidarthe_model, N, t, (alpha, beta, gamma, delta, epsi, theta, rho, mu))


# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Tested')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, E, label='Exposed')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
