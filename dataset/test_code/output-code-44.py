import numpy as np


def model_function(t, y, params):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    beta, gamma, delta, alpha, theta, kappa, omega, rho, sigma, phi, psi, xi = params
    
    dSdt = -beta * S * (I + alpha*A) / N
    dIdt = beta * S * (I + alpha*A) / N - (gamma + delta + sigma) * I
    dDdt = delta * I - (omega + phi) * D
    dAdt = alpha * beta * S * (I + alpha*A) / N - (theta + rho) * A
    dRdt = gamma * I + omega * D - kappa * R
    dTdt = theta * A - (kappa + psi + xi) * T
    dHdt = psi * T - xi * H
    dEdt = rho * A + phi * D + xi * T
    
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def run_simulation(S0, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, alpha, theta, kappa, omega, rho, sigma, phi, psi, xi, tmax, dt):
    t = np.arange(0, tmax + dt, dt)
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]
    params = [beta, gamma, delta, alpha, theta, kappa, omega, rho, sigma, phi, psi, xi]
    
    result = odeint(model_function, y0, t, args=(params,))
    S, I, D, A, R, T, H, E = result.T
    
    return S, I, D, A, R, T, H, E


# Example usage

S0 = 1000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

beta = 0.2
gamma = 0.1
delta = 0.1
alpha = 0.2
theta = 0.1
kappa = 0.1
omega = 0.1
rho = 0.1
sigma = 0.1
phi = 0.1
psi = 0.1
xi = 0.1

tmax = 100
dt = 0.1

S, I, D, A, R, T, H, E = run_simulation(S0, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, alpha, theta, kappa, omega, rho, sigma, phi, psi, xi, tmax, dt)

# Plot results
import matplotlib.pyplot as plt

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Dead')
plt.plot(A, label='Active')
plt.plot(R, label='Recovered')
plt.plot(T, label='Serious')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model Simulation')
plt.show()
