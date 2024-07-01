import numpy as np
import matplotlib.pyplot as plt

def seirhd_model(t, y, params):
    S, E, I, R, H, D = y
    beta, epsilon, gamma, delta, mu = params
    N = S + E + I + R + H + D

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - epsilon * E
    dIdt = epsilon * E - (gamma + delta + mu) * I
    dRdt = gamma * I
    dHdt = delta * I
    dDdt = mu * I

    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt


def euler_method(model, t0, tf, y0, params, dt):
    t = np.arange(t0, tf+dt, dt)
    num_steps = len(t)

    y = np.zeros((num_steps, len(y0)))
    y[0] = y0

    for i in range(1, num_steps):
        y[i] = y[i-1] + dt * model(t[i-1], y[i-1], params)

    return t, y


# Set initial conditions and parameters
t0 = 0
tf = 100
y0 = [1000, 1, 0, 0, 0, 0]
params = [0.3, 0.1, 0.05, 0.02, 0.01]
dt = 0.1

# Run simulation using euler_method
t, y = euler_method(seirhd_model, t0, tf, y0, params, dt)

# Plot results
plt.plot(t, y[:, 0], label='S')
plt.plot(t, y[:, 1], label='E')
plt.plot(t, y[:, 2], label='I')
plt.plot(t, y[:, 3], label='R')
plt.plot(t, y[:, 4], label='H')
plt.plot(t, y[:, 5], label='D')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
