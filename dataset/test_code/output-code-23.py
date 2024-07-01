import numpy as np


def seirhd_model(t, y, params):
    S, E, I, R, H, D = y
    beta, sigma, gamma, delta, mu = params
    N = S + E + I + R + H + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + delta + mu) * I
    dRdt = gamma * I
    dHdt = delta * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]


def rk3_solver(model, t_span, initial_conditions, params, dt):
    t_start, t_end = t_span
    t = np.arange(t_start, t_end+dt, dt)
    num_steps = len(t)
    num_compartments = len(initial_conditions)
    y = np.zeros((num_steps, num_compartments))
    y[0, :] = initial_conditions

    for i in range(num_steps-1):
        k1 = model(t[i], y[i, :], params)
        k2 = model(t[i] + dt/2, y[i, :] + dt/2 * k1, params)
        k3 = model(t[i] + dt, y[i, :] - dt * k1 + 2 * dt * k2, params)
        y[i+1, :] = y[i, :] + dt/6 * (k1 + 4 * k2 + k3)

    return t, y


def plot_seirhd(t, y):
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[:, 0], label='S')
    plt.plot(t, y[:, 1], label='E')
    plt.plot(t, y[:, 2], label='I')
    plt.plot(t, y[:, 3], label='R')
    plt.plot(t, y[:, 4], label='H')
    plt.plot(t, y[:, 5], label='D')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SEIRHD Model')
    plt.legend()
    plt.show()
