import numpy as np
import matplotlib.pyplot as plt


def SEIRHD(x, t, beta, sigma, gamma, eta, mu):
    S, E, I, R, H, D = x
    N = S + E + I + R + H + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + eta + mu) * I
    dRdt = gamma * I
    dHdt = eta * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]


def simulate_and_plot_SEIRHD(S0, E0, I0, R0, H0, D0, beta, sigma, gamma, eta, mu, t_max):
    y0 = [S0, E0, I0, R0, H0, D0]
    t = np.linspace(0, t_max, t_max + 1)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = SEIRHD(y[i-1], t[i-1], beta, sigma, gamma, eta, mu)
        k2 = SEIRHD(y[i-1] + k1 * (t[i] - t[i-1]), t[i], beta, sigma, gamma, eta, mu)
        y[i] = y[i-1] + (t[i] - t[i-1]) * (k1 + k2) / 2

    plt.figure()
    plt.plot(t, y[:, 0], label='Susceptible')
    plt.plot(t, y[:, 1], label='Exposed')
    plt.plot(t, y[:, 2], label='Infected')
    plt.plot(t, y[:, 3], label='Recovered')
    plt.plot(t, y[:, 4], label='Hospitalized')
    plt.plot(t, y[:, 5], label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Simulation of SEIRHD Model')
    plt.show()


