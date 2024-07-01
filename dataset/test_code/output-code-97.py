import numpy as np
import matplotlib.pyplot as plt


def seirhd_model(y, t, beta, sigma, gamma, rho, delta):
    S, E, I, R, H, D = y
    N = S + E + I + R + H + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - rho * I - delta * I
    dRdt = gamma * I
    dHdt = rho * I
    dDdt = delta * I
    return [dSdt, dEdt, dIdt, dRdt, dHdt, dDdt]


def run_seirhd_model(S0, E0, I0, R0, H0, D0, N, beta, sigma, gamma, rho, delta, t_max, num_steps):
    y0 = [S0, E0, I0, R0, H0, D0]
    t = np.linspace(0, t_max, num_steps)
    
    result = odeint(seirhd_model, y0, t, args=(beta, sigma, gamma, rho, delta))
    
    S = result[:, 0]
    E = result[:, 1]
    I = result[:, 2]
    R = result[:, 3]
    H = result[:, 4]
    D = result[:, 5]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SEIRHD Model')
    plt.legend()
    plt.show()


