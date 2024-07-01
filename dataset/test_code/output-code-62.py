import numpy as np
import matplotlib.pyplot as plt

def seird_model(t, y, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def run_seird_model(init_conditions, beta, sigma, gamma, mu, t_max):
    t = np.linspace(0, t_max, t_max + 1)
    sol = odeint(seird_model, init_conditions, t, args=(beta, sigma, gamma, mu))
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='S')
    plt.plot(t, E, label='E')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.plot(t, D, label='D')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIRD Model')
    plt.legend()
    plt.show()


# Example usage
init_conditions = [999, 1, 0, 0, 0]
beta = 0.2
sigma = 1/5
gamma = 1/7
mu = 1/14
t_max = 100

run_seird_model(init_conditions, beta, sigma, gamma, mu, t_max)
