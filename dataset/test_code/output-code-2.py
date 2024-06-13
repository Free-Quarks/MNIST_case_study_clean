import numpy as np
import matplotlib.pyplot as plt


def SIR_model(t, y, N, beta, gamma):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def RK2_step(t, y, h, N, beta, gamma):
    k1 = SIR_model(t, y, N, beta, gamma)
    k2 = SIR_model(t + h, [y[i] + h*k1[i] for i in range(len(y))], N, beta, gamma)
    y_next = [y[i] + h/2 * (k1[i] + k2[i]) for i in range(len(y))]
    return y_next


def simulate_SIR(S0, I0, R0, N, beta, gamma, t_max, h):
    num_steps = int(t_max / h)
    t = np.linspace(0, t_max, num_steps)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(num_steps-1):
        y = RK2_step(t[i], [S[i], I[i], R[i]], h, N, beta, gamma)
        S[i+1], I[i+1], R[i+1] = y
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

