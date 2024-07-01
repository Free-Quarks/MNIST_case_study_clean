import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, T):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = T[1] - T[0]
    for t in range(1, len(T)):
        dSdt = -beta * S[-1] * I[-1] / N
        dIdt = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dRdt = gamma * I[-1]
        S.append(S[-1] + dSdt * dt)
        I.append(I[-1] + dIdt * dt)
        R.append(R[-1] + dRdt * dt)
    return S, I, R


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
T = np.linspace(0, 100, 1000)
S, I, R = SIR_model(S0, I0, R0, beta, gamma, T)

plt.plot(T, S, label='Susceptible')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
