import numpy as np
import matplotlib.pyplot as plt

def seirhd_model(N, E0, I0, R0, H0, D0, beta, sigma, gamma, delta, alpha, T):
    S0 = N - E0 - I0 - R0 - H0 - D0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    H = [H0]
    D = [D0]

    dt = T[1] - T[0]

    for t in range(len(T)-1):
        next_S = S[t] - (beta * S[t] * I[t]) / N
        next_E = E[t] + (beta * S[t] * I[t]) / N - sigma * E[t]
        next_I = I[t] + sigma * E[t] - (gamma * I[t]) - (delta * I[t]) - (alpha * I[t])
        next_R = R[t] + gamma * I[t]
        next_H = H[t] + delta * I[t]
        next_D = D[t] + alpha * I[t]

        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        H.append(next_H)
        D.append(next_D)

    return S, E, I, R, H, D


N = 1000000
E0 = 100
I0 = 10
R0 = 0
H0 = 0
D0 = 0
beta = 0.3
sigma = 1/5
gamma = 1/7
alpha = 1/14
T = np.linspace(0, 200, 200)

S, E, I, R, H, D = seirhd_model(N, E0, I0, R0, H0, D0, beta, sigma, gamma, alpha, T)

plt.plot(T, S, label='Susceptible')
plt.plot(T, E, label='Exposed')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.plot(T, H, label='Hospitalized')
plt.plot(T, D, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SEIRHD Model')
plt.legend()
plt.show()
