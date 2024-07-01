import numpy as np
import matplotlib.pyplot as plt


def seir_model(N, beta, gamma, sigma, I0, E0, R0, T):
    S0 = N - I0 - E0 - R0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    for t in range(1, T):
        S.append(S[t-1] - beta*S[t-1]*I[t-1]/N)
        E.append(E[t-1] + beta*S[t-1]*I[t-1]/N - sigma*E[t-1])
        I.append(I[t-1] + sigma*E[t-1] - gamma*I[t-1])
        R.append(R[t-1] + gamma*I[t-1])
    return S, E, I, R


N = 10000
beta = 0.6
gamma = 0.2
sigma = 0.1
I0 = 1
E0 = 0
R0 = 0
T = 100

S, E, I, R = seir_model(N, beta, gamma, sigma, I0, E0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

