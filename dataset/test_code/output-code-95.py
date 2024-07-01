import numpy as np
import matplotlib.pyplot as plt


def seirhd_model(N, beta, sigma, gamma, mu, delta, I0, E0, R0, H0, D0, t_max):
    S0 = N - I0 - E0 - R0 - H0 - D0
    
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    H = [H0]
    D = [D0]
    
    dt = 0.1
    t = np.arange(0, t_max + dt, dt)
    
    for i in range(len(t) - 1):
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i] - mu * I[i] - delta * I[i]
        dRdt = gamma * I[i]
        dHdt = mu * I[i]
        dDdt = delta * I[i]
        
        S.append(S[i] + dt * dSdt)
        E.append(E[i] + dt * dEdt)
        I.append(I[i] + dt * dIdt)
        R.append(R[i] + dt * dRdt)
        H.append(H[i] + dt * dHdt)
        D.append(D[i] + dt * dDdt)
    
    return t, S, E, I, R, H, D


N = 100000
beta = 0.2
sigma = 0.3
gamma = 0.1
mu = 0.05
delta = 0.02
I0 = 100
E0 = 50
R0 = 0
H0 = 0
D0 = 0
t_max = 100


t, S, E, I, R, H, D = seirhd_model(N, beta, sigma, gamma, mu, delta, I0, E0, R0, H0, D0, t_max)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRHD Model')
plt.legend()
plt.show()
