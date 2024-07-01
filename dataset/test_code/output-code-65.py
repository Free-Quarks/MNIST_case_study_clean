import numpy as np
import matplotlib.pyplot as plt


def model(t, y, beta, sigma, gamma, mu):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    lambd = beta * (I + A) / N
    dS = -lambd * S
    dI = lambd * S - sigma * I - gamma * I - mu * I
    dD = gamma * I
    dA = sigma * I - gamma * A
    dR = gamma * A
    dT = sigma * I
    dH = sigma * I
    dE = mu * I
    return [dS, dI, dD, dA, dR, dT, dH, dE]


def euler_integrate(model, y0, t, args=()):
    N = len(t)
    n = len(y0)
    y = np.zeros((N, n))
    y[0] = y0
    for i in range(N - 1):
        dt = t[i + 1] - t[i]
        dy = model(t[i], y[i], *args)
        y[i + 1] = y[i] + dt * dy
    return y


beta = 0.2
sigma = 0.1
gamma = 0.05
mu = 0.01

S_0 = 9900
I_0 = 100
D_0 = 0
A_0 = 0
R_0 = 0
T_0 = 0
H_0 = 0
E_0 = 0
y0 = [S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0]

t = np.linspace(0, 100, 1000)

y = euler_integrate(model, y0, t, args=(beta, sigma, gamma, mu))

plt.plot(t, y)
plt.legend(['S', 'I', 'D', 'A', 'R', 'T', 'H', 'E'])
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model Simulation')
plt.show()

