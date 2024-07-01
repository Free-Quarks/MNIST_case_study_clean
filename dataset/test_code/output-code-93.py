import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(y, t, alpha, beta, gamma, kappa, theta, delta, rho):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    
    dSdt = -alpha * S * (I + delta * A) / N
    dIdt = alpha * S * (I + delta * A) / N - (beta + gamma + kappa) * I
    dDdt = delta * theta * alpha * S * A / N
    dAdt = delta * (1 - theta) * alpha * S * A / N - (gamma + kappa) * A
    dRdt = gamma * (I + A) - rho * R
    dTdt = kappa * (I + A)
    dHdt = beta * I - gamma * H
    dEdt = rho * R
    
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def run_simulation(initial_conditions, params, t_max, dt):
    t = np.arange(0, t_max, dt)
    
    sol = odeint(sidarthe_model, initial_conditions, t, args=tuple(params))
    
    S, I, D, A, R, T, H, E = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4], sol[:, 5], sol[:, 6], sol[:, 7]
    
    plt.plot(t, S, label='S')
    plt.plot(t, I, label='I')
    plt.plot(t, D, label='D')
    plt.plot(t, A, label='A')
    plt.plot(t, R, label='R')
    plt.plot(t, T, label='T')
    plt.plot(t, H, label='H')
    plt.plot(t, E, label='E')
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


