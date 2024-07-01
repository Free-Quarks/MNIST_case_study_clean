def sidarthe_model(y, t, params):
    S, I, D, A, R, T, H, E = y
    beta, gamma, delta, alpha, rho, theta, mu, lambda_1, lambda_2, k_1, k_2, kappa, psi_1, psi_2, psi_3 = params
    N = S + I + D + A + R + T + H + E

    dS = - beta * S * (I + rho * A) / N
    dI = beta * S * (I + rho * A) / N - (gamma + delta + alpha + mu) * I
    dD = delta * I - (theta + lambda_1 + k_1 + psi_1) * D
    dA = alpha * I - (lambda_2 + k_2 + psi_2) * A
    dR = gamma * I - (kappa + psi_3) * R
    dT = theta * D + lambda_1 * D + k_1 * D + lambda_2 * A + k_2 * A + kappa * R
    dH = psi_1 * D + psi_2 * A + psi_3 * R
    dE = mu * I

    return [dS, dI, dD, dA, dR, dT, dH, dE]


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def simulate_sidarthe_model(S0, I0, D0, A0, R0, T0, H0, E0, params, t_max):
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]
    t_span = (0, t_max)
    t_eval = np.arange(t_span[0], t_span[1] + 1)

    res = solve_ivp(sidarthe_model, t_span, y0, args=(params,), t_eval=t_eval, method='RK45')

    t = res.t
    S, I, D, A, R, T, H, E = res.y

    plt.figure(figsize=(12,6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deceased')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Total Active')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exposed')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.show()


params = [beta, gamma, delta, alpha, rho, theta, mu, lambda_1, lambda_2, k_1, k_2, kappa, psi_1, psi_2, psi_3]
simulate_sidarthe_model(S0, I0, D0, A0, R0, T0, H0, E0, params, t_max)
