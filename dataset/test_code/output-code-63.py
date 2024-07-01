import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def seird_model(t, y, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def simulate_seird_model(beta, sigma, gamma, mu, S0, E0, I0, R0, D0, duration):
    y0 = S0, E0, I0, R0, D0
    t_span = [0, duration]
    t_eval = np.linspace(0, duration, num=1000)
    result = solve_ivp(seird_model, t_span, y0, args=(beta, sigma, gamma, mu), t_eval=t_eval, method='RK45')
    return result.t, result.y[0], result.y[1], result.y[2], result.y[3], result.y[4]


beta = 0.5
sigma = 0.1
gamma = 0.2
mu = 0.02
S0 = 990
E0 = 10
I0 = 0
R0 = 0
D0 = 0
duration = 100

# Simulate SEIRD model
t, S, E, I, R, D = simulate_seird_model(beta, sigma, gamma, mu, S0, E0, I0, R0, D0, duration)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.show()
