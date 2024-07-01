import numpy as np
import matplotlib.pyplot as plt


def seir_model(t, y, beta, gamma, sigma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_seir_model(init_vals, params, tmax):
    t = np.linspace(0, tmax, tmax+1)
    S_0, E_0, I_0, R_0 = init_vals
    beta, gamma, sigma = params
    res = spi.odeint(seir_model, [S_0, E_0, I_0, R_0], t, args=(beta, gamma, sigma))
    S, E, I, R = res.T
    return S, E, I, R


# Model parameters
init_vals = 0.99, 0.01, 0, 0
params = 0.2, 1/7, 1/5

# Run simulation
S, E, I, R = run_seir_model(init_vals, params, tmax=160)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(S, 'b', label='Susceptible')
plt.plot(E, 'y', label='Exposed')
plt.plot(I, 'r', label='Infected')
plt.plot(R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Fraction of population')
plt.title('SEIR Model Simulation')
plt.legend()
plt.show()

