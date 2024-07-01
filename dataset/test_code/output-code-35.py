import numpy as np
import matplotlib.pyplot as plt


def seird_model(init_vals, params, t):
    S_0, E_0, I_0, R_0, D_0 = init_vals
    S, E, I, R, D = [S_0], [E_0], [I_0], [R_0], [D_0]
    alpha, beta, gamma, delta = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (alpha * S[-1] * I[-1]) * dt
        next_E = E[-1] + (alpha * S[-1] * I[-1] - beta * E[-1]) * dt
        next_I = I[-1] + (beta * E[-1] - gamma * I[-1] - delta * I[-1]) * dt
        next_R = R[-1] + (gamma * I[-1]) * dt
        next_D = D[-1] + (delta * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        D.append(next_D)
    return S, E, I, R, D


# Initial values
init_vals = 1 - 1e-6, 1e-6, 0, 0, 0

# Parameters
params = 0.35, 0.2, 0.1, 0.05

# Time vector
t = np.linspace(0, 100, 100)

# Run the model
S, E, I, R, D = seird_model(init_vals, params, t)

# Plot the result
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Proportion of Population')
plt.title('SEIRD Model')
plt.legend()
plt.show()
