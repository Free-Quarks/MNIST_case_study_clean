import numpy as np
import matplotlib.pyplot as plt

def sir_model(init_vals, params, t):
    S_0, I_0, R_0 = init_vals
    S, I, R = [S_0], [I_0], [R_0]
    alpha, beta = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (alpha * S[-1] * I[-1]) * dt
        next_I = I[-1] + (alpha * S[-1] * I[-1] - beta * I[-1]) * dt
        next_R = R[-1] + (beta * I[-1]) * dt
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, I, R]).T

# Define the initial values and parameters
init_vals = 0.99, 0.01, 0
params = 0.2, 0.1
t = np.linspace(0, 100, 100)

# Run the SIR model
results = sir_model(init_vals, params, t)

# Plot the results
plt.plot(t, results[:, 0], label='S(t)')
plt.plot(t, results[:, 1], label='I(t)')
plt.plot(t, results[:, 2], label='R(t)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Proportion of population')
plt.title('SIR Model')
plt.show()
