import numpy as np
import matplotlib.pyplot as plt

# Define the model parameters
alpha = 0.2
beta = 0.4
gamma = 0.1
delta = 0.3
N = 1000
initial_conditions = [N-10, 10, 0, 0]

def seird_model(t, y):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I - delta * R
    return [dSdt, dEdt, dIdt, dRdt]

# Solve the model equations
t = np.linspace(0, 100, 1000)
solution = odeint(seird_model, initial_conditions, t)

# Plot the results
plt.plot(t, solution[:, 0], label='Susceptible')
plt.plot(t, solution[:, 1], label='Exposed')
plt.plot(t, solution[:, 2], label='Infected')
plt.plot(t, solution[:, 3], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SEIRD Model Simulation')
plt.show()
