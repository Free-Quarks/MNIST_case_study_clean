import numpy as np
import matplotlib.pyplot as plt

# Define model parameters
beta = 0.2  # transmission rate
sigma = 0.1  # incubation rate
gamma = 0.05  # recovery rate

# Define initial conditions
S0 = 990  # initial susceptible population
E0 = 10  # initial exposed population
I0 = 0  # initial infected population
R0 = 0  # initial recovered population
N = S0 + E0 + I0 + R0  # total population

# Define time points
t0 = 0  # initial time
t_end = 100  # end time
h = 0.1  # step size

# Define SEIR model function

def seir_model(t, y):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])

# Implement the Runge-Kutta 4th order method

def rk4_step(f, h, t, y):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Initialize arrays to store results
t_points = np.arange(t0, t_end + h, h)
S_points = np.zeros_like(t_points)
E_points = np.zeros_like(t_points)
I_points = np.zeros_like(t_points)
R_points = np.zeros_like(t_points)

# Set initial conditions
S_points[0] = S0
E_points[0] = E0
I_points[0] = I0
R_points[0] = R0

# Simulate the SEIR model using RK4
for i in range(1, len(t_points)):
    t = t_points[i - 1]
    y = np.array([S_points[i - 1], E_points[i - 1], I_points[i - 1], R_points[i - 1]])
    y_next = rk4_step(seir_model, h, t, y)
    S_points[i] = y_next[0]
    E_points[i] = y_next[1]
    I_points[i] = y_next[2]
    R_points[i] = y_next[3]

# Plot the results
plt.figure()
plt.plot(t_points, S_points, label='Susceptible')
plt.plot(t_points, E_points, label='Exposed')
plt.plot(t_points, I_points, label='Infected')
plt.plot(t_points, R_points, label='Recovered')
plt.title('SEIR Model Simulation')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()
