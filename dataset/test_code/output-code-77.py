import numpy as np
import matplotlib.pyplot as plt


# Function that contains the model dynamics

def SIR_model(y, t, N, beta, gamma):
   S, I, R = y
   dSdt = -beta * S * I / N
   dIdt = beta * S * I / N - gamma * I
   dRdt = gamma * I
   return dSdt, dIdt, dRdt


# Parameters
N = 1000
beta = 0.2
gamma = 0.1

# Initial conditions
S0, I0, R0 = N-1, 1, 0

# Time vector
t = np.linspace(0, 100, 100)

# Solve the differential equations
y = S0, I0, R0
solution = odeint(SIR_model, y, t, args=(N, beta, gamma))
S, I, R = solution.T

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
