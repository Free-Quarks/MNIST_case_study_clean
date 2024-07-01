import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    # Initial conditions
    S0 = N - I0 - E0 - R0
    Y0 = [S0, E0, I0, R0]
    
    # Time vector
    t = np.linspace(0, T, num=T+1)
    
    # Step size
    h = t[1] - t[0]
    
    # Store the results
    S = np.zeros(T+1)
    E = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    
    S[0] = Y0[0]
    E[0] = Y0[1]
    I[0] = Y0[2]
    R[0] = Y0[3]
    
    # Runge-Kutta 3rd order method
    for i in range(T):
        k1 = h * (-beta * S[i] * I[i] / N)
        k2 = h * (beta * S[i] * I[i] / N - sigma * E[i])
        k3 = h * (sigma * E[i] - gamma * I[i])
        k4 = h * (gamma * I[i])
        
        S[i+1] = S[i] + k1/6 + k2/6 + k3/6
        E[i+1] = E[i] + k2/6 + k3/6 + k4/6
        I[i+1] = I[i] + k3/6 + k4/6
        R[i+1] = R[i] + k4/6
    
    return t, S, E, I, R

# Example usage
beta = 0.2
gamma = 0.1
sigma = 0.05
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = 100

t, S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SEIR Model')
plt.legend()
plt.show()
