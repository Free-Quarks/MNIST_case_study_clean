import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S_0, I_0, R_0, num_steps):
    # Initialize arrays to store values
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    t = np.arange(num_steps)
    
    # Set initial values
    S[0] = S_0
    I[0] = I_0
    R[0] = R_0
    
    # Euler method
    for i in range(1, num_steps):
        dS_dt = -beta * S[i-1] * I[i-1]
        dI_dt = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        dR_dt = gamma * I[i-1]
        
        S[i] = S[i-1] + dS_dt
        I[i] = I[i-1] + dI_dt
        R[i] = R[i-1] + dR_dt
    
    return S, I, R

# Example usage
beta = 0.2
gamma = 0.1
S_0 = 990
I_0 = 10
R_0 = 0
num_steps = 100

S, I, R = SIR_model(beta, gamma, S_0, I_0, R_0, num_steps)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time step')
plt.ylabel('Population')
plt.legend()
plt.show()
