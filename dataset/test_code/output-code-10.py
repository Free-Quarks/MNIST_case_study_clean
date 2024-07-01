import numpy as np


def seird_model(params, initial_conditions, t):
    # Extract parameter values
    beta = params['beta']
    gamma = params['gamma']
    sigma = params['sigma']
    mu = params['mu']
    N = params['N']
    
    # Extract initial conditions
    S, E, I, R, D = initial_conditions
    
    # Initialize arrays to store the results
    S_result = np.zeros(len(t))
    E_result = np.zeros(len(t))
    I_result = np.zeros(len(t))
    R_result = np.zeros(len(t))
    D_result = np.zeros(len(t))
    
    # Set initial values
    S_result[0] = S
    E_result[0] = E
    I_result[0] = I
    R_result[0] = R
    D_result[0] = D
    
    # Iterate over time steps
    for i in range(1, len(t)):
        # Compute new values
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I
        dDdt = mu * I
        
        # Update values
        S += dSdt
        E += dEdt
        I += dIdt
        R += dRdt
        D += dDdt
        
        # Store results
        S_result[i] = S
        E_result[i] = E
        I_result[i] = I
        R_result[i] = R
        D_result[i] = D
    
    # Return results
    return {'S': S_result, 'E': E_result, 'I': I_result, 'R': R_result, 'D': D_result}


# Define parameters
params = {
    'beta': 0.2,
    'gamma': 0.1,
    'sigma': 0.1,
    'mu': 0.01,
    'N': 10000
}

# Define initial conditions
initial_conditions = np.array([9999, 1, 0, 0, 0])

# Define time vector
t = np.linspace(0, 100, 1000)

# Simulate the model
results = seird_model(params, initial_conditions, t)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t, results['S'], label='Susceptible')
plt.plot(t, results['E'], label='Exposed')
plt.plot(t, results['I'], label='Infected')
plt.plot(t, results['R'], label='Recovered')
plt.plot(t, results['D'], label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.show()
