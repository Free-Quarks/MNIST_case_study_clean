import numpy as np
import matplotlib.pyplot as plt

def seird_model(initial_conditions, parameters, time):
    # Get initial values
    S, E, I, R, D = initial_conditions
    # Get parameters
    beta, sigma, gamma, mu = parameters
    # Initialize arrays
    S_values = [S]
    E_values = [E]
    I_values = [I]
    R_values = [R]
    D_values = [D]
    t_values = [0]
    # Euler's method
    for t in time:
        # Calculate derivatives
        dS_dt = -beta * S * I
        dE_dt = beta * S * I - sigma * E
        dI_dt = sigma * E - gamma * I - mu * I
        dR_dt = gamma * I
        dD_dt = mu * I
        # Update values
        S += dS_dt
        E += dE_dt
        I += dI_dt
        R += dR_dt
        D += dD_dt
        # Append values
        S_values.append(S)
        E_values.append(E)
        I_values.append(I)
        R_values.append(R)
        D_values.append(D)
        t_values.append(t)
    # Return results
    return S_values, E_values, I_values, R_values, D_values, t_values

# Initial conditions
initial_conditions = (100000, 100, 10, 0, 0)
# Model parameters
parameters = (0.5, 0.2, 0.1, 0.01)
# Time vector
time = np.linspace(0, 100, 100)

# Run model
S, E, I, R, D, t = seird_model(initial_conditions, parameters, time)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
