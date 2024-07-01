import numpy as np
import matplotlib.pyplot as plt

def seir_model(N, beta, sigma, gamma, initial_infected, initial_exposed, initial_recovered, initial_susceptible, tmax, dt):
    # Define the SEIR model
    def model(t, y):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Initialize arrays to store the solution
    t_values = np.arange(0, tmax, dt)
    num_steps = len(t_values)
    S_values = np.zeros(num_steps)
    E_values = np.zeros(num_steps)
    I_values = np.zeros(num_steps)
    R_values = np.zeros(num_steps)

    # Set initial values
    S_values[0] = initial_susceptible
    E_values[0] = initial_exposed
    I_values[0] = initial_infected
    R_values[0] = initial_recovered

    # Solve the system of equations using Runge-Kutta 2nd order method
    for i in range(1, num_steps):
        t = t_values[i]
        y = S_values[i-1], E_values[i-1], I_values[i-1], R_values[i-1]
        k1 = dt * np.array(model(t, y))
        k2 = dt * np.array(model(t + dt, y + k1))
        y_next = y + 0.5 * (k1 + k2)
        S_values[i], E_values[i], I_values[i], R_values[i] = y_next

    # Plotting
    plt.plot(t_values, S_values, label='Susceptible')
    plt.plot(t_values, E_values, label='Exposed')
    plt.plot(t_values, I_values, label='Infected')
    plt.plot(t_values, R_values, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


# Example usage
N = 10000
beta = 0.3
sigma = 0.1
gamma = 0.05
initial_infected = 100
initial_exposed = 50
initial_recovered = 0
initial_susceptible = N - initial_infected - initial_exposed

tmax = 100
dt = 0.1

seir_model(N, beta, sigma, gamma, initial_infected, initial_exposed, initial_recovered, initial_susceptible, tmax, dt)
