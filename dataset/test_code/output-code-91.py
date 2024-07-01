import numpy as np
from scipy.integrate import odeint


# Function that defines the model equations

def sidarthe_model(y, t, alpha, beta, gamma, delta, epsilon, rho, theta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    
    dSdt = -alpha*S*(I+D+A+R+T+H+E)/N
    dIdt = alpha*S*(I+D+A+R+T+H+E)/N - beta*I
    dDdt = delta*theta*I - gamma*D
    dAdt = (1-delta)*theta*I - epsilon*A
    dRdt = gamma*D
    dTdt = rho*epsilon*A
    dHdt = (1-rho)*epsilon*A
    dEdt = -delta*theta*I
    
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


# Function to solve the model equations

def solve_model(initial_conditions, params, t):
    alpha, beta, gamma, delta, epsilon, rho, theta = params
    solution = odeint(sidarthe_model, initial_conditions, t, args=(alpha, beta, gamma, delta, epsilon, rho, theta))
    return solution


# Example usage

initial_conditions = [100000, 1, 0, 0, 0, 0, 0, 0]  # Initial conditions for S, I, D, A, R, T, H, E
params = (0.1, 0.2, 0.05, 0.3, 0.4, 0.6, 0.7)  # Parameter values for alpha, beta, gamma, delta, epsilon, rho, theta
t = np.linspace(0, 100, 100)  # Time vector

solution = solve_model(initial_conditions, params, t)

print(solution)

