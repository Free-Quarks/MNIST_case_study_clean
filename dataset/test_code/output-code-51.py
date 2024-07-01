import numpy as np
from scipy.integrate import odeint

# Function that contains the model dynamics
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

# Initial conditions
S0 = 1000
I0 = 1
R0 = 0
y0 = [S0, I0, R0]

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 1000)

# Solve the differential equation
solution = odeint(sir_model, y0, t, args=(beta, gamma))

# Extract the solution components
S = solution[:, 0]
I = solution[:, 1]
R = solution[:, 2]
