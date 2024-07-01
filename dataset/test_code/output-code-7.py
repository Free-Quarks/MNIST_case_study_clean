import numpy as np


def seir_model(t, y, params):
    # unpack the state variables
    S, E, I, R = y
    # unpack the parameters
    beta, sigma, gamma = params
    # compute the derivatives
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def rk2_step(t, y, h, f, params):
    # compute the intermediate state
    k1 = f(t, y, params)
    k2 = f(t + h, y + h * k1, params)
    # update the state variables
    y_next = y + 0.5 * h * (k1 + k2)
    return y_next


def simulate_seir_model(seir_model, initial_conditions, params, t_end, h):
    # initialize the time points and state variables
    t_values = np.arange(0, t_end, h)
    num_steps = len(t_values)
    num_variables = len(initial_conditions)
    y_values = np.zeros((num_steps, num_variables))
    # set the initial conditions
    y_values[0] = initial_conditions
    # simulate the model
    for i in range(1, num_steps):
        y_values[i] = rk2_step(t_values[i-1], y_values[i-1], h, seir_model, params)
    return t_values, y_values
