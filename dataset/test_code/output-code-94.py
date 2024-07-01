import numpy as np


def sidarthe_model(t, y, p):
    # Unpack the state variables
    S, I, D, A, R, T, H, E = y
    
    # Unpack the parameters
    beta, gammaI, gammaA, gammaR, gammaT, alpha, delta, rho, theta, kappa = p
    
    # Calculate the derivatives
    dSdt = -beta * S * (I + rho * A) / (S + I + D + A + R + T + H + E)
    dIdt = beta * S * (I + rho * A) / (S + I + D + A + R + T + H + E) - (1 - alpha) * gammaI * I - alpha * delta * I
    dDdt = delta * alpha * I - (1 - theta) * gammaR * D - theta * gammaT * D
    dAdt = delta * (1 - alpha) * I - gammaA * A
    dRdt = (1 - theta) * gammaR * D
    dTdt = theta * gammaT * D
    dHdt = rho * (1 - delta) * alpha * I - kappa * H
    dEdt = kappa * H
    
    # Return the derivatives
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def rk4_step(t, y, p, h):
    # Calculate the derivatives at the beginning of the time step
    k1 = sidarthe_model(t, y, p)
    
    # Calculate the derivatives at the midpoint of the time step
    k2 = sidarthe_model(t + h/2, y + h/2 * k1, p)
    
    # Calculate the derivatives at the midpoint of the time step
    k3 = sidarthe_model(t + h/2, y + h/2 * k2, p)
    
    # Calculate the derivatives at the end of the time step
    k4 = sidarthe_model(t + h, y + h * k3, p)
    
    # Calculate the next state variables
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Return the next state variables
    return y_next
