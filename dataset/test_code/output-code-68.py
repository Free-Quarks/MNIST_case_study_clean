import numpy as np


def sidarthe_model(params, initial_conditions, t):
    # Unpack parameters
    alpha, beta, gamma, delta, mu, omega, rho, sigma = params
    # Unpack initial conditions
    S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0 = initial_conditions
    # Initialize arrays
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    D = np.zeros_like(t)
    A = np.zeros_like(t)
    R = np.zeros_like(t)
    T = np.zeros_like(t)
    H = np.zeros_like(t)
    E = np.zeros_like(t)
    # Set initial conditions
    S[0] = S_0
    I[0] = I_0
    D[0] = D_0
    A[0] = A_0
    R[0] = R_0
    T[0] = T_0
    H[0] = H_0
    E[0] = E_0
    # Run simulation
    for i in range(1, len(t)):
        dSdt = -alpha * S[i-1] * I[i-1] - mu * S[i-1] + omega * R[i-1] - rho * S[i-1]
        dIdt = alpha * S[i-1] * I[i-1] - beta * I[i-1] - delta * I[i-1]
        dDdt = delta * I[i-1] - gamma * D[i-1]
        dAdt = rho * S[i-1] - sigma * A[i-1] - mu * A[i-1]
        dRdt = beta * I[i-1] + sigma * A[i-1] - omega * R[i-1] - mu * R[i-1]
        dTdt = gamma * D[i-1] + sigma * A[i-1] - mu * T[i-1]
        dHdt = alpha * S[i-1] * I[i-1] - beta * I[i-1] - delta * I[i-1] - mu * H[i-1]
        dEdt = mu * S[i-1] + mu * A[i-1] + mu * R[i-1] + mu * T[i-1] + mu * H[i-1]
        # Update states
        S[i] = S[i-1] + dSdt * (t[i] - t[i-1])
        I[i] = I[i-1] + dIdt * (t[i] - t[i-1])
        D[i] = D[i-1] + dDdt * (t[i] - t[i-1])
        A[i] = A[i-1] + dAdt * (t[i] - t[i-1])
        R[i] = R[i-1] + dRdt * (t[i] - t[i-1])
        T[i] = T[i-1] + dTdt * (t[i] - t[i-1])
        H[i] = H[i-1] + dHdt * (t[i] - t[i-1])
        E[i] = E[i-1] + dEdt * (t[i] - t[i-1])
    return S, I, D, A, R, T, H, E

