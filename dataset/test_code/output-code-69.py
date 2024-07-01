import numpy as np

def sidarthe_model(t, y, p):
    S, Ia, Is, It, Ic, Deaths, R, Rd, D, H, C = y
    beta, alpha, gamma, theta, delta, epsilon, rho, mu, eta, kappa, tau, psi, xi = p
    N = np.sum(y)
    dydt = [
        -beta * S * (Ia + theta * Is + delta * It + epsilon * Ic) / N,
        beta * S * (Ia + theta * Is + delta * It + epsilon * Ic) / N - alpha * Ia,
        alpha * (1 - rho) * Ia - (gamma + eta) * Is,
        alpha * rho * Ia - (gamma + kappa) * It,
        alpha * eta * Is + alpha * kappa * It - (mu + psi + xi) * Ic,
        xi * Ic,
        gamma * (1 - epsilon) * Ic + gamma * (1 - theta - delta) * It,
        gamma * theta * It,
        gamma * delta * It,
        alpha * eta * Is + alpha * kappa * It,
        alpha * (1 - rho) * Ia + alpha * rho * Ia + alpha * eta * Is + alpha * kappa * It,
    ]
    return dydt

