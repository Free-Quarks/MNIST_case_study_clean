import numpy as np

def seird_model(beta, sigma, gamma, mu, N, initial_conditions, t_start, t_end, t_step):
    S_0, E_0, I_0, R_0, D_0 = initial_conditions
    S, E, I, R, D = [S_0], [E_0], [I_0], [R_0], [D_0]
    t = np.arange(t_start, t_end + t_step, t_step)
    h = t_step

    for time in t[:-1]:
        S_t, E_t, I_t, R_t, D_t = S[-1], E[-1], I[-1], R[-1], D[-1]
        S_k1 = -(beta * S_t * I_t) / N
        E_k1 = (beta * S_t * I_t) / N - sigma * E_t
        I_k1 = sigma * E_t - (gamma + mu) * I_t
        R_k1 = gamma * I_t
        D_k1 = mu * I_t

        S_k2 = -(beta * (S_t + S_k1 * (h/2)) * (I_t + I_k1 * (h/2))) / N
        E_k2 = (beta * (S_t + S_k1 * (h/2)) * (I_t + I_k1 * (h/2))) / N - sigma * (E_t + E_k1 * (h/2))
        I_k2 = sigma * (E_t + E_k1 * (h/2)) - (gamma + mu) * (I_t + I_k1 * (h/2))
        R_k2 = gamma * (I_t + I_k1 * (h/2))
        D_k2 = mu * (I_t + I_k1 * (h/2))

        S_k3 = -(beta * (S_t + S_k2 * h) * (I_t + I_k2 * h)) / N
        E_k3 = (beta * (S_t + S_k2 * h) * (I_t + I_k2 * h)) / N - sigma * (E_t + E_k2 * h)
        I_k3 = sigma * (E_t + E_k2 * h) - (gamma + mu) * (I_t + I_k2 * h)
        R_k3 = gamma * (I_t + I_k2 * h)
        D_k3 = mu * (I_t + I_k2 * h)

        S.append(S_t + (h/6) * (S_k1 + 4*S_k2 + S_k3))
        E.append(E_t + (h/6) * (E_k1 + 4*E_k2 + E_k3))
        I.append(I_t + (h/6) * (I_k1 + 4*I_k2 + I_k3))
        R.append(R_t + (h/6) * (R_k1 + 4*R_k2 + R_k3))
        D.append(D_t + (h/6) * (D_k1 + 4*D_k2 + D_k3))

    return S, E, I, R, D
