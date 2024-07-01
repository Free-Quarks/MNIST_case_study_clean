import numpy as np
import matplotlib.pyplot as plt


# Model Parameters
beta = 0.2
sigma = 0.1
alpha = 0.1
rho = 0.1
lambda_ = 0.1
gamma = 0.05
epsilon = 0.01
N = 1000


# Initial Conditions
S0 = N-1
I0 = 1
D0 = 0
R0 = 0
E0 = 0
t0 = 0


# Model

def dSdt(t, S, E, I, D, R, A):
    return -beta*S*(I + alpha*A)/N

def dEdt(t, S, E, I, D, R, A):
    return beta*S*(I + alpha*A)/N - sigma*E

def dIdt(t, S, E, I, D, R, A):
    return sigma*E - (rho + gamma)*I

def dDdt(t, S, E, I, D, R, A):
    return rho*I - lambda_*D

def dRdt(t, S, E, I, D, R, A):
    return gamma*I + lambda_*D

def dAdt(t, S, E, I, D, R, A):
    return epsilon*D - alpha*A


# Runge-Kutta 2nd Order Method

def RK2_step(t, S, E, I, D, R, A, h):
    k1_S = h*dSdt(t, S, E, I, D, R, A)
    k1_E = h*dEdt(t, S, E, I, D, R, A)
    k1_I = h*dIdt(t, S, E, I, D, R, A)
    k1_D = h*dDdt(t, S, E, I, D, R, A)
    k1_R = h*dRdt(t, S, E, I, D, R, A)
    k1_A = h*dAdt(t, S, E, I, D, R, A)

    t_half = t + h/2
    S_half = S + k1_S/2
    E_half = E + k1_E/2
    I_half = I + k1_I/2
    D_half = D + k1_D/2
    R_half = R + k1_R/2
    A_half = A + k1_A/2

    k2_S = h*dSdt(t_half, S_half, E_half, I_half, D_half, R_half, A_half)
    k2_E = h*dEdt(t_half, S_half, E_half, I_half, D_half, R_half, A_half)
    k2_I = h*dIdt(t_half, S_half, E_half, I_half, D_half, R_half, A_half)
    k2_D = h*dDdt(t_half, S_half, E_half, I_half, D_half, R_half, A_half)
    k2_R = h*dRdt(t_half, S_half, E_half, I_half, D_half, R_half, A_half)
    k2_A = h*dAdt(t_half, S_half, E_half, I_half, D_half, R_half, A_half)

    t_new = t + h
    S_new = S + k2_S
    E_new = E + k2_E
    I_new = I + k2_I
    D_new = D + k2_D
    R_new = R + k2_R
    A_new = A + k2_A

    return t_new, S_new, E_new, I_new, D_new, R_new, A_new


# Simulation

def simulate(t0, S0, E0, I0, D0, R0, A0, beta, sigma, alpha, rho, lambda_, gamma, epsilon, N, dt, steps):
    t_values = np.zeros(steps+1)
    S_values = np.zeros(steps+1)
    E_values = np.zeros(steps+1)
    I_values = np.zeros(steps+1)
    D_values = np.zeros(steps+1)
    R_values = np.zeros(steps+1)
    A_values = np.zeros(steps+1)

    t_values[0] = t0
    S_values[0] = S0
    E_values[0] = E0
    I_values[0] = I0
    D_values[0] = D0
    R_values[0] = R0
    A_values[0] = A0

    for i in range(steps):
        t, S, E, I, D, R, A = RK2_step(t_values[i], S_values[i], E_values[i], I_values[i], D_values[i], R_values[i], A_values[i], dt)
        t_values[i+1] = t
        S_values[i+1] = S
        E_values[i+1] = E
        I_values[i+1] = I
        D_values[i+1] = D
        R_values[i+1] = R
        A_values[i+1] = A

    return t_values, S_values, E_values, I_values, D_values, R_values, A_values


# Run simulation

t_values, S_values, E_values, I_values, D_values, R_values, A_values = simulate(t0, S0, E0, I0, D0, R0, A0, beta, sigma, alpha, rho, lambda_, gamma, epsilon, N, 0.1, 100)


# Plotting

plt.plot(t_values, S_values, label='S')
plt.plot(t_values, E_values, label='E')
plt.plot(t_values, I_values, label='I')
plt.plot(t_values, D_values, label='D')
plt.plot(t_values, R_values, label='R')
plt.plot(t_values, A_values, label='A')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model Simulation')
plt.legend()
plt.show()
