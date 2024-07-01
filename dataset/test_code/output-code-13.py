import numpy as np
import matplotlib.pyplot as plt

def seird_model(beta, sigma, gamma, delta, N, I0, E0, R0, D0, t_end, dt):
    # Define the SEIRD model dynamics
    def seird(y, t, beta, sigma, gamma, delta, N):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - (gamma + delta) * I
        dRdt = gamma * I
        dDdt = delta * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    # Set initial conditions
    S0 = N - I0 - E0 - R0 - D0
    y0 = S0, E0, I0, R0, D0
    t = np.linspace(0, t_end, int(t_end/dt))

    # Integrate the SEIRD equations
    sol = odeint(seird, y0, t, args=(beta, sigma, gamma, delta, N))
    S, E, I, R, D = sol.T

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIRD Model')
    plt.legend()
    plt.show()


# Run the SEIRD model
seird_model(beta=0.2, sigma=0.5, gamma=0.1, delta=0.05, N=1000, I0=10, E0=10, R0=0, D0=0, t_end=100, dt=0.1)
