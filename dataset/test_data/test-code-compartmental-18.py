import numpy as np

def get_beta(intrinsic_growth_rate, gamma, s_c, relative_contact_rate):
    inv_contact_rate = 1.0 - relative_contact_rate
    updated_growth_rate = intrinsic_growth_rate + gamma
    beta = updated_growth_rate / s_c * inv_contact_rate
    return beta

def get_growth_rate(doubling_time):
    growth_rate = np.log(2) / doubling_time
    return growth_rate

def sir(s_c, i_c, r_c, beta, gamma, n):
    dsdt = -beta * s_c * i_c / n
    didt = beta * s_c * i_c / n - gamma * i_c
    drdt = gamma * i_c
    s_c += dsdt
    i_c += didt
    r_c += drdt
    return s_c, i_c, r_c

def sim_sir(s_c, i_c, r_c, gamma, i_day, N_p, N_t, betas, days, T, S, EI, I, R):
    n = s_c + i_c + r_c
    d = i_day

    idx = 0
    for p_idx in range(N_p):
        beta = betas[p_idx]
        N_d = days[p_idx]
        for _ in range(N_d):
            T[idx] = d
            S[idx] = s_c
            I[idx] = i_c
            R[idx] = r_c
            idx += 1
            s_c, i_c, r_c = sir(s_c, i_c, r_c, beta, gamma, n)
            d += 1

    T[idx] = d
    S[idx] = s_c
    I[idx] = i_c
    R[idx] = r_c

    for t_idx in range(N_t):
        total_inf = I[t_idx] + R[t_idx]
        EI[t_idx] = total_inf

def main():
    s_c = 1000
    i_c = 0
    r_c = 0
    doubling_time = 0
    growth_rate = 0
    gamma = 1.0 / 14
    relative_contact_rate = 0.05
    i_day = 17
    n_days = 20
    N_p = 3
    N_t = 121
    policy_betas = np.zeros(N_p)
    policy_days = np.zeros(N_p, dtype=int)
    T = np.zeros(N_t, dtype=int)
    S = np.zeros(N_t)
    EI = np.zeros(N_t)
    I = np.zeros(N_t)
    R = np.zeros(N_t)

    for p_idx in range(N_p):
        doubling_time = (p_idx) * 5.0
        growth_rate = get_growth_rate(doubling_time)
        beta = get_beta(growth_rate, gamma, s_c, relative_contact_rate)
        policy_betas[p_idx] = beta
        days_for_policy = n_days * (p_idx + 1)
        policy_days[p_idx] = days_for_policy

    sim_sir(s_c, i_c, r_c, gamma, i_day, N_p, N_t, policy_betas, policy_days, T, S, EI, I, R)

    print(s_c, i_c, r_c)
    print(EI)

if __name__ == "__main__":
    main()