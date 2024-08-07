import numpy as np

def get_vaccine_rate(t):
    # Placeholder function - Replace with the actual implementation
    vr1 = 0.01
    vr2 = 0.01
    return vr1, vr2

def get_beta(t):
    # Placeholder function - Replace with the actual implementation
    return 1.0

def svair(t, y, beta, beta_v1, beta_v2, beta_R, ai_beta_ratio, gamma, nu_v1, nu_v2, nu_R, ai, ai_V, ai_R, mu, mu_I, mu_IV, 
          new_beta, new_beta_v1, new_beta_v2, new_beta_R, new_ai, t_new_voc):
    # Retrieve current populations
    nv = 2  # number of viruses simulated
    ind = 0
    S = y[ind]
    ind += 1
    SVR = y[ind]
    ind += 1
    V1 = y[ind:ind+2]
    ind += 2
    V2 = y[ind:ind+2]
    ind += 2
    I = y[ind:ind+nv]
    ind += nv
    IV = y[ind:ind+nv]
    ind += nv
    IR = y[ind:ind+nv]
    ind += nv
    A = y[ind:ind+nv]
    ind += nv
    AR = y[ind:ind+nv]
    ind += nv
    R = y[ind:ind+nv]
    ind += nv
    R2 = y[ind]
    ind += 1

    # Get time-dependent parameters
    vr1, vr2 = get_vaccine_rate(t)
    if t >= t_new_voc:  # switch WT to new VOC
        beta[0] = new_beta
        beta_v1[:, 0] = new_beta_v1
        beta_v2[:, 0] = new_beta_v2
        beta_R[0] = new_beta_R
        ai[0] = new_ai
    
    beta_scale = get_beta(t)
    beta *= beta_scale
    beta_v1 *= beta_scale
    beta_v2 *= beta_scale
    beta_R *= beta_scale

    # Total infectious population
    I_total = I + IV + IR + ai_beta_ratio * (A + AR)
    # Compute infection of recovered from another variant
    mm = np.ones((nv+1, nv+1)) - np.eye(nv+1)
    mv = mm * np.tile(R, (nv+1, 1)).T
    Rv = np.sum(mv, axis=1)

    # Compute time derivatives
    dSdt = - np.sum(beta * S * I_total) - np.sum(vr1 * S) + mu * (1 - S)
    dSVRdt = nu_v1 * np.sum(V1) + nu_v2 * np.sum(V2) + np.sum(nu_R * R) + nu_R * R2 - np.sum(beta * SVR * I_total) - mu * SVR
    dV1dt = vr1 * (S + np.sum(A)) - vr2 * V1 - nu_v1 * V1 - np.sum(beta_v1 * (V1[:, np.newaxis] * I_total), axis=1) - mu * V1
    dV2dt = vr2 * V1 - nu_v2 * V2 - np.sum(beta_v2 * (V2[:, np.newaxis] * I_total), axis=1) - mu * V2
    dIdt = (1 - ai) * (beta * S * I_total + beta * SVR * I_total) - gamma * I - mu_I * I
    dIVdt = (1 - ai_V) * (np.sum(beta_v1 * (V1[:, np.newaxis] * I_total), axis=1) + np.sum(beta_v2 * (V2[:, np.newaxis] * I_total), axis=1)) - gamma * IV - mu_IV * IV
    dIRdt = (1 - ai_R) * (beta_R * Rv * I_total) - gamma * IR - mu * IR
    dAdt = ai * (beta * S * I_total + beta * SVR * I_total) + ai_V * (np.sum(beta_v1 * (V1[:, np.newaxis] * I_total), axis=1) + np.sum(beta_v2 * (V2[:, np.newaxis] * I_total), axis=1)) - np.sum(vr1) * A - gamma * A - mu * A
    dARdt = ai_R * (beta_R * Rv * I_total) - gamma * AR - mu * AR
    dRdt = gamma * I_total - nu_R * R - beta_R * Rv * I_total - mu * R
    dR2dt = np.sum(gamma * IR) - nu_R * R2 - mu * R2

    # Simulate mutation of WT into variants and importation
    if 315 < t < 365:  # alpha appears
        dWTtoA = 0e-5 * dIdt[0]  # don't assume mutation
        dIdt[0] -= dWTtoA
        dAdt[1] += dWTtoA + 50 / 14570000  # people enter Ontario with alpha

    elif 385 < t < 445:
        dAdt[2] += 25 / 14570000  # delta was born

    # Combine all derivatives into a single vector
    yp = np.concatenate([dSdt, dSVRdt, dV1dt, dV2dt, dIdt, dIVdt, dIRdt, dAdt, dARdt, dRdt, dR2dt])

    return yp