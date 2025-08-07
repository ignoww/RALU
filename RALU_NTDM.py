import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.optimize import minimize

# Flow-matching PDF
def pdf_shift(t, h):
    return h / ((1 + (h-1)*t)**2)

# Flow-matching CDF
def cdf_shift(t, h):
    return (h*t) / (1 + (h-1)*t)

# Truncated PDF (eq. (7) in the paper)
def pdf_trunc_shift(t, h, s, e):
    z = cdf_shift(e, h) - cdf_shift(s, h)
    return np.where((t>=s)&(t<=e), pdf_shift(t, h)/z, 0)

# P_target(t) (eq. (8) in the paper)
def P_target(t, h_ori, E, Z):
    S = [0.0]
    for i in range(len(E)-1):
        S.append(E[i] / (Z*(1-E[i]) + E[i]))

    return (pdf_shift(t, h_ori) +
            (E[0]-S[1]) * pdf_trunc_shift(t, h_ori, S[1], E[0]) +
            (E[1]-S[2]) * pdf_trunc_shift(t, h_ori, S[2], E[1])
            ) / (1 + (E[0]-S[1]) + (E[1]-S[2]))

# P(t) (eq. (9) in the paper)
def P(t, N, H, E, Z):
    S = [0.0]
    for i in range(len(N)-1):
        S.append(E[i] / (Z*(1-E[i]) + E[i]))
    return (N[0] * pdf_trunc_shift(t, H[0], S[0], E[0]) +
            N[1] * pdf_trunc_shift(t, H[1], S[1], E[1]) +
            N[2] * pdf_trunc_shift(t, H[2], S[2], E[2])
            ) / sum(N)

# Jensen-Shannon divergence
def js_objective(params, N, E, h_ori, grid):
    H = params[:3]
    Z = params[3]

    p_vals = P_target(grid, h_ori, E, Z)
    q_vals = P(grid, N, H, E, Z)

    epsilon = 1e-10
    p_vals = np.clip(p_vals, epsilon, None)
    q_vals = np.clip(q_vals, epsilon, None)
    p_vals /= p_vals.sum()
    q_vals /= q_vals.sum()

    return jensenshannon(p_vals, q_vals)

# NT-DM: minimize JSD(P_target(t), P(t))
def NT_DM(N, E, h_ori=3):
    grid = np.linspace(0, 1, 1000)
    initial_guess = [3.0, 3.0, 3.0, 2.0]
    bounds = [(0.5, 10.0), (0.5, 10.0), (0.5, 10.0), (2.0, 100.0)]
    result = minimize(js_objective, initial_guess, args=(N, E, h_ori, grid), bounds=bounds)
    
    return result.x[:3], result.x[3]




if __name__ == "__main__":
    N = [5, 6, 7]
    E = [0.3, 0.45, 1.0]
    h_ori = 3  # shift factor (FLUX.1-dev, Stable Diffusion 3)

    h_opt, z_opt = NT_DM(N, E, h_ori) # z = 1/sqrt(c)

    print(f"Optimal h: {h_opt}")
    print(f"Optimal z: {z_opt}")