import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    Handles any array shape (1D, 2D, etc.) using NumPy vectorization.
    """
    # Write code here
    # Vectorized implementation only (no Python loops)
    
    # 1. Convert inputs to NumPy float arrays
    w_arr = np.asarray(w, dtype=np.float64)
    g_arr = np.asarray(g, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)

    # 2. Update moving average of squared gradients (new_s)
    new_st = beta * s_arr + (1 - beta) * (g_arr * g_arr)

    # 3. Parameter Update
    new_w = w_arr - (lr / np.sqrt(new_st + eps)) * g_arr
    
    return np.round(new_w, 4), np.round(new_st, 4)
    