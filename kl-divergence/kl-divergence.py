import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p, np.float32)
    q = np.asarray(q, np.float32)

    # Filter out points where p == 0 (since 0 * log(0) limit is 0 in KL divergence)
    mask = p > 0
    p_pos = p[mask]
    q_pos = q[mask]

    # Clip q to avoid division by zero or log(0)
    clipped_q = np.clip(q_pos, eps, None)

    # Fully vectorized sum: sum(p * log(p / q))
    kl = sum(p_pos * (np.log(p_pos / clipped_q)))
    return round(float(kl), 6)