import numpy as np

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample binary cross-entropy (log loss).
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip predictions to prevent log(0) undefined errors
    p_prime = np.clip(y_pred, eps, 1 - eps)

    # Fully vectorized binary cross-entropy formula
    loss = -(y_true * np.log(p_prime) + (1 - y_true) * np.log(1 - p_prime))

    return loss.tolist()