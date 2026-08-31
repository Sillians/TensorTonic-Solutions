import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here
    p_hat = [min(1 - eps, max(eps, pred)) for pred in y_pred]
    loss = [- (1 - y_t) * math.log(1 - p_h) - y_t * math.log(p_h) for y_t, p_h in zip(y_true, p_hat)]
    rounded_loss = [round(l, 6) for l in loss]
    return rounded_loss