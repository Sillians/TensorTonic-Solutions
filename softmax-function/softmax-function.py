import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x_ = np.asarray(x)
    
    if len(x_.shape) == 2:
        max_x = np.max(x_, axis=1, keepdims=True)
        sub_x = (x_ - max_x)
        num_x = np.exp(sub_x)
        dem_x = np.sum(num_x, axis=1, keepdims=True)
        tmax = num_x / dem_x
    else:
        max_ = np.max(x_)
        numx_ = np.exp(x_ - max_)
        demx_ = np.sum(numx_)
        tmax = numx_ / demx_
    return tmax
