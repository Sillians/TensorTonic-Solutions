import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    new_A = np.asarray(A)
    A_T = []
    for i in range(len(new_A[0])):
        A_T.append([new_A[j][i] for j in range(len(new_A))])
    return np.asarray(A_T)
    # pass
