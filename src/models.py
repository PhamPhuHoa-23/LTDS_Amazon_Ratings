import numpy as np


def svd_numpy(matrix, k_factors):
    """
    Truncated SVD using NumPy's linalg.svd.
    
    Returns:
        U, Sigma, Vt reduced to k factors
    """
    U, Sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    U_k = U[:, :k_factors]
    Sigma_k = Sigma[:k_factors]
    Vt_k = Vt[:k_factors, :]
    
    return U_k, Sigma_k, Vt_k

