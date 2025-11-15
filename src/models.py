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


def matrix_factorization_sgd(R, K, steps=5000, alpha=0.002, beta=0.02):
    """
    Matrix Factorization using Stochastic Gradient Descent.
    
    Factorize R (user-item matrix) into P (user features) and Q (item features).
    R ≈ P × Q^T
    
    Args:
        R: User-item rating matrix
        K: Number of latent factors
        steps: Training iterations
        alpha: Learning rate
        beta: Regularization parameter
    
    Returns:
        P, Q matrices
    """
    n_users, n_items = R.shape
    
    P = np.random.normal(scale=0.1, size=(n_users, K))
    Q = np.random.normal(scale=0.1, size=(n_items, K))
    
    non_zero_mask = R != 0
    
    for step in range(steps):
        for i in range(n_users):
            for j in range(n_items):
                if non_zero_mask[i, j]:
                    eij = R[i, j] - np.dot(P[i, :], Q[j, :])
                    
                    P_i_old = P[i, :].copy()
                    P[i, :] += alpha * (eij * Q[j, :] - beta * P[i, :])
                    Q[j, :] += alpha * (eij * P_i_old - beta * Q[j, :])
    
    return P, Q


def predict_rating(P, Q, user_idx, item_idx):
    """
    Predict rating for user-item pair.
    
    Formula: r_ui = p_u · q_i^T
    """
    prediction = np.dot(P[user_idx, :], Q[item_idx, :])
    prediction = np.clip(prediction, 1.0, 5.0)
    return prediction

