import numpy as np


def create_user_item_matrix(user_ids, product_ids, ratings):
    """
    Create sparse user-item matrix.
    Rows: users, Columns: products, Values: ratings
    
    Returns:
        matrix, user_map, product_map
    """
    unique_users = np.unique(user_ids)
    unique_products = np.unique(product_ids)
    
    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    product_map = {product_id: idx for idx, product_id in enumerate(unique_products)}
    
    n_users = len(unique_users)
    n_products = len(unique_products)
    
    matrix = np.zeros((n_users, n_products))
    
    for user_id, product_id, rating in zip(user_ids, product_ids, ratings):
        user_idx = user_map[user_id]
        product_idx = product_map[product_id]
        matrix[user_idx, product_idx] = rating
    
    return matrix, user_map, product_map


def cosine_similarity(vector_a, vector_b):
    """
    Compute cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| * ||B||)
    
    Pure NumPy implementation using broadcasting.
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def cosine_similarity_matrix(matrix, axis=0):
    """
    Compute pairwise cosine similarities.
    
    Args:
        axis: 0 for user-user, 1 for item-item
    
    Returns:
        Similarity matrix
    """
    if axis == 0:
        matrix = matrix.T
    
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    
    normalized = matrix / norms
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix


def pearson_correlation(matrix, axis=0):
    """
    Pearson correlation for collaborative filtering.
    Handles mean-centering automatically.
    """
    if axis == 0:
        matrix = matrix.T
    
    mean_centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    
    stds = np.std(mean_centered, axis=1, keepdims=True)
    stds[stds == 0] = 1
    
    normalized = mean_centered / stds
    correlation_matrix = np.dot(normalized, normalized.T) / (matrix.shape[1] - 1)
    
    return correlation_matrix


def find_top_k_similar(similarity_matrix, idx, k=10):
    """
    Find top-k most similar items/users.
    
    Returns:
        Indices and similarity scores
    """
    similarities = similarity_matrix[idx, :]
    similarities[idx] = -np.inf
    
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]
    
    return top_k_indices, top_k_scores

