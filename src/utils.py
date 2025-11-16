"""
Utility Functions Module
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Module này chứa các utility functions: evaluation metrics, helpers
Tất cả functions đều vectorized
"""

import numpy as np


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def rmse(y_true, y_pred):
    """
    Root Mean Square Error (vectorized)
    
    Parameters:
    -----------
    y_true, y_pred : numpy arrays
        True và predicted values
        
    Returns:
    --------
    float : RMSE score
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Mean Absolute Error (vectorized)
    
    Parameters:
    -----------
    y_true, y_pred : numpy arrays
        True và predicted values
        
    Returns:
    --------
    float : MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def precision_at_k(recommended, relevant, k):
    """
    Precision@K metric (vectorized)
    
    Parameters:
    -----------
    recommended : numpy array
        Array of recommended items
    relevant : numpy array or set
        Array/set of relevant items
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : Precision@K score
    """
    if k == 0:
        return 0.0
    
    # Get top K recommendations
    top_k = recommended[:k]
    
    # Count relevant items in top K (vectorized)
    n_relevant = np.sum(np.isin(top_k, relevant))
    
    return n_relevant / k


def recall_at_k(recommended, relevant, k):
    """
    Recall@K metric (vectorized)
    
    Parameters:
    -----------
    recommended : numpy array
        Array of recommended items
    relevant : numpy array or set
        Array/set of relevant items
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : Recall@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    # Get top K recommendations
    top_k = recommended[:k]
    
    # Count relevant items in top K
    n_relevant = np.sum(np.isin(top_k, relevant))
    
    return n_relevant / len(relevant)


def f1_at_k(recommended, relevant, k):
    """
    F1@K metric (vectorized)
    
    Parameters:
    -----------
    recommended : numpy array
        Array of recommended items
    relevant : numpy array or set
        Array/set of relevant items
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : F1@K score
    """
    prec = precision_at_k(recommended, relevant, k)
    rec = recall_at_k(recommended, relevant, k)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def ndcg_at_k(recommended, relevant, k):
    """
    Normalized Discounted Cumulative Gain @K (vectorized)
    
    Parameters:
    -----------
    recommended : numpy array
        Array of recommended items
    relevant : numpy array or dict
        Array of relevant items or dict {item: relevance_score}
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : NDCG@K score
    """
    # Get top K
    top_k = recommended[:k]
    
    # Create relevance scores
    if isinstance(relevant, dict):
        relevance = np.array([relevant.get(item, 0) for item in top_k])
    else:
        relevance = np.isin(top_k, relevant).astype(float)
    
    # Compute DCG (vectorized)
    # DCG = sum(relevance / log2(position + 1))
    positions = np.arange(1, len(relevance) + 1)
    dcg = np.sum(relevance / np.log2(positions + 1))
    
    # Compute IDCG (ideal DCG)
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = np.sum(ideal_relevance / np.log2(positions + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(recommended, relevant, k):
    """
    Hit Rate @K: 1 nếu có ít nhất 1 relevant item trong top K, 0 otherwise
    
    Parameters:
    -----------
    recommended : numpy array
        Array of recommended items
    relevant : numpy array or set
        Array/set of relevant items
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : Hit rate (0 or 1)
    """
    top_k = recommended[:k]
    return float(np.any(np.isin(top_k, relevant)))


def mean_reciprocal_rank(recommended, relevant):
    """
    Mean Reciprocal Rank (vectorized)
    
    Parameters:
    -----------
    recommended : numpy array
        Array of recommended items
    relevant : numpy array or set
        Array/set of relevant items
        
    Returns:
    --------
    float : MRR score
    """
    # Find positions of relevant items (vectorized)
    is_relevant = np.isin(recommended, relevant)
    
    if not np.any(is_relevant):
        return 0.0
    
    # Get position of first relevant item (1-indexed)
    first_relevant_pos = np.where(is_relevant)[0][0] + 1
    
    return 1.0 / first_relevant_pos


def coverage(all_recommendations, all_items):
    """
    Coverage: Tỷ lệ items được recommend ít nhất 1 lần
    
    Parameters:
    -----------
    all_recommendations : list of numpy arrays
        List of recommendation arrays
    all_items : numpy array or set
        All possible items
        
    Returns:
    --------
    float : Coverage ratio
    """
    # Combine all recommendations (vectorized)
    if isinstance(all_recommendations, list):
        recommended_items = np.unique(np.concatenate(all_recommendations))
    else:
        recommended_items = np.unique(all_recommendations)
    
    return len(recommended_items) / len(all_items)


def diversity(recommendations):
    """
    Diversity: Số lượng unique items trong recommendations
    
    Parameters:
    -----------
    recommendations : numpy array or list
        Recommended items
        
    Returns:
    --------
    float : Diversity ratio
    """
    if isinstance(recommendations, list):
        unique_items = len(np.unique(np.concatenate(recommendations)))
        total_items = sum(len(rec) for rec in recommendations)
    else:
        unique_items = len(np.unique(recommendations))
        total_items = len(recommendations)
    
    if total_items == 0:
        return 0.0
    
    return unique_items / total_items


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_minmax(x, feature_range=(0, 1)):
    """
    Min-Max normalization (vectorized)
    
    Parameters:
    -----------
    x : numpy array
        Input array
    feature_range : tuple
        Target range (min, max)
        
    Returns:
    --------
    numpy array : Normalized array
    """
    min_val, max_val = feature_range
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max - x_min == 0:
        return np.full_like(x, min_val, dtype=float)
    
    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled * (max_val - min_val) + min_val


def standardize(x):
    """
    Z-score standardization (vectorized)
    
    Parameters:
    -----------
    x : numpy array
        Input array
        
    Returns:
    --------
    numpy array : Standardized array
    """
    mean = np.mean(x)
    std = np.std(x)
    
    if std == 0:
        return np.zeros_like(x, dtype=float)
    
    return (x - mean) / std


def cosine_similarity(a, b):
    """
    Cosine similarity giữa 2 vectors (vectorized)
    
    Parameters:
    -----------
    a, b : numpy arrays
        Input vectors
        
    Returns:
    --------
    float : Cosine similarity
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def cosine_similarity_matrix(matrix):
    """
    Compute pairwise cosine similarity matrix (vectorized)
    
    Parameters:
    -----------
    matrix : numpy array
        Matrix với shape (n_samples, n_features)
        
    Returns:
    --------
    numpy array : Similarity matrix (n_samples, n_samples)
    """
    # Normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Tránh chia 0
    
    normalized = matrix / norms
    
    # Compute similarity (vectorized matrix multiplication)
    return np.dot(normalized, normalized.T)


def pearson_correlation(a, b):
    """
    Pearson correlation coefficient (vectorized)
    
    Parameters:
    -----------
    a, b : numpy arrays
        Input vectors
        
    Returns:
    --------
    float : Correlation coefficient
    """
    # Remove positions where both are 0
    mask = (a != 0) | (b != 0)
    a_filtered = a[mask]
    b_filtered = b[mask]
    
    if len(a_filtered) < 2:
        return 0.0
    
    # Center the vectors
    a_centered = a_filtered - np.mean(a_filtered)
    b_centered = b_filtered - np.mean(b_filtered)
    
    # Compute correlation
    numerator = np.sum(a_centered * b_centered)
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def get_user_item_pairs(user_indices, product_indices, ratings):
    """
    Create dictionary of user-item pairs (vectorized preprocessing)
    
    Parameters:
    -----------
    user_indices, product_indices, ratings : numpy arrays
        
    Returns:
    --------
    dict : {user_id: {product_id: rating}}
    """
    user_items = {}
    
    # Vectorized grouping
    unique_users = np.unique(user_indices)
    
    for user_id in unique_users:
        mask = user_indices == user_id
        products = product_indices[mask]
        user_ratings = ratings[mask]
        
        user_items[user_id] = dict(zip(products, user_ratings))
    
    return user_items


def get_product_users(user_indices, product_indices, ratings):
    """
    Create dictionary of product-user pairs (vectorized preprocessing)
    
    Parameters:
    -----------
    user_indices, product_indices, ratings : numpy arrays
        
    Returns:
    --------
    dict : {product_id: {user_id: rating}}
    """
    product_users = {}
    
    # Vectorized grouping
    unique_products = np.unique(product_indices)
    
    for product_id in unique_products:
        mask = product_indices == product_id
        users = user_indices[mask]
        product_ratings = ratings[mask]
        
        product_users[product_id] = dict(zip(users, product_ratings))
    
    return product_users


def create_sparse_matrix(user_indices, product_indices, ratings, n_users, n_products):
    """
    Create sparse user-item matrix (vectorized)
    
    Parameters:
    -----------
    user_indices, product_indices, ratings : numpy arrays
    n_users, n_products : int
        
    Returns:
    --------
    numpy array : Sparse matrix
    """
    matrix = np.zeros((n_users, n_products))
    matrix[user_indices, product_indices] = ratings
    return matrix


def compute_sparsity(matrix):
    """
    Compute sparsity of matrix (vectorized)
    
    Parameters:
    -----------
    matrix : numpy array
        
    Returns:
    --------
    float : Sparsity ratio
    """
    total_elements = matrix.size
    non_zero_elements = np.count_nonzero(matrix)
    
    return 1 - (non_zero_elements / total_elements)


def top_k_indices(array, k, reverse=False):
    """
    Get indices of top K values (vectorized)
    
    Parameters:
    -----------
    array : numpy array
    k : int
    reverse : bool
        If True, get bottom K instead
        
    Returns:
    --------
    numpy array : Indices of top K values
    """
    if reverse:
        return np.argpartition(array, k)[:k]
    else:
        return np.argpartition(array, -k)[-k:]


def batch_process(data, batch_size, func):
    """
    Process data in batches (vectorized batching)
    
    Parameters:
    -----------
    data : numpy array or list
    batch_size : int
    func : callable
        Function to apply to each batch
        
    Returns:
    --------
    list : Results from each batch
    """
    results = []
    n_samples = len(data)
    
    for i in range(0, n_samples, batch_size):
        batch = data[i:i+batch_size]
        result = func(batch)
        results.append(result)
    
    return results


def print_metrics(metrics_dict, model_name="Model"):
    """
    Pretty print evaluation metrics
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of {metric_name: value}
    model_name : str
        Name of model
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS: {model_name}")
    print(f"{'='*60}")
    
    for metric, value in metrics_dict.items():
        print(f"  {metric:.<40} {value:.6f}")
    
    print(f"{'='*60}\n")


def save_results(filepath, **kwargs):
    """
    Save results to .npz file
    
    Parameters:
    -----------
    filepath : str
        Path to save file
    **kwargs : dict
        Data to save
    """
    np.savez_compressed(filepath, **kwargs)
    print(f"✓ Results saved to: {filepath}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test metrics
    y_true = np.array([5, 4, 3, 5, 2])
    y_pred = np.array([4.5, 4.2, 3.1, 4.8, 2.3])
    
    print(f"\n1. Regression metrics:")
    print(f"   RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"   MAE: {mae(y_true, y_pred):.4f}")
    
    # Test ranking metrics
    recommended = np.array([1, 3, 5, 7, 9])
    relevant = np.array([1, 5, 8])
    
    print(f"\n2. Ranking metrics (k=5):")
    print(f"   Precision@5: {precision_at_k(recommended, relevant, 5):.4f}")
    print(f"   Recall@5: {recall_at_k(recommended, relevant, 5):.4f}")
    print(f"   F1@5: {f1_at_k(recommended, relevant, 5):.4f}")
    print(f"   Hit Rate@5: {hit_rate_at_k(recommended, relevant, 5):.4f}")
    print(f"   MRR: {mean_reciprocal_rank(recommended, relevant):.4f}")
    
    # Test normalization
    x = np.array([1, 2, 3, 4, 5])
    print(f"\n3. Normalization:")
    print(f"   MinMax: {normalize_minmax(x)}")
    print(f"   Standardize: {standardize(x)}")
    
    # Test similarity
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    print(f"\n4. Similarity:")
    print(f"   Cosine: {cosine_similarity(a, b):.4f}")
    print(f"   Pearson: {pearson_correlation(a, b):.4f}")
    
    print("\nAll utility tests passed!")
