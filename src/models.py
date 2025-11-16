"""
Recommendation Models Module (Consolidated with TruncatedSVD)
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Module này chứa các recommendation algorithms
Tất cả models được implement chỉ bằng NumPy (không dùng Scikit-learn)
Bao gồm TruncatedSVD implementation từ scratch (power iteration)
"""

import numpy as np


# ============================================================================
# TRUNCATED SVD CLASS (FROM SCRATCH - POWER ITERATION)
# ============================================================================

class TruncatedSVD:
    """
    Truncated SVD dùng power iteration method (từ scratch, không dùng sklearn)
    Phù hợp cho dense hoặc sparse matrices
    """
    
    def __init__(self, n_components=50, n_iterations=20, random_state=42):
        """
        Parameters:
        -----------
        n_components : int
            Số lượng components (rank k)
        n_iterations : int
            Số iterations cho power method
        random_state : int
            Random seed
        """
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.U = None
        self.Vt = None
        self.singular_values = None
    
    def fit(self, X):
        """
        Fit SVD trên matrix X
        
        Parameters:
        -----------
        X : numpy array
            Dense matrix (m x n)
        """
        np.random.seed(self.random_state)
        m, n = X.shape
        
        # Initialize random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        # Power iteration để tính principal components
        U_list = []
        singular_values = []
        X_copy = X.copy()
        
        for i in range(min(self.n_components, min(m, n))):
            # Power iterations
            for _ in range(self.n_iterations):
                u = X_copy.T @ v
                u = u / np.linalg.norm(u)
                v = X_copy @ u
                v = v / np.linalg.norm(v)
            
            # Compute singular value
            sigma = np.linalg.norm(X_copy @ u)
            
            if sigma < 1e-10:
                break
            
            U_list.append(u)
            singular_values.append(sigma)
            
            # Deflation: remove component từ X
            X_copy = X_copy - sigma * np.outer(v, u)
        
        # Store results
        self.singular_values = np.array(singular_values)
        self.Vt = np.array(U_list).T  # Shape: (n, k)
        
        # Compute U: project X onto V
        self.U = X @ self.Vt  # Shape: (m, k)
        
        # Normalize U columns
        for i in range(self.U.shape[1]):
            norm = np.linalg.norm(self.U[:, i])
            if norm > 0:
                self.U[:, i] /= norm
                self.Vt[:, i] *= norm
    
    def reconstruct(self):
        """Reconstruct approximation của original matrix"""
        if self.U is None or self.Vt is None:
            raise ValueError("Model chưa được fit")
        return self.U @ (self.singular_values[:, None] * self.Vt.T)
    
    def transform(self, X):
        """Transform new data"""
        if self.Vt is None:
            raise ValueError("Model chưa được fit")
        return X @ self.Vt


class PopularityRecommender:
    """
    Popularity-based Recommender
    Recommend sản phẩm dựa trên số lượng ratings (cold start solution)
    """
    
    def __init__(self):
        self.product_popularity = None
        self.product_ids = None
        
    def fit(self, product_indices, ratings=None):
        """
        Train model bằng cách đếm số lượng ratings cho mỗi product
        
        Parameters:
        -----------
        product_indices : numpy array
            Array chứa product indices
        ratings : numpy array, optional
            Array chứa ratings (không dùng trong model này)
        """
        # Vectorized counting
        self.product_ids, counts = np.unique(product_indices, return_counts=True)
        self.product_popularity = counts
        
        # Sort by popularity
        sorted_idx = np.argsort(self.product_popularity)[::-1]
        self.product_ids = self.product_ids[sorted_idx]
        self.product_popularity = self.product_popularity[sorted_idx]
        
    def recommend(self, user_id=None, top_n=10, exclude_products=None):
        """
        Recommend top N popular products
        
        Parameters:
        -----------
        user_id : any, optional
            User ID (không sử dụng trong popularity-based)
        top_n : int
            Số lượng recommendations
        exclude_products : set or list, optional
            Danh sách products cần loại trừ
            
        Returns:
        --------
        numpy array : Top N product IDs
        """
        if exclude_products is None:
            return self.product_ids[:top_n]
        
        # Filter out excluded products
        mask = ~np.isin(self.product_ids, list(exclude_products))
        filtered_products = self.product_ids[mask]
        return filtered_products[:top_n]


class ItemBasedCF:
    """
    Item-based Collaborative Filtering
    Recommend dựa trên similarity giữa các products
    """
    
    def __init__(self, min_similarity=0.0):
        self.similarity_matrix = None
        self.product_ids = None
        self.min_similarity = min_similarity
        
    def fit(self, user_indices, product_indices, ratings, n_products):
        """
        Train model bằng cách tính item-item similarity matrix
        
        Parameters:
        -----------
        user_indices : numpy array
            User indices
        product_indices : numpy array
            Product indices
        ratings : numpy array
            Ratings
        n_products : int
            Số lượng unique products
        """
        # Create user-item matrix (vectorized)
        # Matrix shape: (n_users, n_products)
        n_users = np.max(user_indices) + 1
        
        # Sparse representation using arrays
        user_item_matrix = np.zeros((n_users, n_products))
        user_item_matrix[user_indices, product_indices] = ratings
        
        # Transpose to get item-user matrix
        # Shape: (n_products, n_users)
        item_user_matrix = user_item_matrix.T
        
        # Compute item-item similarity using cosine similarity (vectorized)
        # Cosine similarity: (A · B) / (||A|| * ||B||)
        
        # Compute norms (vectorized)
        norms = np.linalg.norm(item_user_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Tránh chia cho 0
        
        # Normalize vectors
        normalized_matrix = item_user_matrix / norms
        
        # Compute similarity matrix (vectorized matrix multiplication)
        self.similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        
        # Set diagonal to 0 (product không similar với chính nó)
        np.fill_diagonal(self.similarity_matrix, 0)
        
        self.product_ids = np.arange(n_products)
        
        # similarity matrix đã được tính xong (không in thông tin chi tiết)
        
    def recommend(self, product_id, top_n=10, exclude_products=None):
        """
        Recommend similar products
        
        Parameters:
        -----------
        product_id : int
            Product ID để tìm similar items
        top_n : int
            Số lượng recommendations
        exclude_products : set or list, optional
            Products cần loại trừ
            
        Returns:
        --------
        numpy array : Top N similar product IDs
        """
        # Get similarity scores for this product
        similarities = self.similarity_matrix[product_id]
        
        # Filter by minimum similarity
        mask = similarities >= self.min_similarity
        
        # Exclude products if specified
        if exclude_products is not None:
            exclude_mask = ~np.isin(self.product_ids, list(exclude_products))
            mask = mask & exclude_mask
        
        # Get valid products and their similarities
        valid_products = self.product_ids[mask]
        valid_similarities = similarities[mask]
        
        # Sort by similarity (descending)
        sorted_idx = np.argsort(valid_similarities)[::-1]
        
        return valid_products[sorted_idx][:top_n]
    
    def recommend_for_user(self, user_rated_products, user_ratings, top_n=10):
        """
        Recommend products cho user dựa trên items họ đã rate
        
        Parameters:
        -----------
        user_rated_products : numpy array
            Array of product IDs user đã rate
        user_ratings : numpy array
            Ratings tương ứng
        top_n : int
            Số lượng recommendations
            
        Returns:
        --------
        numpy array : Top N recommended product IDs
        """
        # Tính weighted average of similarities (vectorized)
        scores = np.zeros(len(self.product_ids))
        
        for prod_id, rating in zip(user_rated_products, user_ratings):
            # Lấy similarity scores
            similarities = self.similarity_matrix[prod_id]
            # Weight by rating
            scores += similarities * rating
        
        # Normalize by number of rated items
        scores /= len(user_rated_products)
        
        # Exclude already rated products
        scores[user_rated_products] = -np.inf
        
        # Get top N
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        return self.product_ids[top_indices]


class UserBasedCF:
    """
    User-based Collaborative Filtering
    Recommend dựa trên similarity giữa users
    """
    
    def __init__(self, k_neighbors=50, min_similarity=0.0):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.user_item_matrix = None
        
    def fit(self, user_indices, product_indices, ratings, n_users, n_products):
        """
        Train model
        
        Parameters:
        -----------
        user_indices, product_indices, ratings : numpy arrays
        n_users, n_products : int
        """
        # Create user-item matrix
        self.user_item_matrix = np.zeros((n_users, n_products))
        self.user_item_matrix[user_indices, product_indices] = ratings
        
        self.n_users = n_users
        self.n_products = n_products
        
    def _compute_user_similarity(self, user_id):
        """
        Compute similarity giữa user_id và tất cả users khác (vectorized)
        
        Parameters:
        -----------
        user_id : int
            User ID
            
        Returns:
        --------
        numpy array : Similarity scores với tất cả users
        """
        # Get user's rating vector
        user_vector = self.user_item_matrix[user_id]
        
        # Compute cosine similarity với tất cả users (vectorized)
        # Similarity = (u · v) / (||u|| * ||v||)
        
        dot_products = np.dot(self.user_item_matrix, user_vector)
        
        user_norm = np.linalg.norm(user_vector)
        other_norms = np.linalg.norm(self.user_item_matrix, axis=1)
        
        # Tránh chia cho 0
        denominators = user_norm * other_norms
        denominators[denominators == 0] = 1
        
        similarities = dot_products / denominators
        
        # Set similarity với chính mình = 0
        similarities[user_id] = 0
        
        return similarities
        
    def recommend(self, user_id, top_n=10, exclude_products=None):
        """
        Recommend products cho user
        
        Parameters:
        -----------
        user_id : int
        top_n : int
        exclude_products : set or list, optional
            
        Returns:
        --------
        numpy array : Top N product IDs
        """
        # Compute similarities với tất cả users
        similarities = self._compute_user_similarity(user_id)
        
        # Get top K similar users
        # Filter by minimum similarity
        valid_mask = similarities >= self.min_similarity
        valid_similarities = similarities[valid_mask]
        valid_users = np.where(valid_mask)[0]
        
        # Sort và lấy top K
        sorted_idx = np.argsort(valid_similarities)[::-1][:self.k_neighbors]
        neighbor_ids = valid_users[sorted_idx]
        neighbor_sims = valid_similarities[sorted_idx]
        
        # Tính predicted ratings (vectorized)
        # Weighted average of neighbor ratings
        neighbor_ratings = self.user_item_matrix[neighbor_ids]  # Shape: (k, n_products)
        
        # Weighted sum
        weighted_sum = np.dot(neighbor_sims, neighbor_ratings)  # Vectorized
        
        # Normalize
        sim_sum = np.sum(neighbor_sims)
        if sim_sum == 0:
            predicted_ratings = weighted_sum
        else:
            predicted_ratings = weighted_sum / sim_sum
        
        # Exclude products user đã rate
        user_rated_mask = self.user_item_matrix[user_id] > 0
        predicted_ratings[user_rated_mask] = -np.inf
        
        # Exclude specified products
        if exclude_products is not None:
            predicted_ratings[list(exclude_products)] = -np.inf
        
        # Get top N
        top_indices = np.argsort(predicted_ratings)[::-1][:top_n]
        
        return top_indices


class TruncatedSVD:
    """
    Truncated SVD (chỉ tính k largest singular values)
    Implement từ scratch dùng power iteration method (không dùng sklearn)
    """
    
    def __init__(self, n_components=50, n_iterations=20):
        """
        Parameters:
        -----------
        n_components : int
            Số lượng singular values cần tính
        n_iterations : int
            Số lần iterations cho power method
        """
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.U = None
        self.sigma = None
        self.Vt = None
        
    def _power_method(self, matrix, k=1, max_iter=20):
        """
        Power iteration method để tính eigenvectors
        (được dùng để tính singular vectors)
        
        Parameters:
        -----------
        matrix : numpy array
            Input matrix (normally M^T @ M hoặc M @ M^T)
        k : int
            Số lượng eigenvectors cần tính
        max_iter : int
            Số lần iterations
            
        Returns:
        --------
        tuple : (eigenvalues, eigenvectors)
        """
        m, n = matrix.shape
        eigenvalues = []
        eigenvectors = []
        
        # Copy matrix để tránh modify original
        A = matrix.copy().astype(np.float64)
        
        for _ in range(k):
            # Initialize random vector
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            
            # Power iteration
            for _ in range(max_iter):
                # v_new = (A^T @ A) @ v
                u = np.dot(A.T, np.dot(A, v))
                u = u / (np.linalg.norm(u) + 1e-10)
                
                # Check convergence
                if np.linalg.norm(u - v) < 1e-6:
                    break
                v = u
            
            # Eigenvalue = ||A @ v||
            Av = np.dot(A, v)
            eigenvalue = np.linalg.norm(Av)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            
            # Deflation: A = A - eigenvalue * u * v^T
            u = Av / (eigenvalue + 1e-10)
            A = A - eigenvalue * np.outer(u, v)
        
        return np.array(eigenvalues), np.array(eigenvectors).T
    
    def fit(self, matrix):
        """
        Tính Truncated SVD của matrix
        
        Parameters:
        -----------
        matrix : numpy array
            Input matrix (user-item matrix)
        """
        m, n = matrix.shape
        k = min(self.n_components, m, n)
        
        # Tính right singular vectors (V) từ M^T @ M (vectorized)
        MTM = np.dot(matrix.T, matrix)
        eigenvalues_v, right_vecs = self._power_method(MTM, k=k, max_iter=self.n_iterations)
        
        # Singular values = sqrt(eigenvalues)
        self.sigma = np.sqrt(np.maximum(eigenvalues_v, 0))  # Tránh sqrt(negative)
        
        # Tính left singular vectors (U) từ M @ M^T (vectorized)
        MMT = np.dot(matrix, matrix.T)
        eigenvalues_u, left_vecs = self._power_method(MMT, k=k, max_iter=self.n_iterations)
        
        # Left singular vectors
        self.U = left_vecs
        
        # Right singular vectors (V^T)
        self.Vt = right_vecs.T
    
    def transform(self, matrix):
        """
        Transform matrix using learned SVD
        
        Parameters:
        -----------
        matrix : numpy array
            Input matrix
            
        Returns:
        --------
        numpy array : Transformed matrix (U * Sigma)
        """
        if self.U is None or self.sigma is None:
            raise ValueError("Model chưa fit. Gọi fit() trước.")
        
        # Project matrix lên latent space
        return np.dot(matrix, np.dot(self.Vt.T, np.diag(1 / (self.sigma + 1e-10))))
    
    def reconstruct(self):
        """
        Reconstruct approximated matrix
        
        Returns:
        --------
        numpy array : Reconstructed matrix
        """
        if self.U is None or self.sigma is None or self.Vt is None:
            raise ValueError("Model chưa fit. Gọi fit() trước.")
        
        # Reconstruct: U * Sigma * V^T (vectorized)
        return np.dot(self.U, np.dot(np.diag(self.sigma), self.Vt))


class SVDRecommender:
    """
    Matrix Factorization using Truncated SVD (dùng TruncatedSVD từ scratch)
    """
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd_model = TruncatedSVD(n_components=n_components)
        self.user_item_matrix = None
        
    def fit(self, user_indices, product_indices, ratings, n_users, n_products):
        """
        Train model using Truncated SVD
        
        Parameters:
        -----------
        user_indices, product_indices, ratings : numpy arrays
        n_users, n_products : int
        """
        # Create user-item matrix
        self.user_item_matrix = np.zeros((n_users, n_products))
        self.user_item_matrix[user_indices, product_indices] = ratings
        
        # Fit Truncated SVD (vectorized)
        self.svd_model.fit(self.user_item_matrix)
        
        # SVD hoàn tất (không in thông tin chi tiết)
        
    def reconstruct_matrix(self):
        """
        Reconstruct approximated user-item matrix
        
        Returns:
        --------
        numpy array : Reconstructed matrix
        """
        return self.svd_model.reconstruct()
        
    def recommend(self, user_id, top_n=10, exclude_products=None):
        """
        Recommend products cho user
        
        Parameters:
        -----------
        user_id : int
        top_n : int
        exclude_products : set or list, optional
            
        Returns:
        --------
        numpy array : Top N product IDs
        """
        # Get predicted ratings for this user
        reconstructed = self.reconstruct_matrix()
        predicted_ratings = reconstructed[user_id]
        
        # Exclude products user đã rate
        user_rated_mask = self.user_item_matrix[user_id] > 0
        predicted_ratings[user_rated_mask] = -np.inf
        
        # Exclude specified products
        if exclude_products is not None:
            predicted_ratings[list(exclude_products)] = -np.inf
        
        # Get top N
        top_indices = np.argsort(predicted_ratings)[::-1][:top_n]
        
        return top_indices


class WeightedRecommender:
    """
    Weighted Recommender sử dụng multiple features
    """
    
    def __init__(self, weights=None):
        """
        Parameters:
        -----------
        weights : dict
            Dictionary {feature_name: weight}
        """
        if weights is None:
            self.weights = {
                'popularity': 0.33,
                'avg_rating': 0.40,
                'recency': 0.27
            }
        else:
            self.weights = weights
            
        self.scores = None
        self.product_ids = None
        
    def fit(self, product_indices, ratings, timestamps):
        """
        Compute weighted scores
        
        Parameters:
        -----------
        product_indices, ratings, timestamps : numpy arrays
        """
        unique_products = np.unique(product_indices)
        n_products = len(unique_products)
        
        # Initialize scores
        popularity_scores = np.zeros(n_products)
        rating_scores = np.zeros(n_products)
        recency_scores = np.zeros(n_products)
        
        # Compute features for each product (vectorized)
        for idx, prod_id in enumerate(unique_products):
            mask = product_indices == prod_id
            
            # Popularity: count
            popularity_scores[idx] = np.sum(mask)
            
            # Average rating
            rating_scores[idx] = np.mean(ratings[mask])
            
            # Recency: avg timestamp
            recency_scores[idx] = np.mean(timestamps[mask])
        
        # Normalize scores to [0, 1] (vectorized)
        def normalize(x):
            min_x = np.min(x)
            max_x = np.max(x)
            if max_x - min_x == 0:
                return np.zeros_like(x)
            return (x - min_x) / (max_x - min_x)
        
        popularity_scores = normalize(popularity_scores)
        rating_scores = normalize(rating_scores)
        recency_scores = normalize(recency_scores)
        
        # Compute weighted score (vectorized)
        self.scores = (
            self.weights['popularity'] * popularity_scores +
            self.weights['avg_rating'] * rating_scores +
            self.weights['recency'] * recency_scores
        )
        
        self.product_ids = unique_products
        
    def recommend(self, top_n=10, exclude_products=None):
        """
        Recommend top N products
        
        Parameters:
        -----------
        top_n : int
        exclude_products : set or list, optional
            
        Returns:
        --------
        numpy array : Top N product IDs
        """
        scores = self.scores.copy()
        
        # Exclude products
        if exclude_products is not None:
            for prod_id in exclude_products:
                idx = np.where(self.product_ids == prod_id)[0]
                if len(idx) > 0:
                    scores[idx[0]] = -np.inf
        
        # Sort and get top N
        sorted_idx = np.argsort(scores)[::-1][:top_n]
        
        return self.product_ids[sorted_idx]


# ============================================================================
# SIMILARITY FUNCTIONS
# ============================================================================

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
        matrix: Input matrix
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
    
    top_k = recommended[:k]
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
    
    top_k = recommended[:k]
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
    top_k = recommended[:k]
    
    if isinstance(relevant, dict):
        relevance = np.array([relevant.get(item, 0) for item in top_k])
    else:
        relevance = np.isin(top_k, relevant).astype(float)
    
    positions = np.arange(1, len(relevance) + 1)
    dcg = np.sum(relevance / np.log2(positions + 1))
    
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
    is_relevant = np.isin(recommended, relevant)
    
    if not np.any(is_relevant):
        return 0.0
    
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


def print_metrics(metrics_dict, model_name="Model"):
    """
    In các metric đơn giản, mỗi dòng: <metric>: <value>
    """
    print(f"{model_name} metrics:")
    for metric, value in metrics_dict.items():
        try:
            print(f"{metric}: {value:.6f}")
        except Exception:
            print(f"{metric}: {value}")


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
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    # Chạy thử (sanity checks) - in rất ngắn gọn
    np.random.seed(42)
    n_samples = 1000
    n_users = 100
    n_products = 50

    user_indices = np.random.randint(0, n_users, n_samples)
    product_indices = np.random.randint(0, n_products, n_samples)
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_samples)
    timestamps = np.random.randint(1000000000, 1700000000, n_samples)

    # Popularity
    pop_model = PopularityRecommender()
    pop_model.fit(product_indices, ratings)
    recs = pop_model.recommend(top_n=5)
    print("Top 5 (popularity):", recs)

    # Item-based CF
    item_cf = ItemBasedCF()
    item_cf.fit(user_indices, product_indices, ratings, n_products)
    recs = item_cf.recommend(product_id=0, top_n=5)
    print("Top 5 (item-similar to 0):", recs)

    # SVD
    svd_model = SVDRecommender(n_components=10)
    svd_model.fit(user_indices, product_indices, ratings, n_users, n_products)
    recs = svd_model.recommend(user_id=0, top_n=5)
    print("Top 5 (SVD for user 0):", recs)
