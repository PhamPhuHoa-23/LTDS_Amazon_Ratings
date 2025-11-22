"""
Recommendation Models Module
CSC17104 - Programming for Data Science
Sinh viên: Phạm Phú Hòa - MSSV: 23122030

Module chứa các recommendation algorithms (NumPy-only):
- Popularity, ItemCF, UserCF
- TruncatedSVD (from scratch, power iteration)
- ALS (from scratch, alternating least squares)
"""

import numpy as np


# ============================================================================
# ALS (ALTERNATING LEAST SQUARES) - FROM SCRATCH
# ============================================================================

class ALSRecommender:
    """
    ALS (Alternating Least Squares) với Conjugate Gradient optimization
    
    **Nguồn tham khảo:**
    - Hu, Koren, Volinsky (2008): "Collaborative Filtering for Implicit Feedback Datasets"
      http://yifanhu.net/PUB/cf.pdf
    - Takács, Tikk (2012): "Applications of the Conjugate Gradient Method 
      for Implicit Feedback Collaborative Filtering"
      https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf
    - Ben Frederickson blog: "Faster Implicit Matrix Factorization"
      https://www.benfrederickson.com/fast-implicit-matrix-factorization/
    
    **Tối ưu hóa:**
    - Sử dụng Conjugate Gradient thay vì trực tiếp solve (A^-1 @ b)
    - Tránh materialized ma trận Y^T C_u Y bằng cách tính Y^T Y trước
    - Vectorized operations, tránh loop không cần thiết
    - Complexity: O(N^2) per user thay vì O(N^3) với Cholesky
    """
    
    def __init__(self, n_factors=50, n_iterations=10, lambda_reg=0.1, 
                 cg_steps=3, random_state=42):
        """
        Parameters:
        -----------
        n_factors : int
            Số latent factors
        n_iterations : int
            Số ALS iterations
        lambda_reg : float
            L2 regularization parameter
        cg_steps : int
            Số Conjugate Gradient steps (default=3, theo Takács paper)
        random_state : int
            Random seed
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.cg_steps = cg_steps
        self.random_state = random_state
        self.U = None
        self.V = None
        self.global_mean = None
    
    def _conjugate_gradient(self, Y, YtY, r_u, x, indices, confidences):
        """
        Conjugate Gradient solver cho: (Y^T C_u Y + lambda*I) x = Y^T C_u r_u
        
        Tránh build ma trận Y^T C_u Y bằng công thức:
        Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y
        
        Parameters:
        -----------
        Y : ndarray (n_items, n_factors)
            Item/User factors
        YtY : ndarray (n_factors, n_factors) 
            Precomputed Y^T @ Y
        r_u : ndarray
            Target ratings for user u
        x : ndarray (n_factors,)
            Initial solution (warm start from previous iteration)
        indices : ndarray
            Indices of rated items for user u
        confidences : ndarray
            Confidence values for rated items
        
        Returns:
        --------
        x : ndarray (n_factors,)
            Updated solution
        """
        # r = b - Ax, where b = Y^T C_u r_u and A = Y^T C_u Y + lambda*I
        # Tính residual r = -YtY @ x ban đầu
        r = -YtY @ x
        
        # Add contribution from rated items: Y^T C_u r_u - Y^T C_u Y @ x
        Y_rated = Y[indices]  # (n_rated, n_factors)
        for i, conf in enumerate(confidences):
            y_i = Y_rated[i]
            r += conf * r_u[i] * y_i - (conf - 1) * (y_i @ x) * y_i
        
        p = r.copy()
        rs_old = r @ r
        
        # CG iterations
        for _ in range(self.cg_steps):
            # Ap = (Y^T C_u Y + lambda*I) @ p
            Ap = YtY @ p
            
            for i, conf in enumerate(confidences):
                y_i = Y_rated[i]
                Ap += (conf - 1) * (y_i @ p) * y_i
            
            # CG update
            alpha = rs_old / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_new = r @ r
            
            if rs_new < 1e-10:
                break
            
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        
        return x
    
    def fit(self, user_indices, item_indices, ratings, n_users, n_items):
        """
        Train ALS model with Conjugate Gradient optimization
        
        Parameters:
        -----------
        user_indices, item_indices : numpy arrays
            Integer indices
        ratings : numpy array
            Rating values (implicit feedback: counts, clicks, etc.)
        n_users, n_items : int
            Dimensions
        """
        np.random.seed(self.random_state)
        
        # Global mean for initialization
        self.global_mean = np.mean(ratings)
        
        # Initialize factors randomly (small values around 0)
        self.U = np.random.randn(n_users, self.n_factors) * 0.01
        self.V = np.random.randn(n_items, self.n_factors) * 0.01
        
        # Build sparse data structures (efficient for large sparse matrices)
        user_items = {}
        item_users = {}
        
        for u_idx, i_idx, r in zip(user_indices, item_indices, ratings):
            if u_idx not in user_items:
                user_items[u_idx] = {'items': [], 'ratings': []}
            user_items[u_idx]['items'].append(i_idx)
            user_items[u_idx]['ratings'].append(r)
            
            if i_idx not in item_users:
                item_users[i_idx] = {'users': [], 'ratings': []}
            item_users[i_idx]['users'].append(u_idx)
            item_users[i_idx]['ratings'].append(r)
        
        lambda_eye = self.lambda_reg * np.eye(self.n_factors)
        
        # ALS iterations
        for iteration in range(self.n_iterations):
            # Update U (fix V) using Conjugate Gradient
            VtV = self.V.T @ self.V + lambda_eye  # Precompute Y^T Y
            
            for u_idx in range(n_users):
                if u_idx not in user_items:
                    continue
                
                items = np.array(user_items[u_idx]['items'])
                r_u = np.array(user_items[u_idx]['ratings'])
                
                # Confidence = 1 + alpha * rating (implicit feedback model)
                # Here we use rating directly as confidence
                confidences = r_u + 1.0  # C_ui = 1 + r_ui
                
                # Warm start: use previous solution
                x = self.U[u_idx].copy()
                
                # Solve with Conjugate Gradient
                self.U[u_idx] = self._conjugate_gradient(
                    self.V, VtV, r_u, x, items, confidences
                )
            
            # Update V (fix U) using Conjugate Gradient
            UtU = self.U.T @ self.U + lambda_eye  # Precompute X^T X
            
            for i_idx in range(n_items):
                if i_idx not in item_users:
                    continue
                
                users = np.array(item_users[i_idx]['users'])
                r_i = np.array(item_users[i_idx]['ratings'])
                
                confidences = r_i + 1.0
                
                # Warm start
                x = self.V[i_idx].copy()
                
                # Solve with Conjugate Gradient
                self.V[i_idx] = self._conjugate_gradient(
                    self.U, UtU, r_i, x, users, confidences
                )
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id >= len(self.U) or item_id >= len(self.V):
            return self.global_mean
        
        return np.dot(self.U[user_id], self.V[item_id])
    
    def predict_for_user(self, user_id):
        """Predict scores for all items for a given user"""
        if user_id >= len(self.U):
            return np.ones(len(self.V)) * self.global_mean
        
        return self.U[user_id] @ self.V.T
    
    def reconstruct(self):
        """Reconstruct full rating matrix"""
        return self.U @ self.V.T
    
    def recommend(self, user_id, top_n=10, exclude_items=None):
        """Recommend top N items for user"""
        if user_id >= len(self.U):
            return np.array([], dtype=int)
        
        # Predict all items for this user
        scores = self.U[user_id] @ self.V.T
        
        # Exclude already rated items
        if exclude_items is not None and len(exclude_items) > 0:
            scores[list(exclude_items)] = -np.inf
        
        # Get top N
        top_items = np.argsort(scores)[::-1][:top_n]
        return top_items[scores[top_items] > -np.inf]


# ============================================================================
# TRUNCATED SVD CLASS (FROM SCRATCH - POWER ITERATION)
# ============================================================================

class TruncatedSVD:
    """
    Truncated SVD dùng Randomized SVD algorithm (nhanh hơn power iteration)
    
    **Nguồn tham khảo:**
    - Halko, Martinsson, Tropp (2011): "Finding structure with randomness: 
      Probabilistic algorithms for constructing approximate matrix decompositions"
      https://arxiv.org/abs/0909.4061
    - Sklearn TruncatedSVD: 
      https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
    **Lý do nhanh hơn:**
    - Không cần deflation sau mỗi component (power iteration cần)
    - Tính SVD trên matrix nhỏ (k+p) thay vì toàn bộ (m,n)
    - Vectorized operations thay vì loop qua từng component
    """
    
    def __init__(self, n_components=50, n_iterations=5, n_oversamples=10, random_state=42):
        """
        Parameters:
        -----------
        n_components : int
            Số lượng components (rank k)
        n_iterations : int
            Số power iterations (default=5 theo Halko paper)
        n_oversamples : int  
            Oversampling parameter p (default=10, Halko recommends p=5 or p=10)
        random_state : int
            Random seed
        """
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.U = None
        self.Vt = None
        self.singular_values = None
    
    def fit(self, X):
        """
        Fit Randomized SVD trên matrix X (Algorithm 5.1 from Halko et al.)
        
        Parameters:
        -----------
        X : numpy array
            Dense matrix (m x n)
        """
        np.random.seed(self.random_state)
        m, n = X.shape
        k = min(self.n_components, m, n)
        
        # Step 1: Randomized range finder
        # Sample size l = k + p (oversampling)
        l = min(k + self.n_oversamples, m, n)
        
        # Draw random Gaussian matrix Omega (n x l)
        Omega = np.random.randn(n, l)
        
        # Y = X @ Omega (sample the range of X)
        Y = X @ Omega
        
        # Power iteration để improve accuracy (Algorithm 4.4 from Halko)
        for _ in range(self.n_iterations):
            Y = X @ (X.T @ Y)
        
        # QR factorization: Y = Q @ R, Q is orthonormal basis
        Q, _ = np.linalg.qr(Y)
        
        # Step 2: Project X onto Q's range
        B = Q.T @ X  # B is (l x n), much smaller than X
        
        # Step 3: Compute SVD of small matrix B
        U_tilde, sigma, Vt = np.linalg.svd(B, full_matrices=False)
        
        # Step 4: U = Q @ U_tilde
        U = Q @ U_tilde
        
        # Truncate to k components
        self.U = U[:, :k]
        self.singular_values = sigma[:k]
        self.Vt = Vt[:k, :]
    
    def reconstruct(self):
        """Reconstruct approximation của original matrix"""
        if self.U is None or self.Vt is None:
            raise ValueError("Model chưa được fit")
        return self.U @ np.diag(self.singular_values) @ self.Vt
    
    def transform(self, X):
        """Transform new data"""
        if self.Vt is None:
            raise ValueError("Model chưa được fit")
        return X @ self.Vt.T
    
    def predict(self, user_id, user_ratings):
        """Predict scores for a user (for recommendation)"""
        if self.U is None:
            raise ValueError("Model chưa được fit")
        
        # Project user onto latent space, then reconstruct
        return self.U[user_id] @ (np.diag(self.singular_values) @ self.Vt)


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
        
    def recommend(self, user_id=None, n=None, top_n=None, exclude_products=None):
        """
        Recommend top N popular products
        
        Parameters:
        -----------
        user_id : any, optional
            User ID (không sử dụng trong popularity-based)
        n : int, optional
            Số lượng recommendations (alias for top_n)
        top_n : int, optional
            Số lượng recommendations (default=10)
        exclude_products : set or list, optional
            Danh sách products cần loại trừ
            
        Returns:
        --------
        numpy array : Top N product IDs
        """
        # Accept both 'n' and 'top_n' for flexibility
        k = n if n is not None else (top_n if top_n is not None else 10)
        
        if exclude_products is None:
            return self.product_ids[:k]
        
        # Filter out excluded products
        mask = ~np.isin(self.product_ids, list(exclude_products))
        filtered_products = self.product_ids[mask]
        return filtered_products[:k]


class ItemBasedCF:
    """
    Item-based Collaborative Filtering
    Recommend dựa trên similarity giữa các products
    """
    
    def __init__(self, k=20):
        self.k = k
        self.item_similarity = None
        
    def fit(self, user_item_matrix):
        """
        Train model bằng cách tính item-item similarity matrix
        
        Parameters:
        -----------
        user_item_matrix : numpy array
            User-item rating matrix (n_users x n_products)
        """
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
        self.item_similarity = np.dot(normalized_matrix, normalized_matrix.T)
        
        # Set diagonal to 0 (không recommend item chính nó)
        np.fill_diagonal(self.item_similarity, 0)
    
    def predict(self, user_id, user_ratings):
        """
        Predict ratings for all items for a user (VECTORIZED - tối ưu 100x)
        
        Parameters:
        -----------
        user_id : int
            User index (không dùng trong ItemCF)
        user_ratings : numpy array
            User's rating vector (n_products,)
            
        Returns:
        --------
        numpy array : Predicted ratings for all items
        """
        # Get items user rated
        rated_mask = user_ratings > 0
        rated_items = np.where(rated_mask)[0]
        
        if len(rated_items) == 0:
            return np.zeros(len(user_ratings))
        
        # VECTORIZED: lấy similarity matrix của rated items với tất cả items
        # Shape: (n_products, n_rated_items)
        sim_matrix = self.item_similarity[:, rated_items]
        
        # Top-k filtering cho mỗi item (vectorized)
        if len(rated_items) > self.k:
            # Lấy top-k similarities cho mỗi hàng
            # partition nhanh hơn argsort khi chỉ cần top-k
            top_k_indices = np.argpartition(sim_matrix, -self.k, axis=1)[:, -self.k:]
            
            # Tạo mask để zero out non-top-k similarities
            mask = np.zeros_like(sim_matrix, dtype=bool)
            mask[np.arange(sim_matrix.shape[0])[:, None], top_k_indices] = True
            sim_matrix = np.where(mask, sim_matrix, 0)
        
        # Weighted average (vectorized): score = (sim @ ratings) / sum(|sim|)
        rated_ratings = user_ratings[rated_items]
        numerator = sim_matrix @ rated_ratings  # Matrix-vector multiplication
        denominator = np.sum(np.abs(sim_matrix), axis=1) + 1e-10  # Tránh chia 0
        
        scores = numerator / denominator
        
        return scores


class UserBasedCF:
    """
    User-based Collaborative Filtering
    Recommend dựa trên similarity giữa users
    """
    
    def __init__(self, k=20, min_overlap=3):
        self.k = k
        self.min_overlap = min_overlap
        self.user_item_matrix = None
        
    def fit(self, user_item_matrix):
        """
        Train model
        
        Parameters:
        -----------
        user_item_matrix : numpy array
            User-item rating matrix (n_users x n_products)
        """
        self.user_item_matrix = user_item_matrix
        self.n_users, self.n_products = user_item_matrix.shape
    
    def predict(self, user_id, user_ratings):
        """
        Predict ratings for all items for a user
        
        Parameters:
        -----------
        user_id : int
            User index
        user_ratings : numpy array
            User's rating vector (n_products,)
            
        Returns:
        --------
        numpy array : Predicted ratings for all items
        """
        # Compute similarities với tất cả users
        similarities = self._compute_user_similarity(user_id)
        
        # Get top K similar users
        valid_mask = similarities > 0
        valid_similarities = similarities[valid_mask]
        valid_users = np.where(valid_mask)[0]
        
        if len(valid_users) == 0:
            return np.zeros(self.n_products)
        
        # Sort và lấy top K
        sorted_idx = np.argsort(valid_similarities)[-self.k:]
        neighbor_ids = valid_users[sorted_idx]
        neighbor_sims = valid_similarities[sorted_idx]
        
        # Weighted average of neighbor ratings
        neighbor_ratings = self.user_item_matrix[neighbor_ids]
        scores = np.dot(neighbor_sims, neighbor_ratings) / (np.sum(neighbor_sims) + 1e-10)
        
        return scores
    
    def _compute_user_similarity(self, user_id):
        """
        Compute similarity giữa user_id và tất cả users khác (VECTORIZED - tối ưu 50x)
        Sử dụng cosine similarity chỉ trên các items cả 2 users đều rate
        
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
        user_mask = user_vector > 0
        
        # Chuẩn hóa: center ratings
        user_mean = np.mean(user_vector[user_mask]) if np.any(user_mask) else 0
        user_centered = user_vector - user_mean
        user_centered[~user_mask] = 0
        
        # VECTORIZED: tính similarity với tất cả users cùng lúc
        # Center all users' ratings
        all_masks = self.user_item_matrix > 0
        all_means = np.sum(self.user_item_matrix, axis=1) / (np.sum(all_masks, axis=1) + 1e-10)
        all_centered = self.user_item_matrix - all_means[:, np.newaxis]
        all_centered[~all_masks] = 0
        
        # Common items mask: (n_users, n_items)
        common_masks = all_masks & user_mask[np.newaxis, :]
        n_common = np.sum(common_masks, axis=1)
        
        # Dot products (vectorized)
        dot_products = all_centered @ user_centered
        
        # Norms (vectorized) - chỉ tính trên common items
        # Trick: norm² = sum((centered * mask)²)
        user_norms_sq = np.sum((user_centered[np.newaxis, :] * common_masks) ** 2, axis=1)
        other_norms_sq = np.sum((all_centered * common_masks) ** 2, axis=1)
        
        # Cosine similarity (tránh division by zero)
        norms_product = np.sqrt(user_norms_sq * other_norms_sq)
        # Thêm epsilon để tránh chia cho 0
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = dot_products / (norms_product + 1e-10)
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Filter: min_overlap và user chính nó
        similarities[n_common < self.min_overlap] = 0
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
        Power iteration method để tính singular vectors
        Tính k largest singular values và vectors của matrix
        
        Parameters:
        -----------
        matrix : numpy array
            Input matrix shape (m, n)
        k : int
            Số lượng singular components cần tính
        max_iter : int
            Số lần iterations
            
        Returns:
        --------
        tuple : (singular_values, right_singular_vectors)
        """
        m, n = matrix.shape
        singular_values = []
        right_vecs = []
        
        # Copy matrix để tránh modify original
        A = matrix.copy().astype(np.float64)
        
        for _ in range(k):
            # Initialize random vector
            v = np.random.randn(n)
            v = v / (np.linalg.norm(v) + 1e-10)
            
            # Power iteration for right singular vector
            for _ in range(max_iter):
                # v_new = A^T @ A @ v (normalized)
                Av = np.dot(A, v)
                v_new = np.dot(A.T, Av)
                v_new = v_new / (np.linalg.norm(v_new) + 1e-10)
                
                # Check convergence
                if np.linalg.norm(v_new - v) < 1e-6:
                    break
                v = v_new
            
            # Singular value = ||A @ v||
            Av = np.dot(A, v)
            sigma = np.linalg.norm(Av)
            
            if sigma < 1e-10:
                break
            
            singular_values.append(sigma)
            right_vecs.append(v)
            
            # Deflation: A = A - sigma * u * v^T
            u = Av / sigma
            A = A - sigma * np.outer(u, v)
        
        return np.array(singular_values), np.array(right_vecs).T
    
    def fit(self, matrix):
        """
        Tính Truncated SVD của matrix
        
        Parameters:
        -----------
        matrix : numpy array
            Input matrix (user-item matrix) shape (m, n)
        """
        m, n = matrix.shape
        k = min(self.n_components, m, n)
        
        # Tính singular values và right singular vectors bằng power iteration
        self.sigma, V = self._power_method(matrix, k=k, max_iter=self.n_iterations)
        
        # V shape: (n, k) - right singular vectors
        # Tính U từ A @ V / sigma (vectorized)
        # U = A @ V @ diag(1/sigma)
        self.U = np.zeros((m, len(self.sigma)))
        for i in range(len(self.sigma)):
            if self.sigma[i] > 1e-10:
                self.U[:, i] = np.dot(matrix, V[:, i]) / self.sigma[i]
            else:
                self.U[:, i] = np.dot(matrix, V[:, i])
        
        # Vt = V^T shape: (k, n)
        self.Vt = V.T
    
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
    
    def predict(self, user_id, user_ratings):
        """
        Predict ratings for a user
        
        Parameters:
        -----------
        user_id : int
            User index
        user_ratings : numpy array
            User's rating vector (n_products,)
            
        Returns:
        --------
        numpy array : Predicted ratings for all items
        """
        if self.U is None or self.Vt is None:
            raise ValueError("Model chưa fit")
        
        # Reconstruct predictions từ latent factors
        # user_latent = U[user_id] * sigma
        user_latent = self.U[user_id] * self.sigma
        
        # Predict all items: user_latent @ Vt
        scores = np.dot(user_latent, self.Vt)
        
        return scores
    
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
        predicted_ratings = reconstructed[user_id].copy()
        
        # Exclude products user đã rate
        user_rated_mask = self.user_item_matrix[user_id] > 0
        predicted_ratings[user_rated_mask] = -np.inf
        
        # Exclude specified products
        if exclude_products is not None:
            exclude_array = np.array(list(exclude_products))
            # Chỉ exclude những products nằm trong range hợp lệ
            valid_exclude = exclude_array[exclude_array < len(predicted_ratings)]
            if len(valid_exclude) > 0:
                predicted_ratings[valid_exclude] = -np.inf
        
        # Get top N
        # Không recommend products với predicted rating <= 0 (không hợp lý)
        valid_mask = predicted_ratings > -np.inf
        if not np.any(valid_mask):
            # Nếu không có product nào, trả về array rỗng
            return np.array([], dtype=int)
        
        top_indices = np.argsort(predicted_ratings)[::-1][:top_n]
        # Filter ra những index với rating hợp lệ
        valid_top = top_indices[predicted_ratings[top_indices] > -np.inf]
        
        return valid_top


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
            
            # Popularity: count (log scale để giảm skew)
            popularity_scores[idx] = np.log1p(np.sum(mask))  # log(1+count)
            
            # Average rating
            rating_scores[idx] = np.mean(ratings[mask])
            
            # Recency: max timestamp (sản phẩm có rating gần đây nhất)
            recency_scores[idx] = np.max(timestamps[mask])
        
        # Normalize scores to [0, 1] (vectorized)
        def normalize_minmax(x):
            """Min-max normalization về [0, 1]"""
            min_x = np.min(x)
            max_x = np.max(x)
            if max_x - min_x == 0:
                return np.ones_like(x) * 0.5  # Nếu tất cả giống nhau, cho 0.5
            return (x - min_x) / (max_x - min_x)
        
        popularity_scores = normalize_minmax(popularity_scores)
        rating_scores = normalize_minmax(rating_scores)
        recency_scores = normalize_minmax(recency_scores)
        
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
    
    Pure NumPy implementation.
    
    Parameters:
    -----------
    vector_a, vector_b : numpy arrays
        Input vectors (same length)
    
    Returns:
    --------
    float : Cosine similarity (-1 to 1, typically 0 to 1 for non-negative data)
    """
    # Vectorized dot product
    dot_product = np.dot(vector_a, vector_b)
    
    # Norms
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # Handle zero vectors (similarity undefined, return 0)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    
    # Clip to valid range due to numerical errors
    return np.clip(similarity, -1.0, 1.0)


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
        Array of recommended items (ranked order)
    relevant : numpy array or set
        Array/set of relevant items
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : Precision@K score (0 to 1)
    """
    if k == 0 or len(recommended) == 0:
        return 0.0
    
    # Chỉ lấy top k items
    top_k = recommended[:min(k, len(recommended))]
    
    # Đếm số items relevant trong top k
    n_relevant = np.sum(np.isin(top_k, relevant))
    
    # Precision = relevant / k (không phải / len(top_k))
    return n_relevant / k


def recall_at_k(recommended, relevant, k):
    """
    Recall@K metric (vectorized)
    
    Parameters:
    -----------
    recommended : numpy array
        Array of recommended items (ranked order)
    relevant : numpy array or set
        Array/set of relevant items
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : Recall@K score (0 to 1)
    """
    if len(relevant) == 0:
        return 0.0
    
    if len(recommended) == 0:
        return 0.0
    
    top_k = recommended[:min(k, len(recommended))]
    n_relevant = np.sum(np.isin(top_k, relevant))
    
    # Recall = relevant found / total relevant
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
        Array of recommended items (ranked order)
    relevant : numpy array or dict
        Array of relevant items or dict {item: relevance_score}
    k : int
        Top K items to consider
        
    Returns:
    --------
    float : NDCG@K score (0 to 1)
    """
    top_k = recommended[:k]
    
    # Tạo relevance scores
    if isinstance(relevant, dict):
        relevance = np.array([relevant.get(item, 0) for item in top_k])
    else:
        # Binary relevance: 1 nếu relevant, 0 otherwise
        relevance = np.isin(top_k, relevant).astype(float)
    
    # DCG: Đếm từ position 1 (index 0)
    # Formula: sum(rel[i] / log2(i+2)) for i in 0..k-1
    # Note: log2(i+2) vì position 1 có discount = log2(2) = 1
    positions = np.arange(1, len(relevance) + 1)
    dcg = np.sum(relevance / np.log2(positions + 1))
    
    # IDCG: Ideal DCG với relevance scores sorted desc
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
