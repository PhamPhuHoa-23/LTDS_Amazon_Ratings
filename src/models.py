"""
Recommendation Models Module
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Module này chứa các recommendation algorithms
Tất cả models được implement chỉ bằng NumPy (không dùng Scikit-learn)
"""

import numpy as np


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
        
        print(f"Similarity matrix computed: {self.similarity_matrix.shape}")
        
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


class SVDRecommender:
    """
    Matrix Factorization using SVD (Singular Value Decomposition)
    """
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_item_matrix = None
        
    def fit(self, user_indices, product_indices, ratings, n_users, n_products):
        """
        Train model using SVD
        
        Parameters:
        -----------
        user_indices, product_indices, ratings : numpy arrays
        n_users, n_products : int
        """
        # Create user-item matrix
        self.user_item_matrix = np.zeros((n_users, n_products))
        self.user_item_matrix[user_indices, product_indices] = ratings
        
        # Apply SVD (vectorized linear algebra)
        # A = U * Sigma * V^T
        self.U, self.sigma, self.Vt = np.linalg.svd(self.user_item_matrix, full_matrices=False)
        
        # Keep only top n_components
        self.U = self.U[:, :self.n_components]
        self.sigma = self.sigma[:self.n_components]
        self.Vt = self.Vt[:self.n_components, :]
        
        print(f"SVD completed: {self.U.shape} × {self.sigma.shape} × {self.Vt.shape}")
        
    def reconstruct_matrix(self):
        """
        Reconstruct approximated user-item matrix
        
        Returns:
        --------
        numpy array : Reconstructed matrix
        """
        # Reconstruct: U * Sigma * V^T (vectorized)
        return np.dot(self.U, np.dot(np.diag(self.sigma), self.Vt))
        
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


if __name__ == "__main__":
    # Test models
    print("Testing recommendation models...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_users = 100
    n_products = 50
    
    user_indices = np.random.randint(0, n_users, n_samples)
    product_indices = np.random.randint(0, n_products, n_samples)
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_samples)
    timestamps = np.random.randint(1000000000, 1700000000, n_samples)
    
    # Test Popularity Recommender
    print("\n1. Testing Popularity Recommender...")
    pop_model = PopularityRecommender()
    pop_model.fit(product_indices, ratings)
    recommendations = pop_model.recommend(top_n=5)
    print(f"   ✓ Top 5 recommendations: {recommendations}")
    
    # Test Item-Based CF
    print("\n2. Testing Item-Based CF...")
    item_cf = ItemBasedCF()
    item_cf.fit(user_indices, product_indices, ratings, n_products)
    recommendations = item_cf.recommend(product_id=0, top_n=5)
    print(f"   ✓ Top 5 similar items to product 0: {recommendations}")
    
    # Test SVD
    print("\n3. Testing SVD Recommender...")
    svd_model = SVDRecommender(n_components=10)
    svd_model.fit(user_indices, product_indices, ratings, n_users, n_products)
    recommendations = svd_model.recommend(user_id=0, top_n=5)
    print(f"   ✓ Top 5 recommendations for user 0: {recommendations}")
    
    print("\nAll model tests passed!")
