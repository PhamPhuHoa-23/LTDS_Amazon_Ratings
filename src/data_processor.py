"""
Data Processor Class - OOP Implementation
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Xử lý loading, validation, preprocessing, và feature engineering
Tất cả operations đều vectorized
"""

import numpy as np
from datetime import datetime


class DataProcessor:
    """
    Xử lý dữ liệu từ CSV đến processed arrays (vectorized)
    
    Sử dụng static methods và class methods cho các utility functions
    """
    
    def __init__(self):
        self.data = None
        self.header = None
        self.user_ids = None
        self.product_ids = None
        self.ratings = None
        self.timestamps = None
        
    @staticmethod
    def load_csv(filepath, max_rows=None):
        """
        Load CSV file bằng NumPy (vectorized I/O)
        
        Parameters:
        -----------
        filepath : str
            Đường dẫn tới file CSV
        max_rows : int, optional
            Số lượng rows tối đa
            
        Returns:
        --------
        numpy array : Loaded data
        """
        data = np.genfromtxt(
            filepath,
            delimiter=',',
            skip_header=1,
            dtype=None,
            encoding='utf-8',
            names=['UserId', 'ProductId', 'Rating', 'Timestamp'],
            max_rows=max_rows
        )
        return data
    
    def load_and_extract(self, filepath):
        """
        Load CSV và extract các columns
        """
        self.data = self.load_csv(filepath)
        self.user_ids = self.data['UserId']
        self.product_ids = self.data['ProductId']
        self.ratings = self.data['Rating'].astype(np.float64)
        self.timestamps = self.data['Timestamp'].astype(np.int64)
        
        return self
    
    @staticmethod
    def validate_data(ratings, timestamps, min_rating=1.0, max_rating=5.0):
        """
        Validate dữ liệu (vectorized)
        
        Returns:
        --------
        numpy array : Boolean mask cho valid data
        """
        nan_ratings = np.isnan(ratings)
        nan_timestamps = timestamps == 0
        valid_rating_mask = (ratings >= min_rating) & (ratings <= max_rating)
        
        return ~(nan_ratings | nan_timestamps) & valid_rating_mask
    
    def filter_valid(self):
        """
        Loại bỏ invalid rows từ dataset
        """
        valid_mask = self.validate_data(self.ratings, self.timestamps)
        
        self.user_ids = self.user_ids[valid_mask]
        self.product_ids = self.product_ids[valid_mask]
        self.ratings = self.ratings[valid_mask]
        self.timestamps = self.timestamps[valid_mask]
        
        return self
    
    @staticmethod
    def get_basic_stats(user_ids, product_ids, ratings):
        """
        Tính toán thống kê cơ bản (vectorized)
        
        Returns:
        --------
        dict : Thống kê
        """
        stats = {
            'n_ratings': len(ratings),
            'n_users': len(np.unique(user_ids)),
            'n_products': len(np.unique(product_ids)),
            'mean_rating': np.mean(ratings),
            'median_rating': np.median(ratings),
            'std_rating': np.std(ratings),
            'min_rating': np.min(ratings),
            'max_rating': np.max(ratings)
        }
        return stats
    
    @staticmethod
    def create_mappings(user_ids, product_ids):
        """
        Tạo ID mappings: string → integer (vectorized)
        
        Returns:
        --------
        tuple : (unique_users, unique_products, user_map, product_map)
        """
        unique_users, user_indices = np.unique(user_ids, return_inverse=True)
        unique_products, product_indices = np.unique(product_ids, return_inverse=True)
        
        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        product_map = {pid: idx for idx, pid in enumerate(unique_products)}
        
        return unique_users, unique_products, user_map, product_map, user_indices, product_indices
    
    @staticmethod
    def create_user_item_matrix(user_indices, product_indices, ratings, n_users, n_products):
        """
        Tạo user-item matrix (vectorized)
        """
        matrix = np.zeros((n_users, n_products))
        matrix[user_indices, product_indices] = ratings
        return matrix
    
    @staticmethod
    def compute_user_features(user_indices, ratings):
        """
        Tính user features (vectorized)
        
        Returns:
        --------
        tuple : (user_n_ratings, user_mean_rating, user_std_rating)
        """
        unique_users, user_inverse = np.unique(user_indices, return_inverse=True)
        
        user_rating_counts = np.bincount(user_inverse)
        user_rating_sums = np.bincount(user_inverse, weights=ratings)
        user_avg_ratings = user_rating_sums / np.maximum(user_rating_counts, 1)
        user_rating_sq_sums = np.bincount(user_inverse, weights=ratings**2)
        user_rating_vars = (user_rating_sq_sums / np.maximum(user_rating_counts, 1)) - user_avg_ratings**2
        user_rating_stds = np.sqrt(np.maximum(user_rating_vars, 0))
        
        user_n_ratings = user_rating_counts[user_inverse]
        user_mean_rating = user_avg_ratings[user_inverse]
        user_std_rating = user_rating_stds[user_inverse]
        
        return user_n_ratings, user_mean_rating, user_std_rating
    
    @staticmethod
    def compute_product_features(product_indices, ratings):
        """
        Tính product features (vectorized)
        
        Returns:
        --------
        tuple : (product_n_ratings, product_mean_rating, product_std_rating)
        """
        unique_products, product_inverse = np.unique(product_indices, return_inverse=True)
        
        product_rating_counts = np.bincount(product_inverse)
        product_rating_sums = np.bincount(product_inverse, weights=ratings)
        product_avg_ratings = product_rating_sums / np.maximum(product_rating_counts, 1)
        product_rating_sq_sums = np.bincount(product_inverse, weights=ratings**2)
        product_rating_vars = (product_rating_sq_sums / np.maximum(product_rating_counts, 1)) - product_avg_ratings**2
        product_rating_stds = np.sqrt(np.maximum(product_rating_vars, 0))
        
        product_n_ratings = product_rating_counts[product_inverse]
        product_mean_rating = product_avg_ratings[product_inverse]
        product_std_rating = product_rating_stds[product_inverse]
        
        return product_n_ratings, product_mean_rating, product_std_rating
    
    @staticmethod
    def extract_temporal_features(timestamps):
        """
        Extract temporal features từ timestamps (vectorized)
        
        Returns:
        --------
        tuple : (years, months, weekdays, days_since, recency_weight)
        """
        datetime_array = timestamps.astype('datetime64[s]')
        years = datetime_array.astype('datetime64[Y]').astype(int) + 1970
        months = datetime_array.astype('datetime64[M]').astype(int) % 12 + 1
        weekdays = (datetime_array.astype('datetime64[D]').view('int64') - 4) % 7
        max_timestamp = np.max(timestamps)
        days_since = (max_timestamp - timestamps) / (24 * 3600)
        recency_weight = np.exp(-days_since / 365.0)
        
        return years, months, weekdays, days_since, recency_weight
    
    @staticmethod
    def compute_interaction_features(ratings, user_mean_rating, product_mean_rating):
        """
        Tính interaction features (vectorized)
        
        Returns:
        --------
        tuple : (user_rating_deviation, product_rating_deviation, 
                 global_rating_deviation, user_rating_zscore)
        """
        global_mean_rating = np.mean(ratings)
        
        user_rating_deviation = ratings - user_mean_rating
        product_rating_deviation = ratings - product_mean_rating
        global_rating_deviation = ratings - global_mean_rating
        user_rating_zscore = user_rating_deviation / np.maximum(np.std(user_mean_rating), 0.01)
        
        return user_rating_deviation, product_rating_deviation, global_rating_deviation, user_rating_zscore
    
    @staticmethod
    def minmax_normalize(x):
        """Min-Max normalization về [0, 1] (vectorized)"""
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min == 0:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    @staticmethod
    def zscore_standardize(x):
        """Z-score standardization (vectorized)"""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return np.zeros_like(x)
        return (x - mean) / std
    
    @staticmethod
    def robust_scale(x):
        """Robust scaling dùng median và IQR (vectorized)"""
        median = np.median(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros_like(x)
        return (x - median) / iqr
    
    @staticmethod
    def iterative_filtering(user_ids, product_ids, ratings, timestamps, 
                           all_features_dict, 
                           min_user_ratings=5, min_product_ratings=5):
        """
        Iterative filtering để giảm sparsity (vectorized)
        
        Parameters:
        -----------
        user_ids, product_ids, ratings, timestamps : arrays
        all_features_dict : dict
            Dictionary chứa tất cả feature arrays
        min_user_ratings, min_product_ratings : int
            Ngưỡng tối thiểu
            
        Returns:
        --------
        tuple : (filtered_arrays, feature_dict)
        """
        iteration = 0
        prev_n_records = len(ratings)
        
        while True:
            iteration += 1
            
            # Count ratings (vectorized)
            unique_users_temp, user_inverse_temp = np.unique(user_ids, return_inverse=True)
            user_counts_temp = np.bincount(user_inverse_temp)
            
            unique_products_temp, product_inverse_temp = np.unique(product_ids, return_inverse=True)
            product_counts_temp = np.bincount(product_inverse_temp)
            
            # Create masks (vectorized)
            user_valid = user_counts_temp[user_inverse_temp] >= min_user_ratings
            product_valid = product_counts_temp[product_inverse_temp] >= min_product_ratings
            valid_mask = user_valid & product_valid
            
            # Apply filter (vectorized)
            user_ids = user_ids[valid_mask]
            product_ids = product_ids[valid_mask]
            ratings = ratings[valid_mask]
            timestamps = timestamps[valid_mask]
            
            # Filter all features
            filtered_features = {}
            for key, value in all_features_dict.items():
                if isinstance(value, np.ndarray) and len(value) == len(valid_mask):
                    filtered_features[key] = value[valid_mask]
                else:
                    filtered_features[key] = value
            
            n_removed = prev_n_records - len(ratings)
            
            if n_removed == 0 or iteration > 10:
                break
            
            prev_n_records = len(ratings)
        
        return user_ids, product_ids, ratings, timestamps, filtered_features
    
    @staticmethod
    def compute_sparsity(n_ratings, n_users, n_products):
        """Tính sparsity của matrix (vectorized formula)"""
        total_possible = n_users * n_products
        sparsity = 1 - (n_ratings / total_possible)
        return sparsity


# Backward compatibility: Keep old function-based API
def load_csv_numpy(filepath, max_rows=None):
    """Wrapper function cho backward compatibility"""
    return DataProcessor.load_csv(filepath, max_rows)


def validate_data(ratings, timestamps, min_rating=1.0, max_rating=5.0):
    """Wrapper function cho backward compatibility"""
    return DataProcessor.validate_data(ratings, timestamps, min_rating, max_rating)


def get_basic_stats(user_ids, product_ids, ratings):
    """Wrapper function cho backward compatibility"""
    return DataProcessor.get_basic_stats(user_ids, product_ids, ratings)


def create_user_item_matrix(user_indices, product_indices, ratings, n_users, n_products):
    """Wrapper function cho backward compatibility"""
    return DataProcessor.create_user_item_matrix(user_indices, product_indices, ratings, n_users, n_products)
