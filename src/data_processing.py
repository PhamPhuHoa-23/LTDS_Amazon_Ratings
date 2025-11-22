"""
Data Processing Module
CSC17104 - Programming for Data Science
Sinh viên: Phạm Phú Hòa - MSSV: 23122030

Module chứa functions để load, validate, filter, và preprocess dữ liệu ratings.
Tất cả operations vectorized bằng NumPy.
"""

import numpy as np
from datetime import datetime


# ============================================================================
# DATA PROCESSOR CLASS (OOP)
# ============================================================================

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
        """Load CSV file bằng NumPy (vectorized I/O)"""
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
        """Load CSV và extract các columns"""
        self.data = self.load_csv(filepath)
        self.user_ids = self.data['UserId']
        self.product_ids = self.data['ProductId']
        self.ratings = self.data['Rating'].astype(np.float64)
        self.timestamps = self.data['Timestamp'].astype(np.int64)
        return self
    
    @staticmethod
    def validate_data(ratings, timestamps, min_rating=1.0, max_rating=5.0):
        """Validate dữ liệu (vectorized)"""
        nan_ratings = np.isnan(ratings)
        nan_timestamps = timestamps == 0
        valid_rating_mask = (ratings >= min_rating) & (ratings <= max_rating)
        return ~(nan_ratings | nan_timestamps) & valid_rating_mask
    
    def filter_valid(self):
        """Loại bỏ invalid rows từ dataset"""
        valid_mask = self.validate_data(self.ratings, self.timestamps)
        self.user_ids = self.user_ids[valid_mask]
        self.product_ids = self.product_ids[valid_mask]
        self.ratings = self.ratings[valid_mask]
        self.timestamps = self.timestamps[valid_mask]
        return self
    
    @staticmethod
    def get_basic_stats(user_ids, product_ids, ratings):
        """Tính toán thống kê cơ bản (vectorized)"""
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
        """Tạo ID mappings: string → integer (vectorized)"""
        unique_users, user_indices = np.unique(user_ids, return_inverse=True)
        unique_products, product_indices = np.unique(product_ids, return_inverse=True)
        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        product_map = {pid: idx for idx, pid in enumerate(unique_products)}
        return unique_users, unique_products, user_map, product_map, user_indices, product_indices
    
    @staticmethod
    def create_user_item_matrix(user_indices, product_indices, ratings, n_users, n_products):
        """Tạo user-item matrix (vectorized)"""
        matrix = np.zeros((n_users, n_products))
        matrix[user_indices, product_indices] = ratings
        return matrix
    
    @staticmethod
    def compute_user_features(user_indices, ratings):
        """Tính user features (vectorized)"""
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
        """Tính product features (vectorized)"""
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
        """Extract temporal features từ timestamps (vectorized)"""
        # Convert sang datetime64 (chuẩn hơn và nhanh hơn)
        datetime_array = timestamps.astype('datetime64[s]')
        
        # Extract year (vectorized)
        years = datetime_array.astype('datetime64[Y]').astype(int) + 1970
        
        # Extract month (vectorized): 1-12
        months = datetime_array.astype('datetime64[M]').astype(int) % 12 + 1
        
        # Extract weekday (vectorized): 0=Monday, 6=Sunday
        # datetime64[D] epoch là 1970-01-01 (Thursday = 3)
        # Cộng thêm offset để Monday = 0
        days_since_epoch = datetime_array.astype('datetime64[D]').view('int64')
        weekdays = (days_since_epoch + 3) % 7  # +3 vì epoch là Thursday
        
        # Recency: days since most recent timestamp
        max_timestamp = np.max(timestamps)
        days_since = (max_timestamp - timestamps) / (24 * 3600)
        
        # Recency weight: exponential decay (recent = higher weight)
        # exp(-days/365) means 1 year ago ≈ 0.368, recent ≈ 1.0
        recency_weight = np.exp(-days_since / 365.0)
        
        return years, months, weekdays, days_since, recency_weight
    
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
    def compute_sparsity(n_ratings, n_users, n_products):
        """Tính sparsity của matrix (vectorized formula)"""
        total_possible = n_users * n_products
        sparsity = 1 - (n_ratings / total_possible)
        return sparsity


# ============================================================================
# UTILITY FUNCTIONS (BACKWARD COMPATIBILITY & LEGACY API)
# ============================================================================

def load_csv_numpy(filepath, max_rows=None):
    """
    Load CSV file bằng NumPy (vectorized I/O)
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn tới file CSV
    max_rows : int, optional
        Số lượng rows tối đa để load (dùng cho sampling)
        
    Returns:
    --------
    data : numpy structured array
        Dữ liệu đã load
    header : list
        Tên các columns
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
    
    header = ['UserId', 'ProductId', 'Rating', 'Timestamp']
    
    return data, header


def validate_data(data):
    """
    Validate dữ liệu và phát hiện missing values (vectorized)
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu cần validate
        
    Returns:
    --------
    validation_info : dict
        Dictionary chứa thông tin validation
    """
    ratings = data['Rating'].astype(np.float64)
    timestamps = data['Timestamp'].astype(np.int64)
    
    nan_ratings = np.sum(np.isnan(ratings))
    nan_timestamps = np.sum(timestamps == 0)
    
    total_nulls = nan_ratings + nan_timestamps
    total_values = len(data) * 2  # 2 numeric columns
    
    return {
        'total_nulls': total_nulls,
        'null_percentage': (total_nulls / total_values) * 100,
        'null_counts': {
            'Rating': nan_ratings,
            'Timestamp': nan_timestamps
        }
    }


def get_basic_stats(data):
    """
    Tính toán basic statistics (vectorized)
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu cần phân tích
        
    Returns:
    --------
    stats : dict
        Dictionary chứa các statistics cơ bản
    """
    n_rows = len(data)
    n_cols = len(data.dtype.names)
    
    unique_counts = {
        'UserId': len(np.unique(data['UserId'])),
        'ProductId': len(np.unique(data['ProductId'])),
        'Rating': len(np.unique(data['Rating'])),
        'Timestamp': len(np.unique(data['Timestamp']))
    }
    
    memory_mb = data.nbytes / (1024 ** 2)
    
    return {
        'total_rows': n_rows,
        'total_columns': n_cols,
        'memory_mb': memory_mb,
        'unique_counts': unique_counts
    }


def extract_arrays(data):
    """
    Extract arrays từ structured array (vectorized)
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu nguồn
        
    Returns:
    --------
    tuple : (user_ids, product_ids, ratings, timestamps)
        Các arrays đã extract
    """
    user_ids = data['UserId']
    product_ids = data['ProductId']
    ratings = data['Rating'].astype(np.float64)
    timestamps = data['Timestamp'].astype(np.int64)
    
    return user_ids, product_ids, ratings, timestamps


def clean_data(user_ids, product_ids, ratings, timestamps):
    """
    Clean dữ liệu: loại bỏ missing values và outliers (vectorized)
    
    Parameters:
    -----------
    user_ids, product_ids, ratings, timestamps : numpy arrays
        Dữ liệu cần clean
        
    Returns:
    --------
    tuple : Cleaned arrays
    """
    valid_rating = ~np.isnan(ratings) & (ratings >= 1.0) & (ratings <= 5.0)
    valid_timestamp = (timestamps > 0) & (timestamps < 2000000000)
    
    valid_mask = valid_rating & valid_timestamp
    
    return (
        user_ids[valid_mask],
        product_ids[valid_mask],
        ratings[valid_mask],
        timestamps[valid_mask]
    )


def filter_by_min_ratings(user_ids, product_ids, ratings, timestamps, 
                          min_user_ratings=5, min_product_ratings=5,
                          max_iterations=10):
    """
    Filter users và products có ít ratings (vectorized, iterative)
    Iterative vì sau khi loại user, số ratings của product giảm, và ngược lại
    
    Parameters:
    -----------
    user_ids, product_ids, ratings, timestamps : numpy arrays
        Dữ liệu cần filter
    min_user_ratings : int
        Số ratings tối thiểu cho user (default: 5)
    min_product_ratings : int
        Số ratings tối thiểu cho product (default: 5)
    max_iterations : int
        Số iterations tối đa (default: 10)
        
    Returns:
    --------
    tuple : Filtered arrays (user_ids, product_ids, ratings, timestamps)
    """
    iteration = 0
    initial_count = len(ratings)
    
    while iteration < max_iterations:
        prev_n = len(ratings)
        
        # Count ratings per user (vectorized)
        unique_users, user_inv = np.unique(user_ids, return_inverse=True)
        user_counts = np.bincount(user_inv)
        
        # Count ratings per product (vectorized)
        unique_products, product_inv = np.unique(product_ids, return_inverse=True)
        product_counts = np.bincount(product_inv)
        
        # Create masks for valid users and products
        user_valid = user_counts[user_inv] >= min_user_ratings
        product_valid = product_counts[product_inv] >= min_product_ratings
        
        # Combined mask
        valid_mask = user_valid & product_valid
        
        # Apply filter
        user_ids = user_ids[valid_mask]
        product_ids = product_ids[valid_mask]
        ratings = ratings[valid_mask]
        timestamps = timestamps[valid_mask]
        
        # Check convergence: nếu không thay đổi thì dừng
        if len(ratings) == prev_n:
            break
            
        iteration += 1
    
    removed_count = initial_count - len(ratings)
    removed_pct = (removed_count / initial_count * 100) if initial_count > 0 else 0
    
    # Không cần print vì sẽ gọi ở notebook level
    
    return user_ids, product_ids, ratings, timestamps


def create_id_mappings(user_ids, product_ids):
    """
    Tạo mappings từ string IDs sang integer indices (vectorized)
    
    Parameters:
    -----------
    user_ids, product_ids : numpy arrays
        String IDs cần map
        
    Returns:
    --------
    tuple : (unique_users, unique_products, user_indices, product_indices)
    """
    unique_users, user_indices = np.unique(user_ids, return_inverse=True)
    unique_products, product_indices = np.unique(product_ids, return_inverse=True)
    
    return unique_users, unique_products, user_indices, product_indices


def load_processed_data(data_dir='../data/processed/'):
    """
    Load preprocessed data từ file đã save
    
    Parameters:
    -----------
    data_dir : str
        Thư mục chứa preprocessed data
        
    Returns:
    --------
    dict : Dictionary chứa tất cả data và features
    """
    data = np.load(data_dir + 'preprocessed_data.npz')
    mappings = np.load(data_dir + 'id_mappings.npz')
    metadata = np.load(data_dir + 'metadata.npy', allow_pickle=True).item()
    
    return {
        'data': data,
        'mappings': mappings,
        'metadata': metadata
    }


def sample_data(user_ids, product_ids, ratings, timestamps, n_samples=100000, random_seed=42):
    """
    Sample dữ liệu cho development (vectorized)
    
    Parameters:
    -----------
    user_ids, product_ids, ratings, timestamps : numpy arrays
        Dữ liệu nguồn
    n_samples : int
        Số samples cần lấy
    random_seed : int
        Random seed để reproducibility
        
    Returns:
    --------
    tuple : Sampled arrays
    """
    np.random.seed(random_seed)
    
    n_total = len(ratings)
    if n_samples >= n_total:
        return user_ids, product_ids, ratings, timestamps
    
    indices = np.random.choice(n_total, size=n_samples, replace=False)
    
    return (
        user_ids[indices],
        product_ids[indices],
        ratings[indices],
        timestamps[indices]
    )


def train_test_split(user_indices, product_indices, ratings, timestamps,
                     test_size=0.2, random_seed=42):
    """
    Split data thành train và test sets (vectorized)
    
    Parameters:
    -----------
    user_indices, product_indices, ratings, timestamps : numpy arrays
        Dữ liệu cần split
    test_size : float
        Tỷ lệ test set (0.0 - 1.0)
    random_seed : int
        Random seed
        
    Returns:
    --------
    dict : Dictionary chứa train và test data
    """
    np.random.seed(random_seed)
    
    n_total = len(ratings)
    n_test = int(n_total * test_size)
    
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return {
        'train': {
            'user_indices': user_indices[train_idx],
            'product_indices': product_indices[train_idx],
            'ratings': ratings[train_idx],
            'timestamps': timestamps[train_idx]
        },
        'test': {
            'user_indices': user_indices[test_idx],
            'product_indices': product_indices[test_idx],
            'ratings': ratings[test_idx],
            'timestamps': timestamps[test_idx]
        }
    }


# ============================================================================
# MISSING VALUE HANDLING
# ============================================================================

def detect_missing_values(data):
    """
    Detect missing, null, NaN values in data.
    
    Returns:
        Boolean array indicating missing positions
    """
    if data.dtype.names:
        missing_mask = {}
        for field in data.dtype.names:
            if np.issubdtype(data[field].dtype, np.number):
                missing_mask[field] = np.isnan(data[field])
            else:
                missing_mask[field] = (data[field] == b'') | (data[field] == '')
        return missing_mask
    else:
        if np.issubdtype(data.dtype, np.number):
            return np.isnan(data)
        else:
            return (data == b'') | (data == '')


def handle_missing_values(data, strategy='mean', fill_value=0):
    """
    Handle missing values in numpy array
    
    Args:
        data: Input array
        strategy: 'mean', 'median', 'constant'
        fill_value: Value to use if strategy='constant'
    
    Returns:
        Array with missing values handled
    """
    mask = np.isnan(data)
    
    if strategy == 'mean':
        fill_val = np.nanmean(data)
    elif strategy == 'median':
        fill_val = np.nanmedian(data)
    else:
        fill_val = fill_value
    
    data_filled = data.copy()
    data_filled[mask] = fill_val
    
    return data_filled


def impute_missing_mean(data, column_name=None):
    """
    Fill missing values with column mean for numeric data.
    
    Args:
        data: NumPy array
        column_name: Column name (for structured arrays)
    
    Returns:
        Data with missing values imputed
    """
    if column_name and data.dtype.names:
        col_data = data[column_name].copy()
        valid_data = col_data[~np.isnan(col_data)]
        if len(valid_data) > 0:
            mean_value = np.mean(valid_data)
            col_data[np.isnan(col_data)] = mean_value
        return col_data
    else:
        data_copy = data.copy()
        valid_data = data_copy[~np.isnan(data_copy)]
        if len(valid_data) > 0:
            mean_value = np.mean(valid_data)
            data_copy[np.isnan(data_copy)] = mean_value
        return data_copy


def impute_missing_median(data, column_name=None):
    """
    Fill missing values with column median for numeric data.
    
    Args:
        data: NumPy array
        column_name: Column name (for structured arrays)
    
    Returns:
        Data with missing values imputed
    """
    if column_name and data.dtype.names:
        col_data = data[column_name].copy()
        valid_data = col_data[~np.isnan(col_data)]
        if len(valid_data) > 0:
            median_value = np.median(valid_data)
            col_data[np.isnan(col_data)] = median_value
        return col_data
    else:
        data_copy = data.copy()
        valid_data = data_copy[~np.isnan(data_copy)]
        if len(valid_data) > 0:
            median_value = np.median(valid_data)
            data_copy[np.isnan(data_copy)] = median_value
        return data_copy


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers_iqr(data, column_name=None, multiplier=1.5):
    """
    Detect outliers using IQR (Interquartile Range) method.
    
    Args:
        data: NumPy array
        column_name: Column name (for structured arrays)
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        Boolean array: True for outliers
    """
    if column_name and data.dtype.names:
        col_data = data[column_name]
    else:
        col_data = data
    
    if not np.issubdtype(col_data.dtype, np.number):
        raise ValueError("IQR method only works with numeric data")
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = (col_data < lower_bound) | (col_data > upper_bound)
    return outliers


def detect_outliers_zscore(data, column_name=None, threshold=3.0):
    """
    Detect outliers using Z-score method.
    
    Args:
        data: NumPy array
        column_name: Column name (for structured arrays)
        threshold: Z-score threshold (default 3.0)
    
    Returns:
        Boolean array: True for outliers
    """
    if column_name and data.dtype.names:
        col_data = data[column_name]
    else:
        col_data = data
    
    if not np.issubdtype(col_data.dtype, np.number):
        raise ValueError("Z-score method only works with numeric data")
    
    mean = np.mean(col_data)
    std = np.std(col_data)
    
    if std == 0:
        return np.zeros_like(col_data, dtype=bool)
    
    z_scores = np.abs((col_data - mean) / std)
    outliers = z_scores > threshold
    
    return outliers


def remove_outliers(data, outlier_mask):
    """
    Remove outliers from data based on boolean mask.
    
    Args:
        data: NumPy array
        outlier_mask: Boolean array indicating outliers
    
    Returns:
        Data with outliers removed
    """
    if data.dtype.names:
        filtered_data = data[~outlier_mask]
    else:
        filtered_data = data[~outlier_mask]
    
    return filtered_data


# ============================================================================
# NORMALIZATION AND STANDARDIZATION
# ============================================================================

def normalize_minmax(data, feature_min=0, feature_max=1):
    """
    Min-Max normalization: scale to [feature_min, feature_max].
    
    Formula: X_norm = (X - X_min) / (X_max - X_min) * (max - min) + min
    
    Args:
        data: NumPy array
        feature_min: Minimum value of normalized range
        feature_max: Maximum value of normalized range
    
    Returns:
        Normalized data
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return np.full_like(data, (feature_min + feature_max) / 2)
    
    normalized = (data - data_min) / (data_max - data_min)
    normalized = normalized * (feature_max - feature_min) + feature_min
    
    return normalized


def normalize_log(data, base='e'):
    """
    Log transformation for skewed distributions.
    
    Args:
        data: NumPy array (must be positive)
        base: Log base ('e' for natural log, 10 for log10)
    
    Returns:
        Log-transformed data
    """
    if np.any(data <= 0):
        data = data - np.min(data) + 1
    
    if base == 'e':
        return np.log(data)
    elif base == 10:
        return np.log10(data)
    else:
        raise ValueError("Base must be 'e' or 10")


def standardize_zscore(data):
    """
    Z-score standardization: mean=0, std=1.
    
    Formula: Z = (X - μ) / σ
    
    Args:
        data: NumPy array
    
    Returns:
        Standardized data
    """
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return np.zeros_like(data)
    
    standardized = (data - mean) / std
    return standardized


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def unix_to_datetime_features(timestamps):
    """
    Convert Unix timestamps to datetime features.
    
    Args:
        timestamps: Array of Unix timestamps (seconds since 1970)
    
    Returns:
        Dictionary of datetime features as NumPy arrays
    """
    features = {}
    features['year'] = np.array([datetime.fromtimestamp(ts).year for ts in timestamps])
    features['month'] = np.array([datetime.fromtimestamp(ts).month for ts in timestamps])
    features['day'] = np.array([datetime.fromtimestamp(ts).day for ts in timestamps])
    features['weekday'] = np.array([datetime.fromtimestamp(ts).weekday() for ts in timestamps])
    features['hour'] = np.array([datetime.fromtimestamp(ts).hour for ts in timestamps])
    
    return features


def compute_user_stats(data, user_id_col='UserId', rating_col='Rating'):
    """
    Compute per-user statistics.
    
    Returns:
        Dictionary of user statistics arrays
    """
    user_ids = data[user_id_col]
    ratings = data[rating_col]
    
    unique_users = np.unique(user_ids)
    stats = {
        'user_id': unique_users,
        'total_ratings': np.zeros(len(unique_users)),
        'avg_rating': np.zeros(len(unique_users)),
        'std_rating': np.zeros(len(unique_users)),
        'min_rating': np.zeros(len(unique_users)),
        'max_rating': np.zeros(len(unique_users))
    }
    
    for i, user_id in enumerate(unique_users):
        user_mask = user_ids == user_id
        user_ratings = ratings[user_mask]
        
        stats['total_ratings'][i] = len(user_ratings)
        stats['avg_rating'][i] = np.mean(user_ratings)
        stats['std_rating'][i] = np.std(user_ratings) if len(user_ratings) > 1 else 0.0
        stats['min_rating'][i] = np.min(user_ratings)
        stats['max_rating'][i] = np.max(user_ratings)
    
    return stats


def compute_product_stats(data, product_id_col='ProductId', rating_col='Rating'):
    """
    Compute per-product statistics.
    
    Returns:
        Dictionary of product statistics arrays
    """
    product_ids = data[product_id_col]
    ratings = data[rating_col]
    
    unique_products = np.unique(product_ids)
    stats = {
        'product_id': unique_products,
        'total_ratings': np.zeros(len(unique_products)),
        'avg_rating': np.zeros(len(unique_products)),
        'std_rating': np.zeros(len(unique_products)),
        'min_rating': np.zeros(len(unique_products)),
        'max_rating': np.zeros(len(unique_products))
    }
    
    for i, product_id in enumerate(unique_products):
        product_mask = product_ids == product_id
        product_ratings = ratings[product_mask]
        
        stats['total_ratings'][i] = len(product_ratings)
        stats['avg_rating'][i] = np.mean(product_ratings)
        stats['std_rating'][i] = np.std(product_ratings) if len(product_ratings) > 1 else 0.0
        stats['min_rating'][i] = np.min(product_ratings)
        stats['max_rating'][i] = np.max(product_ratings)
    
    return stats


def compute_rating_deviation(user_rating, user_avg, product_avg):
    """
    Deviation of rating from user and product averages.
    Personalization signal.
    
    Args:
        user_rating: Individual rating value
        user_avg: User's average rating
        product_avg: Product's average rating
    
    Returns:
        Deviation score
    """
    global_avg = (user_avg + product_avg) / 2
    deviation = user_rating - global_avg
    return deviation


def compute_recency_score(timestamps, decay_factor=0.1):
    """
    Time-based weighting: recent ratings more important.
    Dùng exponential decay theo thời gian.
    
    Args:
        timestamps: Array of Unix timestamps
        decay_factor: Exponential decay factor (higher = faster decay)
                      Default 0.1 means half-life ~6.93 time units
    
    Returns:
        Recency scores (higher = more recent, range 0-1)
    """
    max_ts = np.max(timestamps)
    # Time differences in days
    time_diffs_days = (max_ts - timestamps) / (24 * 3600)
    
    # Exponential decay: score = exp(-decay_factor * days_ago / 365)
    # Ví dụ: với decay=0.1, sau 1 năm score = exp(-0.1) ≈ 0.90
    #         sau 10 năm score = exp(-1.0) ≈ 0.37
    recency_scores = np.exp(-decay_factor * time_diffs_days / 365.0)
    
    return recency_scores


def compute_rating_velocity(timestamps, window_days=30):
    """
    Ratings per time period (trending indicator).
    
    Args:
        timestamps: Array of Unix timestamps
        window_days: Time window in days
    
    Returns:
        Rating velocity (ratings per window)
    """
    window_seconds = window_days * 24 * 3600
    min_ts = np.min(timestamps)
    max_ts = np.max(timestamps)
    
    num_windows = int((max_ts - min_ts) / window_seconds) + 1
    window_counts = np.zeros(num_windows)
    
    for ts in timestamps:
        window_idx = int((ts - min_ts) / window_seconds)
        if window_idx < num_windows:
            window_counts[window_idx] += 1
    
    return window_counts
