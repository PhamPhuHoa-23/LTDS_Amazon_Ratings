"""
Data Loading and Validation Module
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Module này chứa các functions để load và validate dữ liệu
Tất cả operations đều vectorized để tối ưu performance
"""

import numpy as np
from datetime import datetime


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
    # Load toàn bộ file một lần (vectorized)
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
    # Vectorized null detection
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
    
    # Vectorized unique counts
    unique_counts = {
        'UserId': len(np.unique(data['UserId'])),
        'ProductId': len(np.unique(data['ProductId'])),
        'Rating': len(np.unique(data['Rating'])),
        'Timestamp': len(np.unique(data['Timestamp']))
    }
    
    # Memory usage
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
    # Vectorized cleaning
    valid_rating = ~np.isnan(ratings) & (ratings >= 1.0) & (ratings <= 5.0)
    valid_timestamp = (timestamps > 0) & (timestamps < 2000000000)  # Reasonable range
    
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
    
    Parameters:
    -----------
    user_ids, product_ids, ratings, timestamps : numpy arrays
        Dữ liệu cần filter
    min_user_ratings : int
        Số ratings tối thiểu cho user
    min_product_ratings : int
        Số ratings tối thiểu cho product
    max_iterations : int
        Số iterations tối đa
        
    Returns:
    --------
    tuple : Filtered arrays
    """
    iteration = 0
    
    while iteration < max_iterations:
        prev_n = len(ratings)
        
        # Count ratings per user (vectorized)
        unique_users, user_inv = np.unique(user_ids, return_inverse=True)
        user_counts = np.bincount(user_inv)
        
        # Count ratings per product (vectorized)
        unique_products, product_inv = np.unique(product_ids, return_inverse=True)
        product_counts = np.bincount(product_inv)
        
        # Create masks (vectorized)
        user_valid = user_counts[user_inv] >= min_user_ratings
        product_valid = product_counts[product_inv] >= min_product_ratings
        
        valid_mask = user_valid & product_valid
        
        # Apply filter
        user_ids = user_ids[valid_mask]
        product_ids = product_ids[valid_mask]
        ratings = ratings[valid_mask]
        timestamps = timestamps[valid_mask]
        
        # Check convergence
        if len(ratings) == prev_n:
            break
            
        iteration += 1
    
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
    # Vectorized mapping
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
    # Load main data
    data = np.load(data_dir + 'preprocessed_data.npz')
    
    # Load mappings
    mappings = np.load(data_dir + 'id_mappings.npz')
    
    # Load metadata
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
    
    # Vectorized sampling
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
    
    # Vectorized shuffle
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


if __name__ == "__main__":
    # Test functions
    print("Testing data_loader module...")
    
    # Test load
    try:
        data, header = load_csv_numpy('../data/raw/ratings_Beauty.csv', max_rows=1000)
        print(f"✓ Load successful: {len(data)} rows")
        print(f"  Columns: {header}")
    except Exception as e:
        print(f"✗ Load failed: {e}")
    
    # Test validation
    validation = validate_data(data)
    print(f"✓ Validation completed")
    print(f"  Null percentage: {validation['null_percentage']:.2f}%")
    
    # Test stats
    stats = get_basic_stats(data)
    print(f"✓ Stats computed")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")
    
    print("\nAll tests passed!")
