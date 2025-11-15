import numpy as np


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


def unix_to_datetime_features(timestamps):
    """
    Convert Unix timestamps to datetime features.
    
    Args:
        timestamps: Array of Unix timestamps (seconds since 1970)
    
    Returns:
        Dictionary of datetime features as NumPy arrays
    """
    from datetime import datetime
    
    features = {}
    features['year'] = np.array([datetime.fromtimestamp(ts).year for ts in timestamps])
    features['month'] = np.array([datetime.fromtimestamp(ts).month for ts in timestamps])
    features['day'] = np.array([datetime.fromtimestamp(ts).day for ts in timestamps])
    features['weekday'] = np.array([datetime.fromtimestamp(ts).weekday() for ts in timestamps])
    features['hour'] = np.array([datetime.fromtimestamp(ts).hour for ts in timestamps])
    
    return features


def filter_by_min_ratings(data, user_id_col='UserId', product_id_col='ProductId', 
                          min_user_ratings=5, min_product_ratings=5):
    """
    Filter users and products with minimum rating counts.
    Reduces sparsity.
    
    Args:
        data: Structured NumPy array with user and product IDs
        user_id_col: Column name for user IDs
        product_id_col: Column name for product IDs
        min_user_ratings: Minimum ratings per user
        min_product_ratings: Minimum ratings per product
    
    Returns:
        Filtered data array
    """
    user_ids = data[user_id_col]
    product_ids = data[product_id_col]
    
    unique_users, user_counts = np.unique(user_ids, return_counts=True)
    unique_products, product_counts = np.unique(product_ids, return_counts=True)
    
    valid_users = unique_users[user_counts >= min_user_ratings]
    valid_products = unique_products[product_counts >= min_product_ratings]
    
    user_mask = np.isin(user_ids, valid_users)
    product_mask = np.isin(product_ids, valid_products)
    
    filtered_mask = user_mask & product_mask
    filtered_data = data[filtered_mask]
    
    return filtered_data

