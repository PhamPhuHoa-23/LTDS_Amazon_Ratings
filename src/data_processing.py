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

