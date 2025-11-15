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

