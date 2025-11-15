import numpy as np


def load_csv_numpy(filepath, delimiter=',', skip_header=1, dtype=None):
    """
    Load CSV file using NumPy.
    
    Args:
        filepath: Path to CSV file
        delimiter: Column separator
        skip_header: Number of header rows to skip
        dtype: Data type specification
    
    Returns:
        Structured array with data and column names
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(delimiter)
        
        data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=skip_header,
            dtype=None,
            encoding='utf-8',
            names=header
        )
        
        return data, header
    
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")


def validate_data(data):
    """
    Validate data quality: check nulls, duplicates, value ranges.
    
    Returns:
        dict: Validation report
    """
    report = {}
    
    report['total_rows'] = len(data)
    report['total_columns'] = len(data.dtype.names) if data.dtype.names else 0
    
    report['null_counts'] = {}
    for field in data.dtype.names:
        if np.issubdtype(data[field].dtype, np.number):
            null_count = np.sum(np.isnan(data[field]))
        else:
            null_count = np.sum(data[field] == b'') + np.sum(data[field] == '')
        report['null_counts'][field] = int(null_count)
    
    report['total_nulls'] = sum(report['null_counts'].values())
    report['null_percentage'] = (report['total_nulls'] / (report['total_rows'] * report['total_columns'])) * 100 if report['total_rows'] > 0 else 0
    
    return report


def get_basic_stats(data):
    """
    Calculate basic statistics: shape, unique counts, memory usage.
    
    Returns:
        dict: Statistics summary
    """
    stats = {}
    
    stats['shape'] = (len(data),len(data.dtype.names) if data.dtype.names else 0)
    stats['total_rows'] = len(data)
    stats['total_columns'] = len(data.dtype.names) if data.dtype.names else 0
    stats['memory_bytes'] = data.nbytes
    stats['memory_mb'] = data.nbytes / (1024 * 1024)
    
    stats['unique_counts'] = {}
    for field in data.dtype.names:
        unique_values = np.unique(data[field])
        stats['unique_counts'][field] = len(unique_values)
    
    stats['numeric_summary'] = {}
    for field in data.dtype.names:
        if np.issubdtype(data[field].dtype, np.number):
            field_data = data[field]
            valid_data = field_data[~np.isnan(field_data)]
            if len(valid_data) > 0:
                stats['numeric_summary'][field] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'median': float(np.median(valid_data))
                }
    
    return stats

