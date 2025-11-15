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

