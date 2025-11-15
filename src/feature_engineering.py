import numpy as np


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

