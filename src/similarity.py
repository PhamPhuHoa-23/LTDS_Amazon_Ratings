import numpy as np


def create_user_item_matrix(user_ids, product_ids, ratings):
    """
    Create sparse user-item matrix.
    Rows: users, Columns: products, Values: ratings
    
    Returns:
        matrix, user_map, product_map
    """
    unique_users = np.unique(user_ids)
    unique_products = np.unique(product_ids)
    
    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    product_map = {product_id: idx for idx, product_id in enumerate(unique_products)}
    
    n_users = len(unique_users)
    n_products = len(unique_products)
    
    matrix = np.zeros((n_users, n_products))
    
    for user_id, product_id, rating in zip(user_ids, product_ids, ratings):
        user_idx = user_map[user_id]
        product_idx = product_map[product_id]
        matrix[user_idx, product_idx] = rating
    
    return matrix, user_map, product_map

