"""
Evaluation Metrics Module
CSC17104 - Programming for Data Science
Sinh viên: Phạm Phú Hòa - MSSV: 23122030

Module chứa các metrics để đánh giá recommendation models:
- Precision@K, Recall@K, F1@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Coverage, Diversity
"""

import numpy as np


def precision_at_k(relevant, recommended, k):
    """
    Precision@K: tỷ lệ items recommended có trong relevant set
    
    Parameters:
    -----------
    relevant : list or set
        Relevant items (ground truth)
    recommended : list
        Recommended items (ranked)
    k : int
        Top-K cutoff
        
    Returns:
    --------
    float : Precision score
    """
    if len(recommended) == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / min(k, len(recommended_k))


def recall_at_k(relevant, recommended, k):
    """
    Recall@K: tỷ lệ relevant items được recommend
    
    Parameters:
    -----------
    relevant : list or set
        Relevant items (ground truth)
    recommended : list
        Recommended items (ranked)
    k : int
        Top-K cutoff
        
    Returns:
    --------
    float : Recall score
    """
    if len(relevant) == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def f1_at_k(relevant, recommended, k):
    """
    F1@K: harmonic mean của Precision và Recall
    
    Parameters:
    -----------
    relevant : list or set
        Relevant items
    recommended : list
        Recommended items
    k : int
        Top-K cutoff
        
    Returns:
    --------
    float : F1 score
    """
    p = precision_at_k(relevant, recommended, k)
    r = recall_at_k(relevant, recommended, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(relevant, recommended, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain
    Đánh giá cả ranking order (items rank cao hơn có weight lớn hơn)
    
    Parameters:
    -----------
    relevant : list or set
        Relevant items
    recommended : list
        Recommended items (ranked)
    k : int
        Top-K cutoff
        
    Returns:
    --------
    float : NDCG score
    """
    recommended_k = recommended[:k]
    
    # DCG: sum(rel_i / log2(i+2)) cho items trong recommended_k
    dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(recommended_k) 
               if item in relevant])
    
    # IDCG: DCG lý tưởng (tất cả relevant items rank đầu)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommended, n_products):
    """
    Coverage: tỷ lệ unique products được recommend (diversity ở catalog level)
    
    Parameters:
    -----------
    all_recommended : list of lists
        Danh sách recommendations cho tất cả users
    n_products : int
        Tổng số products trong catalog
        
    Returns:
    --------
    float : Coverage score (0-1)
    """
    unique_items = set()
    for recs in all_recommended:
        unique_items.update(recs)
    return len(unique_items) / n_products


def diversity(all_recommended, item_similarity_matrix):
    """
    Diversity: trung bình dissimilarity giữa các recommended pairs
    Đo độ đa dạng trong recommendations
    
    Parameters:
    -----------
    all_recommended : list of lists
        Danh sách recommendations cho tất cả users
    item_similarity_matrix : numpy array
        Matrix similarity giữa items (n_products x n_products)
        
    Returns:
    --------
    float : Diversity score (0-1)
    """
    if len(all_recommended) == 0:
        return 0.0
    
    total_diversity = 0.0
    count = 0
    
    for recs in all_recommended:
        if len(recs) < 2:
            continue
        
        # Pairwise dissimilarity
        for i in range(len(recs)):
            for j in range(i+1, len(recs)):
                # Kiểm tra indices hợp lệ
                if (recs[i] < len(item_similarity_matrix) and 
                    recs[j] < len(item_similarity_matrix)):
                    sim = item_similarity_matrix[recs[i], recs[j]]
                    total_diversity += (1 - sim)
                    count += 1
    
    return total_diversity / count if count > 0 else 0.0


def evaluate_model(model, model_name, test_user_items, train_matrix, k=10):
    """
    Đánh giá một model trên test set
    
    Parameters:
    -----------
    model : object
        Trained recommendation model
    model_name : str
        Tên model ('Popularity', 'ItemCF', 'UserCF', 'SVD', 'ALS')
    test_user_items : dict
        {user_id: [relevant_item_ids]}
    train_matrix : numpy array
        User-item matrix (n_users x n_products)
    k : int
        Top-K recommendations
        
    Returns:
    --------
    dict : Results dictionary với các metrics
    """
    precisions = []
    recalls = []
    f1s = []
    ndcgs = []
    all_recs = []
    
    for user, relevant_items in test_user_items.items():
        # Generate recommendations
        if model_name == 'Popularity':
            recs = model.recommend(n=k)
        elif model_name in ['ItemCF', 'UserCF', 'SVD']:
            user_ratings = train_matrix[user, :]
            if np.sum(user_ratings > 0) == 0:
                continue  # Skip cold-start users
            scores = model.predict(user, user_ratings)
            recs = np.argsort(scores)[::-1][:k]
        elif model_name == 'ALS':
            scores = model.predict_for_user(user)
            recs = np.argsort(scores)[::-1][:k]
        else:
            continue
        
        # Filter out already-rated items
        already_rated = np.where(train_matrix[user, :] > 0)[0]
        recs = [item for item in recs if item not in already_rated][:k]
        
        all_recs.append(recs)
        
        # Compute metrics
        precisions.append(precision_at_k(relevant_items, recs, k))
        recalls.append(recall_at_k(relevant_items, recs, k))
        f1s.append(f1_at_k(relevant_items, recs, k))
        ndcgs.append(ndcg_at_k(relevant_items, recs, k))
    
    return {
        'precisions': precisions,
        'recalls': recalls,
        'f1s': f1s,
        'ndcgs': ndcgs,
        'all_recommendations': all_recs,
        'n_users_evaluated': len(precisions)
    }
