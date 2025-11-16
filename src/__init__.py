"""
Amazon Beauty Recommendation System - Source Package
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Package chứa tất cả modules cho recommendation system
"""

from .data_loader import (
    load_csv_numpy,
    validate_data,
    get_basic_stats,
    extract_arrays,
    clean_data,
    filter_by_min_ratings,
    create_id_mappings,
    load_processed_data,
    sample_data,
    train_test_split
)

from .visualization import (
    plot_rating_distribution,
    plot_top_products,
    plot_user_activity,
    plot_temporal_trend,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_sparsity_analysis,
    plot_metrics_comparison,
    plot_recommendation_results
)

from .models import (
    PopularityRecommender,
    ItemBasedCF,
    UserBasedCF,
    SVDRecommender,
    WeightedRecommender
)

from .utils import (
    rmse,
    mae,
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    coverage,
    diversity,
    normalize_minmax,
    standardize,
    cosine_similarity,
    cosine_similarity_matrix,
    pearson_correlation,
    get_user_item_pairs,
    get_product_users,
    create_sparse_matrix,
    compute_sparsity,
    print_metrics,
    save_results
)

__version__ = '1.0.0'
__author__ = 'Angela - MSSV: 23122030'

__all__ = [
    # Data loader
    'load_csv_numpy',
    'validate_data',
    'get_basic_stats',
    'extract_arrays',
    'clean_data',
    'filter_by_min_ratings',
    'create_id_mappings',
    'load_processed_data',
    'sample_data',
    'train_test_split',
    
    # Visualization
    'plot_rating_distribution',
    'plot_top_products',
    'plot_user_activity',
    'plot_temporal_trend',
    'plot_feature_distributions',
    'plot_correlation_heatmap',
    'plot_sparsity_analysis',
    'plot_metrics_comparison',
    'plot_recommendation_results',
    
    # Models
    'PopularityRecommender',
    'ItemBasedCF',
    'UserBasedCF',
    'SVDRecommender',
    'WeightedRecommender',
    
    # Utils
    'rmse',
    'mae',
    'precision_at_k',
    'recall_at_k',
    'f1_at_k',
    'ndcg_at_k',
    'hit_rate_at_k',
    'mean_reciprocal_rank',
    'coverage',
    'diversity',
    'normalize_minmax',
    'standardize',
    'cosine_similarity',
    'cosine_similarity_matrix',
    'pearson_correlation',
    'get_user_item_pairs',
    'get_product_users',
    'create_sparse_matrix',
    'compute_sparsity',
    'print_metrics',
    'save_results'
]
