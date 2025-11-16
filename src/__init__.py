"""
Amazon Beauty Recommendation System - Source Package
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Package chứa tất cả modules cho recommendation system
"""

# ============================================================================
# DATA PROCESSING IMPORTS
# ============================================================================

from .data_processing import (
    DataProcessor,
    # Utility functions
    load_csv_numpy,
    validate_data,
    get_basic_stats,
    extract_arrays,
    clean_data,
    filter_by_min_ratings,
    create_id_mappings,
    load_processed_data,
    sample_data,
    train_test_split,
    detect_missing_values,
    handle_missing_values,
    impute_missing_mean,
    impute_missing_median,
    detect_outliers_iqr,
    detect_outliers_zscore,
    remove_outliers,
    normalize_minmax,
    normalize_log,
    standardize_zscore,
    unix_to_datetime_features,
    compute_user_stats,
    compute_product_stats,
    compute_rating_deviation,
    compute_recency_score,
    compute_rating_velocity
)

# ============================================================================
# VISUALIZATION IMPORTS
# ============================================================================

from .visualization import (
    Visualizer,
    # Utility functions
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

# ============================================================================
# MODELS IMPORTS
# ============================================================================

from .models import (
    TruncatedSVD,
    PopularityRecommender,
    ItemBasedCF,
    UserBasedCF,
    SVDRecommender,
    WeightedRecommender,
    # Similarity functions
    create_user_item_matrix,
    cosine_similarity,
    cosine_similarity_matrix,
    pearson_correlation,
    find_top_k_similar,
    # Evaluation metrics
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
    # Utility functions
    standardize,
    get_user_item_pairs,
    get_product_users,
    create_sparse_matrix,
    compute_sparsity,
    top_k_indices,
    print_metrics,
    save_results
)

__version__ = '1.0.0'
__author__ = 'Angela - MSSV: 23122030'

__all__ = [
    # OOP Classes
    'DataProcessor',
    'Visualizer',
    'TruncatedSVD',
    # Data processing functions
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
    'detect_missing_values',
    'handle_missing_values',
    'impute_missing_mean',
    'impute_missing_median',
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'remove_outliers',
    'normalize_minmax',
    'normalize_log',
    'standardize_zscore',
    'unix_to_datetime_features',
    'compute_user_stats',
    'compute_product_stats',
    'compute_rating_deviation',
    'compute_recency_score',
    'compute_rating_velocity',
    # Visualization functions
    'plot_rating_distribution',
    'plot_top_products',
    'plot_user_activity',
    'plot_temporal_trend',
    'plot_feature_distributions',
    'plot_correlation_heatmap',
    'plot_sparsity_analysis',
    'plot_metrics_comparison',
    'plot_recommendation_results',
    # Model classes
    'PopularityRecommender',
    'ItemBasedCF',
    'UserBasedCF',
    'SVDRecommender',
    'WeightedRecommender',
    # Similarity & utility functions
    'create_user_item_matrix',
    'cosine_similarity',
    'cosine_similarity_matrix',
    'pearson_correlation',
    'find_top_k_similar',
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
    'standardize',
    'get_user_item_pairs',
    'get_product_users',
    'create_sparse_matrix',
    'compute_sparsity',
    'top_k_indices',
    'print_metrics',
    'save_results'
]


__version__ = '1.0.0'
__author__ = 'Angela - MSSV: 23122030'

__all__ = [
    # Data processing - Loading
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
    # Data processing - Missing values
    'detect_missing_values',
    'handle_missing_values',
    'impute_missing_mean',
    'impute_missing_median',
    # Data processing - Outliers
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'remove_outliers',
    # Data processing - Normalization
    'normalize_minmax',
    'normalize_log',
    'standardize_zscore',
    # Data processing - Feature engineering
    'unix_to_datetime_features',
    'compute_user_stats',
    'compute_product_stats',
    'compute_rating_deviation',
    'compute_recency_score',
    'compute_rating_velocity',
    # Visualization
        'DataProcessor',
        'Visualizer'
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
    # Similarity
    'create_user_item_matrix',
    'cosine_similarity',
    'cosine_similarity_matrix',
    'pearson_correlation',
    'find_top_k_similar',
    # Evaluation metrics
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
    # Utilities
    'normalize_minmax',
    'standardize',
    'get_user_item_pairs',
    'get_product_users',
    'create_sparse_matrix',
    'compute_sparsity',
    'top_k_indices',
    'print_metrics',
    'save_results'
]
