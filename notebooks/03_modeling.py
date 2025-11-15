# Recommendation System Modeling - Amazon Beauty Products
# Extracted from 03_modeling.ipynb for testing

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_csv_numpy
from similarity import (
    create_user_item_matrix, cosine_similarity_matrix,
    pearson_correlation, find_top_k_similar
)
from models import (
    svd_numpy, matrix_factorization_sgd, predict_rating,
    rmse, mae, precision_at_k, recall_at_k, hit_rate,
    train_test_split_numpy
)

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_path = os.path.join(base_dir, 'data', 'processed', 'filtered_data.npy')
raw_path = os.path.join(base_dir, 'data', 'raw', 'ratings_Beauty.csv')

try:
    data = np.load(processed_path, allow_pickle=True)
    print(f"Loaded processed data: {data.shape}")
except:
    print("Processed data not found. Loading raw data and filtering...")
    data, _ = load_csv_numpy(raw_path)
    from data_processing import filter_by_min_ratings
    data = filter_by_min_ratings(data, min_user_ratings=5, min_product_ratings=5)
    print(f"Filtered data: {data.shape}")

print(f"Unique users: {len(np.unique(data['UserId'])):,}")
print(f"Unique products: {len(np.unique(data['ProductId'])):,}")
print(f"Total ratings: {len(data):,}")

# Sample data to avoid memory issues
print("\nSampling data to avoid memory issues...")
sample_size = 5000
if len(data) > sample_size:
    sample_indices = np.random.choice(len(data), sample_size, replace=False)
    data_sample = data[sample_indices]
    print(f"Using sample of {sample_size} ratings")
else:
    data_sample = data

# Create user-item matrix
user_ids = data_sample['UserId']
product_ids = data_sample['ProductId']
ratings = data_sample['Rating']

matrix, user_map, product_map = create_user_item_matrix(user_ids, product_ids, ratings)

print(f"User-item matrix shape: {matrix.shape}")
print(f"Sparsity: {(1 - np.count_nonzero(matrix) / matrix.size) * 100:.2f}%")
print(f"Density: {(np.count_nonzero(matrix) / matrix.size) * 100:.4f}%")

inverse_user_map = {v: k for k, v in user_map.items()}
inverse_product_map = {v: k for k, v in product_map.items()}

# Popularity-based
product_ratings_count = np.sum(matrix > 0, axis=0)
top_n = 20
top_products_idx = np.argsort(product_ratings_count)[-top_n:][::-1]

print("\n=== Top 20 Most Popular Products ===\n")
for i, idx in enumerate(top_products_idx[:10], 1):
    product_id = inverse_product_map[idx]
    count = product_ratings_count[idx]
    print(f"{i:2d}. Product {product_id}: {count:,} ratings")

# User-based CF - FIX: Use sample for large matrices
print("\n=== User-Based Collaborative Filtering ===\n")
print("Using sample matrix to avoid memory issues...")

sample_size = min(1000, matrix.shape[0])
sample_matrix = matrix[:sample_size, :]
print(f"Sample matrix shape: {sample_matrix.shape}")

user_similarity = cosine_similarity_matrix(sample_matrix, axis=0)
print(f"User similarity matrix shape: {user_similarity.shape}")

test_user_idx = 0
top_k_users_idx, top_k_scores = find_top_k_similar(user_similarity, test_user_idx, k=5)

print(f"\nTop 5 similar users to user {test_user_idx}:")
for i, (user_idx, score) in enumerate(zip(top_k_users_idx, top_k_scores), 1):
    print(f"  {i}. User {user_idx}: similarity = {score:.4f}")

# Item-based CF - FIX: Use sample
print("\n=== Item-Based Collaborative Filtering ===\n")
print("Using sample matrix...")

item_sample_size = min(1000, matrix.shape[1])
item_sample_matrix = matrix[:, :item_sample_size]
print(f"Item sample matrix shape: {item_sample_matrix.shape}")

item_similarity = cosine_similarity_matrix(item_sample_matrix, axis=1)
print(f"Item similarity matrix shape: {item_similarity.shape}")

test_product_idx = 0
top_k_products_idx, top_k_scores = find_top_k_similar(item_similarity, test_product_idx, k=5)

print(f"\nTop 5 similar products to product {test_product_idx}:")
for i, (prod_idx, score) in enumerate(zip(top_k_products_idx, top_k_scores), 1):
    if prod_idx < len(inverse_product_map):
        product_id = inverse_product_map[prod_idx]
        print(f"  {i}. Product {product_id}: similarity = {score:.4f}")

# SVD
print("\n=== SVD-based Matrix Factorization ===\n")
k_factors = 10
print(f"Decomposing matrix with {k_factors} latent factors...")

# Use sample for SVD too
svd_sample_size = min(500, matrix.shape[0], matrix.shape[1])
svd_sample_matrix = matrix[:svd_sample_size, :svd_sample_size]
print(f"Using sample matrix: {svd_sample_matrix.shape}")

U_k, Sigma_k, Vt_k = svd_numpy(svd_sample_matrix.T, k_factors)
print(f"U shape: {U_k.shape}, Sigma shape: {Sigma_k.shape}, Vt shape: {Vt_k.shape}")

reconstructed = U_k @ np.diag(Sigma_k) @ Vt_k
print(f"Reconstructed matrix shape: {reconstructed.shape}")

correlation_matrix = np.corrcoef(reconstructed)
print(f"Correlation matrix shape: {correlation_matrix.shape}")

# SGD
print("\n=== Matrix Factorization with SGD ===\n")
sample_size = min(500, matrix.shape[0], matrix.shape[1])
sample_matrix = matrix[:sample_size, :sample_size]
print(f"Using sample matrix: {sample_matrix.shape} for demonstration")

K = 10
steps = 100
print(f"Training with {steps} iterations, {K} latent factors...")

P, Q = matrix_factorization_sgd(sample_matrix, K, steps=steps, alpha=0.002, beta=0.02)
print(f"P shape: {P.shape}, Q shape: {Q.shape}")

print("\nPredicting ratings for sample user-item pairs...")
test_predictions = []
for i in range(min(10, sample_matrix.shape[0])):
    for j in range(min(10, sample_matrix.shape[1])):
        if sample_matrix[i, j] > 0:
            pred = predict_rating(P, Q, i, j)
            test_predictions.append((sample_matrix[i, j], pred))

if test_predictions:
    true_vals = np.array([p[0] for p in test_predictions])
    pred_vals = np.array([p[1] for p in test_predictions])
    print(f"Sample RMSE: {rmse(true_vals, pred_vals):.4f}")
    print(f"Sample MAE: {mae(true_vals, pred_vals):.4f}")

print("\n=== All tests completed successfully! ===")

