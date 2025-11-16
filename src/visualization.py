"""
Visualization Module
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Module này chứa các functions để visualize dữ liệu và kết quả
Tất cả functions sử dụng Matplotlib và Seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def plot_rating_distribution(ratings, save_path=None):
    """
    Vẽ biểu đồ phân phối ratings
    
    Parameters:
    -----------
    ratings : numpy array
        Array chứa ratings
    save_path : str, optional
        Path để save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    unique_ratings, counts = np.unique(ratings, return_counts=True)
    axes[0].bar(unique_ratings, counts, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Rating', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Rating Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    axes[1].pie(counts, labels=[f"{r:.0f} stars" for r in unique_ratings], 
                autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    axes[1].set_title('Rating Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_top_products(product_ids, ratings, top_n=20, save_path=None):
    """
    Vẽ biểu đồ top N products theo số lượng ratings
    
    Parameters:
    -----------
    product_ids : numpy array
        Array chứa product IDs
    ratings : numpy array
        Array chứa ratings
    top_n : int
        Số lượng top products
    save_path : str, optional
        Path để save figure
    """
    # Vectorized counting
    unique_products, product_counts = np.unique(product_ids, return_counts=True)
    
    # Sort và lấy top N
    sorted_idx = np.argsort(product_counts)[::-1]
    top_products = unique_products[sorted_idx[:top_n]]
    top_counts = product_counts[sorted_idx[:top_n]]
    
    # Plot horizontal bar
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_counts[::-1], color='coral', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Product {i+1}" for i in range(top_n)][::-1])
    ax.set_xlabel('Number of Ratings', fontsize=12)
    ax.set_title(f'Top {top_n} Products by Rating Count', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_user_activity(user_ids, top_n=20, save_path=None):
    """
    Vẽ biểu đồ user activity distribution
    
    Parameters:
    -----------
    user_ids : numpy array
        Array chứa user IDs
    top_n : int
        Số lượng bins cho histogram
    save_path : str, optional
        Path để save figure
    """
    # Count ratings per user
    unique_users, user_counts = np.unique(user_ids, return_counts=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full distribution (log scale)
    axes[0].hist(user_counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Ratings per User', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('User Activity Distribution (Full)', fontsize=14, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    
    # Zoomed distribution (≤20 ratings)
    axes[1].hist(user_counts[user_counts <= 20], bins=20, color='salmon', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number of Ratings per User', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('User Activity Distribution (Zoomed: ≤20 ratings)', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_temporal_trend(timestamps, save_path=None):
    """
    Vẽ biểu đồ temporal trend của ratings
    
    Parameters:
    -----------
    timestamps : numpy array
        Array chứa timestamps
    save_path : str, optional
        Path để save figure
    """
    # Sort timestamps
    timestamps_sorted = np.sort(timestamps)
    
    # Create bins
    min_ts = timestamps_sorted[0]
    max_ts = timestamps_sorted[-1]
    bins = np.linspace(min_ts, max_ts, 50)
    counts, _ = np.histogram(timestamps, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Convert to dates
    dates = [datetime.fromtimestamp(ts).strftime('%Y-%m') for ts in bin_centers[::5]]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(bin_centers, counts, linewidth=2, color='darkgreen', marker='o', markersize=4)
    ax.fill_between(bin_centers, counts, alpha=0.3, color='lightgreen')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Number of Ratings', fontsize=12)
    ax.set_title('Rating Activity Over Time', fontsize=14, fontweight='bold')
    ax.set_xticks(bin_centers[::5])
    ax.set_xticklabels(dates, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_distributions(features_dict, save_path=None):
    """
    Vẽ biểu đồ phân phối của nhiều features
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary {feature_name: feature_array}
    save_path : str, optional
        Path để save figure
    """
    n_features = len(features_dict)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    colors = sns.color_palette('husl', n_features)
    
    for idx, (name, values) in enumerate(features_dict.items()):
        ax = axes[idx]
        ax.hist(values, bins=50, color=colors[idx], edgecolor='black', alpha=0.7)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution: {name}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_heatmap(features_matrix, feature_names, save_path=None):
    """
    Vẽ heatmap correlation matrix
    
    Parameters:
    -----------
    features_matrix : numpy array
        Matrix chứa features (rows=samples, cols=features)
    feature_names : list
        Tên các features
    save_path : str, optional
        Path để save figure
    """
    # Compute correlation (vectorized)
    corr_matrix = np.corrcoef(features_matrix.T)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=feature_names,
                center=0, vmin=-1, vmax=1, square=True, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sparsity_analysis(n_users, n_products, n_ratings, save_path=None):
    """
    Visualize matrix sparsity
    
    Parameters:
    -----------
    n_users : int
        Số lượng users
    n_products : int
        Số lượng products
    n_ratings : int
        Số lượng ratings
    save_path : str, optional
        Path để save figure
    """
    total_possible = n_users * n_products
    sparsity = 1 - (n_ratings / total_possible)
    density = 1 - sparsity
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create pie chart
    sizes = [density * 100, sparsity * 100]
    labels = [f'Filled\n({density*100:.4f}%)', f'Empty\n({sparsity*100:.4f}%)']
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12})
    
    ax.set_title(f'User-Product Matrix Sparsity\n({n_users:,} users × {n_products:,} products = {total_possible:,} cells)',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Vẽ biểu đồ so sánh metrics giữa các models
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary {model_name: {metric_name: value}}
    save_path : str, optional
        Path để save figure
    """
    models = list(metrics_dict.keys())
    metric_names = list(metrics_dict[models[0]].keys())
    
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    colors = sns.color_palette('Set2', len(models))
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics_dict[model][metric] for model in models]
        
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_recommendation_results(product_ids, ratings, scores, title="Top Recommendations", 
                                top_n=10, save_path=None):
    """
    Visualize recommendation results
    
    Parameters:
    -----------
    product_ids : numpy array
        Array chứa product IDs
    ratings : numpy array
        Array chứa ratings hoặc scores
    scores : numpy array
        Array chứa recommendation scores
    title : str
        Tiêu đề biểu đồ
    top_n : int
        Số lượng recommendations để hiển thị
    save_path : str, optional
        Path để save figure
    """
    # Sort by scores
    sorted_idx = np.argsort(scores)[::-1][:top_n]
    
    top_products = product_ids[sorted_idx]
    top_scores = scores[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_scores[::-1], color='teal', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Product {pid}" for pid in top_products[::-1]])
    ax.set_xlabel('Recommendation Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization module...")
    
    # Generate sample data
    np.random.seed(42)
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=10000, 
                               p=[0.05, 0.05, 0.1, 0.2, 0.6])
    
    print("✓ Generating test plot...")
    plot_rating_distribution(ratings)
    
    print("\nAll tests passed!")
