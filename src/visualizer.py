"""
Visualizer Class - OOP Implementation
CSC17104 - Programming for Data Science
Student: Angela - MSSV: 23122030

Visualize dữ liệu và kết quả recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """
    Tập hợp các methods để visualize dữ liệu và kết quả
    """
    
    def __init__(self, figsize=(14, 6), dpi=100, style='whitegrid'):
        """
        Parameters:
        -----------
        figsize : tuple
            Kích thước figure mặc định
        dpi : int
            DPI resolution
        style : str
            Seaborn style
        """
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['font.size'] = 10
    
    @staticmethod
    def plot_rating_distribution(ratings, title="Rating Distribution"):
        """
        Vẽ phân bố ratings
        
        Parameters:
        -----------
        ratings : numpy array
            Array của ratings
        title : str
            Tiêu đề
        """
        unique_ratings, counts = np.unique(ratings, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(unique_ratings, counts, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_user_activity(user_n_ratings, title="User Activity Distribution"):
        """
        Vẽ phân bố hoạt động của users
        
        Parameters:
        -----------
        user_n_ratings : numpy array
            Số lượng ratings của mỗi user
        title : str
            Tiêu đề
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(user_n_ratings, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Ratings per User')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_product_popularity(product_n_ratings, title="Product Popularity Distribution"):
        """
        Vẽ phân bố popularity của products
        
        Parameters:
        -----------
        product_n_ratings : numpy array
            Số lượng ratings của mỗi product
        title : str
            Tiêu đề
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(product_n_ratings, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Ratings per Product')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_temporal_trend(years, title="Rating Activity Over Time"):
        """
        Vẽ xu hướng temporal
        
        Parameters:
        -----------
        years : numpy array
            Array của years
        title : str
            Tiêu đề
        """
        year_counts = np.bincount(years - np.min(years))
        year_labels = np.arange(np.min(years), np.max(years) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(year_labels[:len(year_counts)], year_counts, marker='o', linewidth=2, color='darkviolet')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Ratings')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_rating_deviation(user_rating_deviation, title="User Rating Deviation"):
        """
        Vẽ deviation của ratings
        
        Parameters:
        -----------
        user_rating_deviation : numpy array
            Deviation của user ratings
        title : str
            Tiêu đề
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(user_rating_deviation, bins=50, color='orange', edgecolor='black', alpha=0.7)
        ax.set_xlabel('User Rating Deviation')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_recency_weight(recency_weight, title="Recency Weight Distribution"):
        """
        Vẽ phân bố recency weights
        
        Parameters:
        -----------
        recency_weight : numpy array
            Recency weights
        title : str
            Tiêu đề
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(recency_weight, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Recency Weight')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_model_comparison(models_data, metrics, title="Model Comparison"):
        """
        Vẽ biểu đồ so sánh các models
        
        Parameters:
        -----------
        models_data : dict
            Dictionary {model_name: {metric: value}}
        metrics : list
            Danh sách metrics để vẽ
        title : str
            Tiêu đề
        """
        model_names = list(models_data.keys())
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, metric in enumerate(metrics):
            values = [models_data[model].get(metric, 0) for model in model_names]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        return fig, ax
    
    @staticmethod
    def plot_feature_distributions(features_dict, title="Feature Distributions"):
        """
        Vẽ phân bố của nhiều features
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary {feature_name: array}
        title : str
            Tiêu đề
        """
        n_features = len(features_dict)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for idx, (name, values) in enumerate(features_dict.items()):
            ax = axes[idx]
            ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Distribution')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        return fig, axes
    
    @staticmethod
    def plot_similarity_matrix(similarity_matrix, title="Similarity Matrix", cmap='coolwarm'):
        """
        Vẽ heatmap của similarity matrix
        
        Parameters:
        -----------
        similarity_matrix : numpy array
            2D similarity matrix
        title : str
            Tiêu đề
        cmap : str
            Colormap
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix, cmap=cmap, aspect='auto')
        ax.set_xlabel('Item')
        ax.set_ylabel('Item')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return fig, ax
    
    @staticmethod
    def plot_top_products(product_ids, product_n_ratings, top_n=20, title="Top Products by Rating Count"):
        """
        Vẽ top N products
        
        Parameters:
        -----------
        product_ids : numpy array
            Product IDs
        product_n_ratings : numpy array
            Số lượng ratings
        top_n : int
            Số lượng top products
        title : str
            Tiêu đề
        """
        top_indices = np.argsort(product_n_ratings)[-top_n:][::-1]
        top_products = product_ids[top_indices]
        top_counts = product_n_ratings[top_indices]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(top_products)), top_counts, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels(top_products, fontsize=9)
        ax.set_xlabel('Number of Ratings')
        ax.set_title(title)
        ax.grid(alpha=0.3, axis='x')
        
        return fig, ax
    
    @staticmethod
    def plot_recommendations(recommendations, title="Sample Recommendations"):
        """
        Hiển thị danh sách recommendations
        
        Parameters:
        -----------
        recommendations : list or dict
            Danh sách hoặc dict của recommendations
        title : str
            Tiêu đề
        """
        # Có thể in ra console hoặc vẽ visualize
        print(f"\n{title}:")
        if isinstance(recommendations, dict):
            for key, recs in recommendations.items():
                print(f"  {key}: {recs}")
        else:
            print(f"  {recommendations}")


# Demo usage (có thể run standalone)
if __name__ == "__main__":
    np.random.seed(42)
    
    # Sample data
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], 1000)
    user_n_ratings = np.random.randint(5, 100, 200)
    product_n_ratings = np.random.randint(5, 100, 50)
    
    viz = Visualizer()
    
    # Plot rating distribution
    fig, ax = viz.plot_rating_distribution(ratings)
    plt.show()
    
    print("Visualization demo completed")
