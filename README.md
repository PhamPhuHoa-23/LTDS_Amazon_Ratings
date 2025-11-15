# Amazon Beauty Products Recommendation System

A comprehensive recommendation system built using **NumPy only** for data processing, implementing collaborative filtering and matrix factorization techniques from scratch.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

---

## Introduction

### Problem Statement

Building an effective recommendation system for e-commerce platforms is crucial for improving user experience and increasing sales. This project addresses the challenge of recommending beauty products to users based on their historical ratings and preferences, while handling the cold start problem for new users and products.

### Motivation and Real-World Applications

Recommendation systems are widely used in:
- **E-commerce platforms**: Amazon, eBay, Alibaba
- **Streaming services**: Netflix, Spotify, YouTube
- **Social media**: Facebook, Instagram, Twitter
- **Content platforms**: Medium, Reddit, Quora

A well-designed recommendation system can:
- Increase user engagement and retention
- Improve conversion rates
- Enhance customer satisfaction
- Drive revenue growth

### Specific Objectives

1. Implement data processing pipeline using **only NumPy** (no Pandas)
2. Build multiple recommendation approaches:
   - Popularity-based (cold start solution)
   - User-based collaborative filtering
   - Item-based collaborative filtering
   - Matrix factorization (SVD and SGD)
3. Implement ML algorithms **from scratch** using NumPy
4. Evaluate and compare different approaches
5. Provide insights and recommendations for business

---

## Dataset

### Source

**Amazon Ratings - Beauty Products**  
[Kaggle Dataset](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)

### Feature Descriptions

| Column | Type | Description |
|--------|------|-------------|
| UserId | String | Unique identifier for each user |
| ProductId | String | Unique identifier for each product |
| Rating | Float | User's rating (1.0 to 5.0) |
| Timestamp | Integer | Unix timestamp of the rating |

### Size and Characteristics

- **Total ratings**: 2,023,070
- **Unique users**: 1,210,271
- **Unique products**: 249,274
- **Time span**: 1998-2014 (15.8 years)
- **Sparsity**: 99.9993% (extremely sparse matrix)
- **Rating distribution**: Highly skewed toward 5 stars (61.7%)

### Data Quality

- **Missing values**: 0 (clean dataset)
- **Rating range**: 1.0 - 5.0
- **Average rating**: 4.15
- **Median rating**: 5.0

---

## Method

### Data Processing Pipeline

1. **Data Loading**: Using `np.genfromtxt()` to load CSV without Pandas
2. **Data Validation**: Check for missing values, duplicates, data types
3. **Outlier Detection**: IQR and Z-score methods (kept as valid ratings)
4. **Normalization**: Min-max, log transformation, decimal scaling
5. **Standardization**: Z-score normalization (mean=0, std=1)
6. **Sparsity Reduction**: Filter users/products with minimum rating threshold (≥5)
7. **Feature Engineering**: User stats, product stats, recency scores, rating velocity

### Algorithms Used

#### 1. Popularity-Based Recommendation

**Formula**: Rank products by total number of ratings

```
Popularity(product) = Count(ratings for product)
```

**Use case**: Cold start problem for new users

#### 2. User-Based Collaborative Filtering

**Formula**: Cosine similarity between users

```
sim(u, v) = (R_u · R_v) / (||R_u|| × ||R_v||)
```

Where:
- `R_u` = rating vector of user u
- `R_v` = rating vector of user v

**Prediction**:
```
r_ui = r̄_u + Σ(sim(u, v) × (r_vi - r̄_v)) / Σ|sim(u, v)|
```

#### 3. Item-Based Collaborative Filtering

**Formula**: Cosine similarity between items

```
sim(i, j) = (R_i · R_j) / (||R_i|| × ||R_j||)
```

**Prediction**:
```
r_ui = Σ(sim(i, j) × r_uj) / Σ|sim(i, j)|
```

#### 4. Matrix Factorization - SVD

**Formula**: Singular Value Decomposition

```
R ≈ U × Σ × V^T
```

Where:
- `R` = user-item matrix (m × n)
- `U` = user features (m × k)
- `Σ` = singular values (k × k)
- `V^T` = item features (k × n)
- `k` = number of latent factors

**Truncated SVD**:
```
R_k = U_k × Σ_k × V_k^T
```

#### 5. Matrix Factorization - SGD

**Formula**: Stochastic Gradient Descent optimization

```
minimize: Σ(r_ui - p_u^T × q_i)^2 + λ(||p_u||^2 + ||q_i||^2)
```

**Update rules**:
```
e_ui = r_ui - p_u^T × q_i
p_u ← p_u + α(e_ui × q_i - β × p_u)
q_i ← q_i + α(e_ui × p_u - β × q_i)
```

Where:
- `α` = learning rate
- `β` = regularization parameter
- `λ` = regularization coefficient

### NumPy Implementation Details

**Key Techniques Used**:

1. **Vectorization**: All operations use NumPy arrays, no for loops on arrays
2. **Broadcasting**: Efficient element-wise operations
3. **Fancy Indexing**: Boolean masks for filtering
4. **Matrix Operations**: `np.dot()`, `np.linalg.svd()`, `np.corrcoef()`
5. **Memory Efficiency**: Views vs copies, in-place operations where possible
6. **Numerical Stability**: Handling division by zero, clipping values

**Example - Cosine Similarity**:
```python
def cosine_similarity_matrix(matrix, axis=0):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = matrix / norms
    similarity_matrix = np.dot(normalized, normalized.T)
    return similarity_matrix
```

---

## Installation & Setup

### Requirements

- Python 3.8+
- NumPy 1.24.3
- Matplotlib 3.7.1
- Seaborn 0.12.2

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/PhamPhuHoa-23/-LTDS-_Amazon---Ratings.git
cd -LTDS-_Amazon---Ratings
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download dataset:
   - Place `ratings_Beauty.csv` in `data/raw/` directory
   - Or download from [Kaggle](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)

---

## Usage

### Step 1: Data Exploration

Run the exploration notebook to understand the dataset:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook covers:
- Data overview and quality check
- Rating distribution analysis
- User behavior patterns
- Product popularity trends
- Temporal analysis
- Sparsity calculation

### Step 2: Data Preprocessing

Run the preprocessing notebook:

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

This applies:
- Missing value handling
- Outlier detection
- Normalization/standardization
- Feature engineering
- Data filtering

### Step 3: Build Recommendation System

Run the modeling notebook:

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

This implements:
- Popularity-based recommendations
- User-based collaborative filtering
- Item-based collaborative filtering
- SVD matrix factorization
- SGD matrix factorization
- Model evaluation and comparison

### Using Functions Directly

```python
from src.data_loader import load_csv_numpy
from src.similarity import create_user_item_matrix, cosine_similarity_matrix
from src.models import svd_numpy, matrix_factorization_sgd

# Load data
data, header = load_csv_numpy('data/raw/ratings_Beauty.csv')

# Create user-item matrix
matrix, user_map, product_map = create_user_item_matrix(
    data['UserId'], data['ProductId'], data['Rating']
)

# Apply SVD
U, Sigma, Vt = svd_numpy(matrix.T, k_factors=10)

# Or use SGD
P, Q = matrix_factorization_sgd(matrix, K=10, steps=1000)
```

---

## Results

### Metrics Achieved

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|-------|------|-----|--------------|-----------|
| Popularity-Based | - | - | Baseline | Baseline |
| User-Based CF | ~1.2 | ~0.9 | ~0.15 | ~0.12 |
| Item-Based CF | ~1.1 | ~0.8 | ~0.18 | ~0.15 |
| SVD (k=10) | ~1.0 | ~0.7 | ~0.20 | ~0.18 |
| MF-SGD (k=10) | ~0.9 | ~0.6 | ~0.22 | ~0.20 |

*Note: Metrics vary based on data subset and hyperparameters*

### Visualizations

The notebooks include visualizations for:
- Rating distribution (bar charts, pie charts)
- User activity patterns (histograms)
- Product popularity (bar charts)
- Temporal trends (line plots)
- Similarity matrices (heatmaps)

### Comparison and Analysis

**Key Findings**:

1. **Sparsity Challenge**: 99.99% sparse matrix requires dimensionality reduction
2. **Popularity Baseline**: Effective for cold start but lacks personalization
3. **Collaborative Filtering**: Provides personalization but computationally expensive
4. **Matrix Factorization**: Best balance of accuracy and efficiency
5. **SGD vs SVD**: SGD more flexible but requires hyperparameter tuning

**Recommendations**:

- Use **popularity-based** for new users (cold start)
- Use **item-based CF** for scalability
- Use **SVD** for large-scale production
- Use **hybrid approach** combining multiple methods

---

## Project Structure

```
Lab2DS/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/
│   ├── raw/
│   │   └── ratings_Beauty.csv   # Original dataset
│   └── processed/               # Processed data files
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and analysis
│   ├── 02_preprocessing.ipynb        # Data preprocessing
│   └── 03_modeling.ipynb             # Recommendation system
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading, validation, stats
│   ├── data_processing.py      # Missing values, outliers, normalization
│   ├── feature_engineering.py  # Feature creation
│   ├── similarity.py           # Similarity computations
│   └── models.py               # ML algorithms from scratch
└── kaggle_insights/            # Reference insights from Kaggle
```

### File Descriptions

- **data_loader.py**: Load CSV using NumPy, validate data, compute statistics
- **data_processing.py**: Handle missing values, detect outliers, normalize/standardize
- **feature_engineering.py**: Compute user/product stats, recency scores, velocity
- **similarity.py**: Cosine similarity, Pearson correlation, user-item matrix
- **models.py**: SVD, matrix factorization SGD, evaluation metrics, cross-validation

---

## Challenges & Solutions

### Challenge 1: Loading Large CSV with NumPy

**Problem**: `np.genfromtxt()` can be slow for large files (2M+ rows)

**Solution**: 
- Use structured arrays with appropriate dtypes
- Process in chunks if needed
- Optimize memory usage

### Challenge 2: Handling Sparse Matrices

**Problem**: User-item matrix is 99.99% sparse, memory-intensive

**Solution**:
- Filter by minimum rating threshold
- Use sample subset for development
- Implement sparse matrix operations where possible

### Challenge 3: Numerical Stability

**Problem**: Division by zero in similarity calculations

**Solution**:
```python
norms[norms == 0] = 1  # Avoid division by zero
if std == 0:
    return np.zeros_like(data)  # Handle constant arrays
```

### Challenge 4: Vectorization Without Pandas

**Problem**: Need groupby-like operations using only NumPy

**Solution**:
- Use `np.unique()` with `return_counts=True`
- Boolean indexing for filtering
- Broadcasting for element-wise operations

### Challenge 5: Implementing SGD from Scratch

**Problem**: Nested loops are slow in Python

**Solution**:
- Use vectorized operations where possible
- Optimize with NumPy's efficient array operations
- Consider batch processing for large datasets

---

## Future Improvements

1. **Hybrid Approaches**: Combine collaborative filtering with content-based methods
2. **Deep Learning**: Implement neural collaborative filtering
3. **Real-time Recommendations**: Optimize for online learning
4. **A/B Testing**: Framework for testing recommendation strategies
5. **Diversity Metrics**: Measure recommendation diversity and serendipity
6. **Cold Start Solutions**: Better handling of new users/products
7. **Scalability**: Distributed computing for large-scale deployment
8. **Hyperparameter Tuning**: Automated grid search or Bayesian optimization
9. **Feature Engineering**: Incorporate product metadata if available
10. **Evaluation Framework**: Comprehensive metrics dashboard

---

## Contributors

**Author**: Pham Phu Hoa  
**Student ID**: [Your Student ID]  
**Email**: [Your Email]  
**Course**: CSC17104 - Programming for Data Science  
**Institution**: [Your University]

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)
- Inspiration from various Kaggle notebooks on recommendation systems
- NumPy documentation and community

---

**Note**: This project was completed as part of a Data Science course assignment, focusing on NumPy mastery and implementing ML algorithms from scratch.
