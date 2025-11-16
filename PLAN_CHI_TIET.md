# KỀ HOẠCH CHI TIẾT - RECOMMENDATION SYSTEM PROJECT

## 🎯 MỤC TIÊU TỔNG QUAN
- **Dataset**: Amazon Beauty Ratings (2M+ ratings, 1.2M users, 249K products)
- **Yêu cầu**: CHỈ dùng NumPy cho data processing, Matplotlib/Seaborn cho visualization
- **Target**: 110% điểm (100% base + 10% bonus từ ML from scratch)
- **Approach**: Commit từng function nhỏ, hạn chế comment, có docstring ngắn gọn

---

## 📊 PHÂN BỔ ĐIỂM VÀ CHIẾN LƯỢC

### 1. PRESENTATION & ORGANIZATION (20%)
- **Notebooks (10%)**: 3 notebooks rõ ràng, markdown cells giải thích, visualizations đẹp
- **GitHub Repository (10%)**: Cấu trúc chuẩn, README.md đầy đủ 13 sections

### 2. NUMPY TECHNIQUES (50%)
- **Vectorization (10%)**: Không dùng for loops, broadcasting, ufuncs
- **Advanced NumPy (15%)**: Fancy indexing, reshaping, np.einsum, memory efficiency
- **Mathematical Operations (10%)**: Numerical stability, statistical calculations, hypothesis testing
- **Code Efficiency (15%)**: Clean code, resource-efficient, reproducible

### 3. RESULTS (30%)
- **Model Performance (15%)**: Metrics đầy đủ (RMSE, Precision@K, Recall@K, etc.)
- **Insights (15%)**: Câu hỏi thú vị + trả lời bằng data + recommendations

### 4. BONUS (10%)
- **ML from scratch**: SVD/Matrix Factorization, similarity metrics, evaluation metrics
- **Interesting analysis**: Temporal patterns, user segmentation, hypothesis testing

---

## 🗂️ CẤU TRÚC PROJECT

```
Lab2DS/
├── README.md                          # 13 sections đầy đủ
├── requirements.txt                   # numpy, matplotlib, seaborn
├── .gitignore                         # Python standard
├── data/
│   ├── raw/
│   │   └── ratings_Beauty.csv        # Đã có
│   └── processed/
│       ├── filtered_data.npy
│       ├── user_item_matrix.npy
│       └── metadata.npy
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA + visualizations
│   ├── 02_preprocessing.ipynb        # Data cleaning + feature engineering
│   └── 03_modeling.ipynb             # Recommendation system + evaluation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Load CSV bằng NumPy
│   ├── data_processing.py            # Missing values, outliers, normalization
│   ├── feature_engineering.py        # Tạo features mới
│   ├── similarity.py                 # Cosine similarity, distance metrics
│   ├── models.py                     # ML from scratch (SVD, MF)
│   └── visualization.py              # Reusable plotting functions
└── kaggle_insights/                   # Đã có (để reference)
```

---

## 🚀 PHASE-BY-PHASE EXECUTION PLAN

### **PHASE 1: PROJECT SETUP** ⏱️ 30-60 phút

#### Bước 1.1: Tạo cấu trúc folders
```bash
mkdir -p notebooks src data/processed
```

**Commit**: "Initial project structure"

#### Bước 1.2: Tạo .gitignore
Nội dung:
```
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.DS_Store
*.npy
data/processed/*
!data/processed/.gitkeep
```

**Commit**: "Add .gitignore"

#### Bước 1.3: Tạo requirements.txt
```
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
```

**Commit**: "Add requirements.txt"

#### Bước 1.4: Tạo src/__init__.py (empty file)
**Commit**: "Add src module structure"

---

### **PHASE 2: DATA LOADER** ⏱️ 1-2 giờ

#### File: `src/data_loader.py`

**Function 1: load_csv_numpy()**
```python
def load_csv_numpy(filepath, delimiter=',', skip_header=1, dtype=None):
    """
    Load CSV file using NumPy only.
    
    Args:
        filepath: Path to CSV file
        delimiter: Column separator
        skip_header: Number of header rows to skip
        dtype: Data type specification
    
    Returns:
        Structured NumPy array with columns
    """
    # Implementation using np.genfromtxt or manual parsing
```

**Commit**: "Add load_csv_numpy function"

**Function 2: validate_data()**
```python
def validate_data(data):
    """
    Validate data quality: check nulls, duplicates, value ranges.
    
    Returns:
        dict: Validation report
    """
```

**Commit**: "Add data validation function"

**Function 3: get_basic_stats()**
```python
def get_basic_stats(data):
    """
    Calculate basic statistics: shape, unique counts, memory usage.
    
    Returns:
        dict: Statistics summary
    """
```

**Commit**: "Add basic statistics function"

---

### **PHASE 3: DATA EXPLORATION** ⏱️ 3-4 giờ

#### Notebook: `notebooks/01_data_exploration.ipynb`

**Section 1: Load Data**
- Import libraries (numpy, matplotlib, seaborn)
- Load data using `data_loader.load_csv_numpy()`
- Display basic info
- **Commit sau section này**: "Add data loading section"

**Section 2: Data Overview**
- Shape, dtypes, memory usage
- Sample rows (head/tail)
- Unique counts (users, products)
- **Markdown cell**: Nhận xét về kích thước dataset

**Commit**: "Add data overview analysis"

**Section 3: Missing Values Analysis**
- Check nulls per column
- Visualize missing patterns
- Calculate missing percentage
- **Markdown cell**: Chiến lược xử lý missing values

**Commit**: "Add missing values analysis"

**Section 4: Rating Distribution**
- Histogram of ratings
- Value counts
- Statistical measures (mean, median, mode, std)
- **Visualization**: Seaborn countplot
- **Markdown cell**: Nhận xét về rating bias (skewed toward 5 stars)

**Commit**: "Add rating distribution analysis"

**Section 5: User Behavior Patterns**
Analyses:
1. Distribution of ratings per user
2. Top 20 most active users
3. User segmentation:
   - Power users (>100 ratings)
   - Regular users (10-100 ratings)
   - Casual users (<10 ratings)
4. Average rating per user (harsh vs lenient raters)

**Visualizations**:
- Histogram: Ratings per user
- Box plot: User activity levels
- Scatter plot: User activity vs avg rating

**Markdown cell**: Insights về user behavior

**Commit**: "Add user behavior analysis"

**Section 6: Product Analysis**
Analyses:
1. Distribution of ratings per product
2. Top 20 most rated products
3. Product segmentation:
   - Popular products (>100 ratings)
   - Moderate products (10-100 ratings)
   - Niche products (<10 ratings)
4. Correlation: Product popularity vs avg rating

**Visualizations**:
- Histogram: Ratings per product
- Scatter plot: Popularity vs avg rating
- Bar chart: Top 20 products

**Markdown cell**: Insights về product patterns

**Commit**: "Add product analysis"

**Section 7: Temporal Analysis**
Analyses:
1. Convert Unix timestamps to datetime (năm, tháng, ngày)
2. Ratings over time (time series)
3. Ratings per month/year
4. Seasonal patterns (if applicable)
5. Product lifecycle analysis

**Visualizations**:
- Line plot: Ratings over time
- Bar plot: Ratings per month
- Heatmap: Month vs Year

**Markdown cell**: Insights về temporal patterns

**Commit**: "Add temporal analysis"

**Section 8: Sparsity Analysis**
- Calculate sparsity percentage
- Visualize sparse matrix structure (sample)
- Implications for recommendation

**Commit**: "Add sparsity analysis"

**Section 9: Statistical Hypothesis Testing**

**Test 1: Rating Distribution**
- H0: Ratings are uniformly distributed across 1-5 stars
- H1: Ratings are not uniformly distributed
- Method: Chi-square goodness-of-fit test
- Implement from scratch using NumPy

**Test 2: Popularity vs Quality**
- H0: Product popularity (# ratings) doesn't correlate with avg rating
- H1: Correlation exists
- Method: Pearson correlation + significance test

**Markdown cells**: 
- State hypotheses clearly
- Show calculations
- Interpret p-values
- Draw conclusions

**Commit**: "Add hypothesis testing"

**Section 10: Key Findings Summary**
- Bullet points of main insights
- Implications for recommendation system

**Commit**: "Add exploration summary"

---

### **PHASE 4: DATA PREPROCESSING FUNCTIONS** ⏱️ 3-4 giờ

#### File: `src/data_processing.py`

**Function 1: detect_missing_values()**
```python
def detect_missing_values(data):
    """
    Detect missing, null, NaN values in data.
    
    Returns:
        Boolean array indicating missing positions
    """
```

**Commit**: "Add detect_missing_values function"

**Function 2: impute_missing_mean()**
```python
def impute_missing_mean(data, columns):
    """
    Fill missing values with column mean.
    """
```

**Commit**: "Add mean imputation function"

**Function 3: impute_missing_median()**
**Commit**: "Add median imputation function"

**Function 4: detect_outliers_iqr()**
```python
def detect_outliers_iqr(data, column_idx, multiplier=1.5):
    """
    Detect outliers using IQR method.
    
    Returns:
        Boolean array: True for outliers
    """
```

**Commit**: "Add IQR outlier detection"

**Function 5: detect_outliers_zscore()**
**Commit**: "Add Z-score outlier detection"

**Function 6: remove_outliers()**
**Commit**: "Add remove_outliers function"

**Function 7: normalize_minmax()**
```python
def normalize_minmax(data, feature_min=0, feature_max=1):
    """
    Min-Max normalization: scale to [feature_min, feature_max].
    
    Formula: X_norm = (X - X_min) / (X_max - X_min) * (max - min) + min
    """
```

**Commit**: "Add min-max normalization"

**Function 8: normalize_log()**
```python
def normalize_log(data, base='e'):
    """
    Log transformation for skewed distributions.
    """
```

**Commit**: "Add log normalization"

**Function 9: standardize_zscore()**
```python
def standardize_zscore(data):
    """
    Z-score standardization: mean=0, std=1.
    
    Formula: Z = (X - μ) / σ
    """
```

**Commit**: "Add z-score standardization"

**Function 10: unix_to_datetime_features()**
```python
def unix_to_datetime_features(timestamps):
    """
    Convert Unix timestamps to year, month, day, hour, weekday.
    
    Returns:
        Dict of datetime features as NumPy arrays
    """
```

**Commit**: "Add timestamp conversion function"

**Function 11: filter_by_min_ratings()**
```python
def filter_by_min_ratings(data, min_user_ratings=5, min_product_ratings=5):
    """
    Filter users and products with minimum rating counts.
    Reduces sparsity.
    """
```

**Commit**: "Add filtering by min ratings"

---

### **PHASE 5: FEATURE ENGINEERING** ⏱️ 2-3 giờ

#### File: `src/feature_engineering.py`

**Function 1: compute_user_stats()**
```python
def compute_user_stats(data):
    """
    Compute per-user statistics:
    - Total ratings count
    - Average rating (harsh vs lenient)
    - Rating std deviation
    - Rating range
    
    Returns:
        Dictionary of user statistics arrays
    """
```

**Commit**: "Add user statistics computation"

**Function 2: compute_product_stats()**
**Commit**: "Add product statistics computation"

**Function 3: compute_rating_deviation()**
```python
def compute_rating_deviation(user_rating, user_avg, product_avg):
    """
    Deviation of rating from user and product averages.
    Personalization signal.
    """
```

**Commit**: "Add rating deviation feature"

**Function 4: compute_recency_score()**
```python
def compute_recency_score(timestamps, decay_factor=0.1):
    """
    Time-based weighting: recent ratings more important.
    """
```

**Commit**: "Add recency score feature"

**Function 5: compute_rating_velocity()**
```python
def compute_rating_velocity(timestamps, window_days=30):
    """
    Ratings per time period (trending indicator).
    """
```

**Commit**: "Add rating velocity feature"

**Function 6: create_feature_matrix()**
```python
def create_feature_matrix(data, feature_list):
    """
    Combine multiple features into matrix for ML.
    """
```

**Commit**: "Add feature matrix creation"

---

### **PHASE 6: PREPROCESSING NOTEBOOK** ⏱️ 2-3 giờ

#### Notebook: `notebooks/02_preprocessing.ipynb`

**Section 1: Load Raw Data**
**Commit**: "Add preprocessing notebook - data loading"

**Section 2: Handle Missing Values**
- Apply detection functions
- Choose strategy (mean/median) with justification
- Show before/after statistics

**Commit**: "Add missing value handling section"

**Section 3: Outlier Detection & Handling**
- Apply IQR and Z-score methods
- Visualize outliers
- Decide whether to remove (with justification)

**Commit**: "Add outlier handling section"

**Section 4: Normalization & Standardization**
- Apply to appropriate features
- Show distributions before/after
- Explain why each method chosen

**Commit**: "Add normalization section"

**Section 5: Timestamp Feature Engineering**
- Extract datetime features
- Visualize temporal features

**Commit**: "Add timestamp features section"

**Section 6: Filter Sparse Data**
- Apply minimum rating thresholds
- Show sparsity improvement
- Final data statistics

**Commit**: "Add data filtering section"

**Section 7: Compute User & Product Features**
- Apply feature engineering functions
- Visualize feature distributions
- Save engineered features

**Commit**: "Add feature engineering section"

**Section 8: Save Processed Data**
- Save as .npy files to data/processed/
- Save metadata (user/product mappings)

**Commit**: "Add data saving section"

---

### **PHASE 7: SIMILARITY COMPUTATION** ⏱️ 2-3 giờ

#### File: `src/similarity.py`

**Function 1: create_user_item_matrix()**
```python
def create_user_item_matrix(user_ids, product_ids, ratings):
    """
    Create sparse user-item matrix.
    Rows: users, Columns: products, Values: ratings
    
    Returns:
        matrix, user_map, product_map
    """
```

**Commit**: "Add user-item matrix creation"

**Function 2: cosine_similarity()**
```python
def cosine_similarity(vector_a, vector_b):
    """
    Compute cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| * ||B||)
    
    Pure NumPy implementation using broadcasting.
    """
```

**Commit**: "Add cosine similarity function"

**Function 3: cosine_similarity_matrix()**
```python
def cosine_similarity_matrix(matrix, axis=0):
    """
    Compute pairwise cosine similarities.
    
    Args:
        axis: 0 for user-user, 1 for item-item
    
    Returns:
        Similarity matrix
    """
```

**Commit**: "Add similarity matrix computation"

**Function 4: pearson_correlation()**
```python
def pearson_correlation(matrix, axis=0):
    """
    Pearson correlation for collaborative filtering.
    Handles mean-centering automatically.
    """
```

**Commit**: "Add Pearson correlation function"

**Function 5: find_top_k_similar()**
```python
def find_top_k_similar(similarity_matrix, idx, k=10):
    """
    Find top-k most similar items/users.
    
    Returns:
        Indices and similarity scores
    """
```

**Commit**: "Add top-k similar items function"

---

### **PHASE 8: ML MODELS FROM SCRATCH** ⏱️ 4-6 giờ (BONUS POINTS!)

#### File: `src/models.py`

**Function 1: matrix_factorization_sgd()**
```python
def matrix_factorization_sgd(R, K, steps=5000, alpha=0.002, beta=0.02):
    """
    Matrix Factorization using Stochastic Gradient Descent.
    
    Factorize R (user-item matrix) into P (user features) and Q (item features).
    R ≈ P × Q^T
    
    Args:
        R: User-item rating matrix
        K: Number of latent factors
        steps: Training iterations
        alpha: Learning rate
        beta: Regularization parameter
    
    Returns:
        P, Q matrices
    """
```

**Commit**: "Add matrix factorization SGD"

**Function 2: svd_numpy()**
```python
def svd_numpy(matrix, k_factors):
    """
    Truncated SVD using NumPy's linalg.svd.
    
    Returns:
        U, Sigma, Vt reduced to k factors
    """
```

**Commit**: "Add SVD implementation"

**Function 3: predict_rating()**
```python
def predict_rating(P, Q, user_idx, item_idx):
    """
    Predict rating for user-item pair.
    
    Formula: r_ui = p_u · q_i^T
    """
```

**Commit**: "Add rating prediction function"

**Function 4: rmse()**
```python
def rmse(true_ratings, predicted_ratings):
    """
    Root Mean Squared Error.
    
    Formula: RMSE = sqrt(mean((y_true - y_pred)^2))
    """
```

**Commit**: "Add RMSE metric"

**Function 5: mae()**
**Commit**: "Add MAE metric"

**Function 6: precision_at_k()**
```python
def precision_at_k(true_items, recommended_items, k):
    """
    Precision@K: Fraction of relevant items in top-K.
    """
```

**Commit**: "Add Precision@K metric"

**Function 7: recall_at_k()**
**Commit**: "Add Recall@K metric"

**Function 8: hit_rate()**
**Commit**: "Add hit rate metric"

**Function 9: train_test_split_numpy()**
```python
def train_test_split_numpy(data, test_size=0.2, random_seed=42):
    """
    Split data into train/test sets.
    Pure NumPy implementation.
    """
```

**Commit**: "Add train-test split function"

**Function 10: cross_validate()**
```python
def cross_validate(data, model_func, k_folds=5):
    """
    K-fold cross-validation implementation.
    Pure NumPy.
    """
```

**Commit**: "Add cross-validation function"

---

### **PHASE 9: MODELING NOTEBOOK** ⏱️ 4-5 giờ

#### Notebook: `notebooks/03_modeling.ipynb`

**Section 1: Load Processed Data**
**Commit**: "Add modeling notebook - data loading"

**Section 2: Create User-Item Matrix**
- Load similarity functions
- Build matrix
- Visualize sparsity pattern

**Commit**: "Add user-item matrix section"

**Section 3: Popularity-Based Recommendations (Baseline)**
- Rank products by rating count
- Top-N most popular
- Evaluate baseline performance

**Markdown cell**: Cold start solution

**Commit**: "Add popularity-based recommendations"

**Section 4: User-Based Collaborative Filtering**
- Compute user-user similarities
- Find similar users
- Generate recommendations
- Evaluate performance

**Visualizations**:
- Similarity heatmap (sample)
- Recommendation examples

**Commit**: "Add user-based CF"

**Section 5: Item-Based Collaborative Filtering**
- Compute item-item similarities
- Find similar items
- Generate recommendations
- Evaluate performance

**Commit**: "Add item-based CF"

**Section 6: Matrix Factorization (SVD)**
- Apply SVD from scratch
- Tune number of latent factors
- Train/test split
- Evaluate RMSE

**Visualizations**:
- RMSE vs # factors
- Latent factor interpretation (if possible)

**Commit**: "Add SVD-based recommendations"

**Section 7: Matrix Factorization (SGD)**
- Train MF with gradient descent
- Learning curves (training loss)
- Hyperparameter tuning
- Compare with SVD

**Commit**: "Add SGD-based MF"

**Section 8: Model Comparison**
- Compare all models
- Table of metrics (RMSE, Precision@K, Recall@K)
- Visualize comparison

**Markdown cell**: Which model works best and why

**Commit**: "Add model comparison section"

**Section 9: Recommendation Examples**
- Show recommendations for specific users
- Explain why recommended
- Diversity analysis

**Commit**: "Add recommendation examples"

**Section 10: Business Insights**
- Key findings
- Recommendations for business
- Limitations
- Future improvements

**Commit**: "Add business insights section"

---

### **PHASE 10: VISUALIZATION MODULE** ⏱️ 1-2 giờ

#### File: `src/visualization.py`

Reusable plotting functions used across notebooks:

**Function 1: plot_distribution()**
**Function 2: plot_time_series()**
**Function 3: plot_heatmap()**
**Function 4: plot_comparison_bar()**
**Function 5: plot_scatter_with_trend()**
**Function 6: plot_learning_curve()**

Mỗi function 1 commit.

---

### **PHASE 11: COMPREHENSIVE README** ⏱️ 2-3 giờ

#### File: `README.md`

**Cấu trúc đầy đủ 13 sections**:

1. **Title & Description** (1 đoạn)
2. **Table of Contents** (links to sections)
3. **Introduction** (3-4 đoạn):
   - Problem statement
   - Motivation & real-world applications
   - Specific objectives
4. **Dataset** (2-3 đoạn):
   - Source (Kaggle link)
   - Feature descriptions table
   - Size & characteristics
5. **Method** (5-6 đoạn):
   - Data processing pipeline diagram/steps
   - Algorithms with mathematical formulas
   - NumPy implementation details
6. **Installation & Setup** (code blocks)
7. **Usage** (step-by-step commands)
8. **Results** (3-4 đoạn + images):
   - Metrics table
   - Visualizations
   - Comparison & analysis
9. **Project Structure** (tree + explanations)
10. **Challenges & Solutions** (3-4 bullet points)
11. **Future Improvements** (3-4 ideas)
12. **Contributors** (name, ID, email)
13. **License** (MIT)

**Commit mỗi section**: 
- "Add README introduction"
- "Add README dataset section"
- etc.

---

### **PHASE 12: FINAL REVIEW & OPTIMIZATION** ⏱️ 2-3 giờ

#### Checklist:

**Code Quality**:
- [ ] No unnecessary comments
- [ ] Concise docstrings only
- [ ] No fancy print statements
- [ ] Clean, readable code
- [ ] No for loops on arrays (vectorized)

**Notebooks**:
- [ ] Run all cells from top to bottom (no errors)
- [ ] Markdown cells explain insights (not in code cells)
- [ ] Visualizations have titles, labels, legends
- [ ] Clear flow: exploration → preprocessing → modeling

**NumPy Mastery Checklist**:
- [ ] Broadcasting used extensively
- [ ] Fancy indexing & boolean masking
- [ ] np.einsum for complex operations
- [ ] Memory-efficient (views vs copies)
- [ ] Numerical stability considerations
- [ ] Statistical calculations from scratch

**Commits**:
- [ ] Each function has its own commit
- [ ] Meaningful commit messages
- [ ] Commit history shows progression

**Testing**:
- [ ] Run notebook 01: No errors
- [ ] Run notebook 02: No errors
- [ ] Run notebook 03: No errors
- [ ] Check all visualizations render correctly

**Final commit**: "Final optimization and polish"

---

## 🎯 EXPECTED SCORING

### Presentation (20%)
- ✅ **Notebooks (10%)**: 3 notebooks, rõ ràng, visualizations đẹp → **10/10**
- ✅ **GitHub (10%)**: Cấu trúc chuẩn, README đầy đủ → **10/10**

### NumPy Techniques (50%)
- ✅ **Vectorization (10%)**: Không for loops, broadcasting → **10/10**
- ✅ **Advanced NumPy (15%)**: Fancy indexing, einsum, memory efficiency → **15/15**
- ✅ **Math Operations (10%)**: Numerical stability, hypothesis testing → **10/10**
- ✅ **Code Efficiency (15%)**: Clean, modular, reproducible → **15/15**

### Results (30%)
- ✅ **Model Performance (15%)**: Multiple metrics, comparison → **15/15**
- ✅ **Insights (15%)**: Interesting questions, data-driven answers → **15/15**

### Bonus (10%)
- ✅ **ML from scratch (7%)**: SVD, MF-SGD, metrics, cross-validation → **7/10**
- ✅ **Interesting analysis (3%)**: Temporal patterns, user segmentation → **3/10**

**TOTAL: 110/100** 🎉

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Commit strategy**: 
   - Mỗi function nhỏ → 1 commit
   - Format: "Add [function_name] function" hoặc "Add [section_name] section"
   
2. **Code style**:
   - Không comment dài dòng trong code
   - Docstring ngắn gọn, đủ ý
   - Không print fancy (===, ***, etc.)
   - Chỉ print thông tin cần thiết

3. **Markdown cells trong notebooks**:
   - Giải thích insights NGOÀI code cells
   - Không viết analysis trong code cells

4. **Testing**:
   - Sau mỗi phase lớn, chạy notebook để check lỗi
   - Fix ngay nếu có lỗi

5. **Reproducibility**:
   - Set random seeds (np.random.seed(42))
   - Document versions trong requirements.txt

---

## 📅 TIMELINE ƯỚC TÍNH

- **Total time**: 30-40 giờ làm việc
- **Breakdown**:
  - Setup: 1 giờ
  - Data loading: 2 giờ
  - Exploration: 4 giờ
  - Preprocessing functions: 4 giờ
  - Feature engineering: 3 giờ
  - Preprocessing notebook: 3 giờ
  - Similarity functions: 3 giờ
  - ML from scratch: 6 giờ (BONUS)
  - Modeling notebook: 5 giờ
  - Visualization module: 2 giờ
  - README: 3 giờ
  - Final review: 3 giờ

---

## 🚀 SẴN SÀNG BẮT ĐẦU

Bạn muốn tôi bắt đầu từ Phase nào? Tôi đề xuất:

**Option 1**: Bắt đầu từ Phase 1 (Setup) → làm tuần tự
**Option 2**: Bắt đầu từ Phase 2 (Data Loader) → vì structure đã có sẵn một phần

Bạn chọn cách nào? Hoặc có điều chỉnh gì trong plan không?



