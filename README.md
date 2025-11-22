# Lab2DS — Amazon Beauty Recommendation System

**NumPy-first recommendation system** implementing multiple algorithms from scratch for CSC17104 - Programming for Data Science.

**Overview:**
- **Architecture**: 3 Notebooks (exploration → preprocessing → modeling) + reusable `src/` modules
- **Dataset**: Amazon Beauty ratings (2.02M ratings, 1.21M users, 249K products)
- **Models**: Popularity, ItemCF, UserCF, SVD (from scratch), ALS (from scratch)
- **Constraints**: NumPy-only (no pandas/sklearn for core processing)
- **Storage**: Compressed `.npz` files in `data/processed/`

---

## Dataset

**Source:** Amazon Beauty Ratings (Kaggle)  
**Timeframe:** 2010-05-19 to 2014-07-07 (4.1 years)

### Statistics

#### Raw Data
| Metric | Value |
|--------|-------|
| **Total ratings** | 2,023,070 |
| **Unique users** | 1,210,271 |
| **Unique products** | 249,274 |
| **Mean rating** | 4.149 / 5.0 |
| **Median rating** | 5.0 |
| **Distribution** | 1★: 9.1%, 2★: 5.6%, 3★: 8.4%, 4★: 15.2%, 5★: 61.7% |

#### After Preprocessing (min 5 ratings per user/product)
| Metric | Value |
|--------|-------|
| **Records** | ~199K |
| **Users** | ~22K |
| **Products** | ~12K |
| **Sparsity** | ~99.93% |

---

## Workflow

Run notebooks **in order**: 01 → 02 → 03. Each saves outputs for the next.

### 1. `01_data_exploration.ipynb` — Data Exploration

**Goal:** Understand dataset characteristics and identify patterns.

**Research Questions:**
- How are ratings distributed? (Bias check)
- What is user engagement pattern?
- What is product popularity distribution?
- Are there temporal trends?
- How sparse is the user-item matrix?

**Outputs:**
- `data/processed/exploration_outputs.npz` (summary statistics)

**Runtime:** ~1 minute

---

### 2. `02_preprocessing.ipynb` — Data Preprocessing

**Goal:** Prepare clean data for modeling.

**Steps:**
1. Load raw data
2. Filter users/products (min 5 ratings each)
3. Create index mappings (string ID → integer index)
4. Temporal train/test split (80/20)
5. Compute user/product statistics

**Outputs:**
- `data/processed/preprocessed_data.npz` (train/test splits)
- `data/processed/id_mappings.npz` (ID mappings)
- `data/processed/user_stats.npy`, `product_stats.npy`

**Runtime:** ~2 minutes

---

### 3. `03_modeling.ipynb` — Model Training & Evaluation

**Goal:** Train and compare recommendation models.

**Models:**
1. **Popularity** — Recommend most popular items
2. **ItemCF** — Item-based collaborative filtering (k=20)
3. **UserCF** — User-based collaborative filtering (k=20, min_overlap=3)
4. **SVD** — Truncated SVD from scratch (50 factors)
5. **ALS** — Alternating Least Squares from scratch (50 factors)

**Metrics:**
- Precision@10, Recall@10, F1@10
- NDCG@10 (ranking quality)
- Coverage (% unique items recommended)
- Diversity (avg dissimilarity)

**Outputs:**
- `results/model_recommendations.npz` (evaluation results)

**Runtime:** ~3-5 minutes (depends on ALS iterations)

---

## Installation & Usage

### Requirements

- Python 3.8+
- NumPy, Matplotlib, Seaborn

### Setup

```bash
# Navigate to project
cd Lab2DS

# Install dependencies
pip install -r requirements.txt

# Place dataset
# Download ratings_Beauty.csv from Kaggle → data/raw/
```

### Run Notebooks

```bash
# Open in VS Code or Jupyter
jupyter notebook notebooks/

# Run in order: 01 → 02 → 03
```

### Quick Tests

```bash
# Test models
python src/models.py

# Test visualization
python src/visualization.py

# Test ALS implementation
python test_als.py
```

---

## Project Structure

```
Lab2DS/
├── data/
│   ├── raw/
│   │   └── ratings_Beauty.csv
│   └── processed/
│       ├── exploration_outputs.npz
│       ├── preprocessed_data.npz
│       ├── id_mappings.npz
│       └── *.npy (user_stats, product_stats)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data loading, filtering, feature engineering
│   ├── models.py                # Recommendation algorithms (NumPy-only)
│   └── visualization.py         # Plotting utilities
├── results/
│   └── model_recommendations.npz
├── .github/
│   └── copilot-instructions.md  # AI agent guidelines
├── requirements.txt
├── test_als.py
└── README.md
```

---

## Implementation Details

### Models (Pure NumPy)

**TruncatedSVD:**
- Power iteration method to compute singular vectors
- No sklearn dependency
- Returns U @ Sigma @ V.T factorization

**ALSRecommender:**
- Alternating Least Squares matrix factorization
- Solves least squares with L2 regularization
- Alternates between updating user and item factors

**UserBasedCF:**
- Centered cosine similarity (rating mean subtraction)
- Filters neighbors by minimum overlap count
- Aggregates neighbor ratings with similarity weights

**ItemBasedCF:**
- Cosine similarity between item rating vectors
- k-nearest neighbors for prediction
- Fast prediction using precomputed similarity matrix

### Data Processing

**Filtering:**
- Iterative removal of users/products below min rating threshold
- Converges when no more users/products are removed
- Ensures data density for meaningful recommendations

**Train/Test Split:**
- Temporal split: 80% earliest ratings → train, 20% latest → test
- Preserves temporal order (no future data leakage)
- Uses timestamps for splitting


---

## Coding Conventions

- **NumPy-first**: Vectorized operations; avoid pandas/scipy for core processing
- **Prints**: Concise, Vietnamese in notebooks; NO decorative characters (===, checkmarks)
- **Comments**: Short Vietnamese; keep English technical terms (SVD, CF, cosine)
- **Persistence**: Save `.npz` to `data/processed/`
- **Notebooks**: Idempotent, top-to-bottom runnable, reuse outputs

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Large raw data (2M+ rows) | Vectorized NumPy operations (no loops) |
| Extreme sparsity (99.93%) | Iterative filtering, matrix factorization |
| Cold-start users/items | Popularity baseline, hybrid approaches |
| Metric computation | Careful implementation: Precision, Recall, NDCG |
| Notebook reusability | Save artifacts to `data/processed/` |

---

## Future Work

- [ ] Hyperparameter tuning (grid search)
- [ ] Cross-validation for robust evaluation
- [ ] Ensemble methods (hybrid models)
- [ ] Neural Collaborative Filtering (NCF)
- [ ] Online learning for real-time updates

---

## References

- **Dataset:** [Amazon Beauty Ratings](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings) (Kaggle)
- **Course:** CSC17104 - Programming for Data Science (HCMUS)
- **Student:** Angela - MSSV: 23122030

---

## License

Educational project for CSC17104. Dataset credit to Amazon and Kaggle contributors.
- [ ] A/B testing framework
- [ ] FastAPI server
- [ ] Docker containerization

---

## Contributors

- **Angela** (MSSV: 23122030) — CSC17104 Programming for Data Science

---

## License

Educational project for CSC17104.

---

**Last Updated:** 2025-11-17  
**Status:** Notebooks 01-03 runnable; models evaluated✓
