# Lab2DS — Hệ Khuyến Nghị Amazon Beauty (NumPy-first)

Repository này triển khai pipeline đơn giản để xử lý dữ liệu ratings và xây dựng các recommender cơ bản cho bộ dữ liệu Amazon Beauty.

**Tóm tắt nhanh:**
- **Kiến trúc**: Notebooks (01→02→03) + mô-đun trong `src/` (NumPy-first)
- **Mục tiêu**: Khám phá → Tiền xử lý → Feature engineering → Xây dựng recommenders (Popularity, CF, SVD)
- **Lưu trữ**: Artifacts lưu trong `data/processed/` (`.npz` compressed)
- **Conventions**: In tiếng Việt, ngắn gọn; không dùng ký tự trang trí

---

## Dữ Liệu

**Nguồn:** Amazon Beauty Ratings từ Kaggle  
**Thời gian:** 2002-06-12 đến 2014-07-23 (12.1 năm)

### Thống Kê Chi Tiết

#### Dữ liệu thô (Raw)
| Metric | Giá trị |
|--------|--------|
| **Total ratings** | 2,023,070 |
| **Unique users** | 1,210,271 |
| **Unique products** | 249,274 |
| **Avg rating** | 4.15 / 5.0 |
| **Median rating** | 5.0 |
| **Distribution** | 1★: 9.1%, 2★: 5.6%, 3★: 8.4%, 4★: 15.2%, 5★: 61.7% |

#### Sau Preprocessing (Min 5 ratings/user và /product)
| Metric | Giá trị |
|--------|--------|
| **Final records** | 198,837 |
| **Final users** | 22,408 |
| **Final products** | 12,140 |
| **Sparsity** | 99.9269% |
| **Avg ratings/user** | 8.87 |
| **Avg ratings/product** | 16.38 |

---

## Quy Trình Pipeline

Chạy **theo thứ tự**: 01 → 02 → 03. Mỗi notebook lưu dữ liệu cho notebook tiếp theo.

### 1. `01_data_exploration.ipynb` — Khám Phá Dữ Liệu

**Mục tiêu:** Load, validate, thống kê, khám phá phân bố ratings.

**Outputs:**
- `data/processed/exploration_outputs.npz`:
  - `ratings`, `timestamps`, `user_ids`, `product_ids`
  - `unique_users`, `unique_products`
  - `user_counts`, `product_counts`

**Thời gian chạy:** ~1 phút

---

### 2. `02_preprocessing.ipynb` — Tiền Xử Lý & Feature Engineering

**Mục tiêu:** Xử lý missing values → Outlier detection → Feature engineering → Normalization → Filtering.

**Các bước:**
1. **Missing Values**: Kiểm tra NaN, invalid timestamps
2. **Outlier Detection**: Validate rating [1, 5], timestamp hợp lệ
3. **Feature Engineering** (23 features):
   - User features: `n_ratings`, `mean_rating`, `std_rating`
   - Product features: `n_ratings`, `mean_rating`, `std_rating`
   - Temporal: `year`, `month`, `weekday`, `days_since`, `recency_weight`
   - Interaction: `user_rating_deviation`, `product_rating_deviation`, `global_rating_deviation`, `user_rating_zscore`
   - Normalized: Min-Max, Z-score, Robust scaling
4. **Data Filtering**: Iterative filtering (min 5 ratings/user, min 5 ratings/product) → converged sau 4 iterations
5. **ID Mapping**: String IDs → Integer indices

**Outputs:**
- `data/processed/preprocessed_data.npz` (all features + indices)
- `data/processed/id_mappings.npz` (user/product mappings)
- `data/processed/metadata.npy` (sparsity, global mean rating)

**Thời gian chạy:** ~5 phút

---

### 3. `03_modeling.ipynb` — Xây Dựng & Đánh Giá Models

**Mục tiêu:** Train recommenders → Evaluate → So sánh.

**Models:**
- **Popularity-based** — Top-N by rating count
- **Item-based CF** — k=10, 20, 50 (cosine similarity)
- **User-based CF** — k=10, 20, 50 (cosine similarity)
- **SVD (TruncatedSVD)** — k=20, 50, 100
- **Weighted Hybrid** — Kết hợp signals

**Metrics:**
- Precision@10, Recall@10, Hit Rate@10, F1 Score
- Coverage, Diversity, Training time

**Comparison Results (sklearn):**

| Model | Precision | Recall | Hit Rate | F1 | Coverage | Diversity | Train (s) |
|-------|-----------|--------|----------|-----|----------|-----------|-----------|
| Popularity | 0.0040 | 0.0400 | 0.0400 | 0.0073 | 0.0008 | 0.0100 | 0.01 |
| ItemCF_k10 | 0.0090 | 0.0900 | 0.0900 | 0.0164 | 0.0602 | 0.7310 | 15.35 |
| ItemCF_k20 | 0.0090 | 0.0900 | 0.0900 | 0.0164 | 0.0602 | 0.7310 | 15.97 |
| ItemCF_k50 | 0.0090 | 0.0900 | 0.0900 | 0.0164 | 0.0602 | 0.7310 | 18.04 |
| UserCF_k10 | 0.0090 | 0.0900 | 0.0900 | 0.0164 | 0.0675 | 0.8280 | 0.10 |
| UserCF_k20 | 0.0090 | 0.0900 | 0.0900 | 0.0164 | 0.0647 | 0.7860 | 0.10 |
| UserCF_k50 | 0.0100 | 0.1000 | 0.1000 | 0.0182 | 0.0567 | 0.6880 | 0.10 |
| SVD_k20 | 0.0090 | 0.0900 | 0.0900 | 0.0164 | 0.0152 | 0.1850 | 0.26 |
| SVD_k50 | 0.0090 | 0.0900 | 0.0900 | 0.0164 | 0.0245 | 0.2970 | 0.20 |
| SVD_k100 | 0.0080 | 0.0800 | 0.0800 | 0.0145 | 0.0325 | 0.3950 | 0.31 |

**Kết luận:**
- **UserCF_k50** có metrics tốt nhất (Recall=0.10, Diversity=0.69)
- **ItemCF** có coverage cao (0.73) nhưng training chậm (~18s)
- **SVD** nhanh nhất nhưng coverage thấp

**Outputs:**
- `results/comparison_sklearn.png` — Biểu đồ so sánh

**Thời gian chạy:** ~1-2 phút

---

## Cài Đặt & Chạy

### Yêu cầu

- Python 3.8+
- NumPy, Matplotlib, Seaborn, scikit-learn

### Setup

```bash
# Clone repo
cd Lab2DS

# Cài dependencies
pip install -r requirements.txt

# Đặt dữ liệu
# Tải ratings_Beauty.csv từ Kaggle → data/raw/
```

### Chạy Notebooks

```bash
# Mở VS Code Notebook hoặc Jupyter
jupyter notebook notebooks/

# Chạy theo thứ tự: 01 → 02 → 03
# Mỗi notebook tải output từ notebook trước
```

### Chạy Scripts (Kiểm tra nhanh)

```bash
# Kiểm tra models
python src/models.py

# Kiểm tra visualization
python src/visualization.py
```

---

## Cấu Trúc Thư Mục

```
Lab2DS/
├── data/
│   ├── raw/
│   │   └── ratings_Beauty.csv        # CSV thô
│   └── processed/
│       ├── exploration_outputs.npz    # From notebook 01
│       ├── preprocessed_data.npz      # From notebook 02
│       ├── id_mappings.npz
│       └── metadata.npy
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py             # Utility functions (vectorized)
│   ├── models.py                      # Recommenders (NumPy)
│   └── visualization.py               # Plot helpers
├── results/
│   └── comparison_sklearn.png         # Generated by notebook 03
├── .github/
│   └── copilot-instructions.md        # AI agent guidelines
├── requirements.txt
└── README.md
```

---

## Coding Conventions

- **NumPy-first**: Vectorized operations; tránh Pandas/SciPy trừ khi cần
- **Prints**: Ngắn gọn, Tiếng Việt ở notebooks; **KHÔNG** ký tự trang trí (===, ✓)
- **Comments**: Tiếng Việt ngắn; giữ English technical terms (SVD, CF, cosine)
- **Persistence**: Lưu `.npz` vào `data/processed/`
- **Notebooks**: Idempotent, top-to-bottom runnable, tái sử dụng outputs

---

## Challenges & Solutions



| Vấn đề | Giải pháp |
|-------|----------|
| Raw data lớn (2M+ rows) | Vectorized NumPy operations (không loop) |
| Dữ liệu very sparse (99.93%) | Filtering iterative, collaborative filtering |
| Feature engineering phức tạp | Modular functions trong `src/` |
| Notebook reusability | Lưu `.npz` vào `data/processed/` |
| Metrics tính toán | Tính kỹ: Precision, Recall, F1, Coverage, Diversity |

---

## Tương Lai

- [ ] Thêm Neural Collaborative Filtering (NCF)
- [ ] Hyperparameter tuning (Grid Search)
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
