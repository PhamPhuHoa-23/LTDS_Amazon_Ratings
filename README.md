# Hệ Thống Gợi Ý Sản Phẩm Amazon Beauty

**Đồ án môn:** CSC17104 - Lập Trình Cho Khoa Học Dữ Liệu  
**Sinh viên:** Phạm Phú Hòa  
**MSSV:** 23122030  
**Năm học:** 2025-2026

---

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
4. [Cài Đặt](#cài-đặt)
5. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
6. [Chi Tiết Notebooks](#chi-tiết-notebooks)
7. [Kết Quả](#kết-quả)
8. [Tham Khảo](#tham-khảo)
9. [License](#license)

---

## Giới Thiệu

Dự án này xây dựng một **hệ thống gợi ý sản phẩm** (Recommendation System) hoàn chỉnh cho lĩnh vực mỹ phẩm (Beauty) trên nền tảng Amazon, sử dụng dữ liệu đánh giá (ratings) thực tế.

---

## Dataset

### Nguồn Dữ Liệu

**Dataset:** Amazon Beauty Ratings  
**Nguồn:** [Amazon Product Data (Kaggle)](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)  
**File:** `ratings_Beauty.csv`  
**Khoảng thời gian:** 1998-10-19 đến 2014-07-23 (15.8 năm)

### Thống Kê Dữ Liệu Gốc

| Chỉ Số | Giá Trị |
|---------|---------|
| Tổng số ratings | 2,023,070 |
| Số lượng users | 1,210,271 |
| Số lượng products | 249,274 |
| Rating trung bình | 4.149 / 5.0 |
| Rating median | 5.0 |
| Độ lệch chuẩn | 1.312 |

### Phân Bố Ratings

| Rating | Số Lượng | Tỷ Lệ |
|--------|----------|-------|
| 1 sao | 183,784 | 9.1% |
| 2 sao | 113,034 | 5.6% |
| 3 sao | 169,791 | 8.4% |
| 4 sao | 307,740 | 15.2% |
| 5 sao | 1,248,721 | **61.7%** |

**Nhận xét:** Dữ liệu có xu hướng nghiêng về ratings cao (positive bias), với hơn 61\% là 5 sao.

### Dữ Liệu Sau Tiền Xử Lý

Sau khi lọc (chỉ giữ users và products có ít nhất 5 ratings):

| Chỉ Số | Giá Trị | Tỷ Lệ Giữ Lại |
|---------|---------|----------------|
| Users | 22,480 | 1.9% |
| Products | 12,153 | 4.9% |
| Ratings | 199,177 | 9.8% |
| Sparsity | 99.94% | - |

---

## Cấu Trúc Dự Án

```
Lab2DS/
├── data/
│   ├── raw/
│   │   └── ratings_Beauty.csv          # Dataset gốc (download từ Kaggle)
│   └── processed/                       # Dữ liệu đã xử lý (tự động tạo)
│       ├── exploration_outputs.npz      # Kết quả EDA
│       ├── preprocessed_data.npz        # Train/test splits
│       ├── id_mappings.npz              # User/product ID mappings
│       ├── user_stats.npy               # Thống kê users
│       └── product_stats.npy            # Thống kê products
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Phân tích dữ liệu
│   ├── 02_preprocessing.ipynb          # Tiền xử lý
│   └── 03_modeling.ipynb               # Xây dựng & đánh giá models
│
├── src/                                 # Module Python tái sử dụng
│   ├── __init__.py
│   ├── data_processing.py              # Load, filter, feature engineering
│   ├── models.py                        # Các thuật toán recommendation
│   ├── evaluation.py                    # Metrics đánh giá
│   └── visualization.py                 # Vẽ biểu đồ
│
├── results/
│   └── model_recommendations.npz        # Kết quả đánh giá models
│
├── requirements.txt                     # Dependencies
└── README.md
```

---

## Cài Đặt

### Yêu Cầu Hệ Thống

- **Python:** 3.8 trở lên
- **Thư viện:** NumPy, Matplotlib, Seaborn
- **RAM:** Tối thiểu 4GB (khuyến nghị 8GB)
- **Dung lượng:** ~500MB (bao gồm dataset)

### Các Bước Cài Đặt

**Bước 1:** Clone repository

```bash
git clone https://github.com/PhamPhuHoa-23/LTDS_Amazon_Ratings.git
cd LTDS_Amazon_Ratings
```

**Bước 2:** Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Bước 3:** Tải dataset

1. Tải file `ratings_Beauty.csv` từ [Kaggle](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)
2. Đặt file vào thư mục `data/raw/`:
   ```
   data/
   └── raw/
       └── ratings_Beauty.csv
   ```

**Bước 4:** Kiểm tra cài đặt

```bash
# Test các module
python src/models.py
```

---

## Hướng Dẫn Sử Dụng

### Chạy Notebooks

**Quan trọng:** Chạy notebooks theo **thứ tự** 01 $\rightarrow$ 02 $\rightarrow$ 03 vì mỗi notebook sử dụng output của notebook trước.

```bash
# Mở Jupyter Notebook hoặc VS Code
jupyter notebook notebooks/

# Hoặc sử dụng VS Code Notebook
code notebooks/
```

**Thứ tự chạy:**
1. `01_data_exploration.ipynb` - Phân tích dữ liệu (~1 phút)
2. `02_preprocessing.ipynb` - Tiền xử lý (~1 phút)
3. `03_modeling.ipynb` - Train & đánh giá models (~15 phút)

### Chạy Module Độc Lập

```bash
# Test model implementations
python src/models.py

# Test visualization functions
python src/visualization.py
```

---

## Chi Tiết Notebooks

### Notebook 01: Data Exploration (Khám Phá Dữ Liệu)

**Mục đích:** Phân tích đặc điểm dữ liệu ratings (phân phối, sparsity, user/product patterns) để xác định chiến lược xây dựng hệ thống khuyến nghị.

#### Các Câu Hỏi Nghiên Cứu

**1. Phân tích phân phối ratings**
- Phân bố ratings như thế nào?
- Có bias không?
- Rating trung bình và độ lệch chuẩn?

**Kết quả:**
- Trung bình: 4.149/5.0, độ lệch chuẩn: 1.312
- Median: 5.0
- Phân phối: 1 sao (9.1%), 2 sao (5.6%), 3 sao (8.4%), 4 sao (15.2%), **5 sao (61.7%)**
- Kết luận: Ratings thiên về 5 sao ($>60\%$), cho thấy bias tích cực phổ biến trong e-commerce

**2. Phân tích hoạt động users**
- User trung bình đánh giá bao nhiêu sản phẩm?
- Có "power users" không?
- Phân bố hoạt động user như thế nào?

**Kết quả:**
- Tổng users: 1,210,271
- Trung bình: 1.67 ratings/user, Median: 1 rating
- Max ratings của 1 user: 389
- Percentile 90%: 3 ratings, Percentile 95%: 5 ratings, Percentile 99%: 15 ratings
- Power users ($\geq 10$ ratings): $\sim 1.0\%$
- Casual users ($<5$ ratings): $\sim 95.7\%$
- Kết luận: Phần lớn users có ít ratings, power users rất hiếm nhưng quan trọng cho hệ khuyến nghị

**3. Phân tích độ phổ biến products**
- Phân phối ratings cho products?
- Có "blockbuster products" không?
- Bao nhiêu products có ít ratings (cold start)?

**Kết quả:**
- Tổng products: 249,274
- Trung bình: 8.11 ratings/product, Median: 2 ratings
- Max ratings cho 1 product: 7,533
- Products chỉ có 1 rating: 103,484 (41.5%)
- Top 10 products được rate nhiều nhất có từ 3,000+ đến 7,533 ratings
- Kết luận: Phân phối long-tail - nhiều products có ít ratings (cold start problem)

**4. Phân tích xu hướng thời gian**
- Có xu hướng tăng/giảm theo thời gian không?
- Khoảng thời gian dữ liệu?
- Hoạt động rating có biến động theo thời gian?

**Kết quả:**
- Khoảng thời gian: 1998-10-19 đến 2014-07-23 (15.8 năm)
- Kết luận: Hoạt động rating tương đối ổn định theo thời gian

**5. Phân tích sparsity**
- Ma trận user-item có sparse như thế nào?
- Mật độ dữ liệu thực tế?

**Kết quả:**
- Users: 1,210,271, Products: 249,274
- Tổng khả thi: 301,751,932,054 interactions
- Thực tế: 2,023,070 ratings
- **Sparsity: 99.999329% (0.000671% density)**
- Kết luận: Sparsity cực cao ($>99.9\%$) - cần matrix factorization (SVD, ALS), filtering users/products theo min ratings, chiến lược xử lý cold-start

#### Output

- File: `data/processed/exploration_outputs.npz`
- Chứa: `n_users`, `n_products`, `n_ratings`, `sparsity`

---

### Notebook 02: Preprocessing (Tiền Xử Lý)

**Mục đích:** Lọc dữ liệu (loại users/products có ít ratings), tạo index mappings, chia train/test theo thời gian, lưu artifacts cho modeling.

#### Quy Trình Xử Lý

**Bước 1: Load dữ liệu gốc**
- Input: `data/raw/ratings_Beauty.csv`
- Format: user_id, product_id, rating, timestamp
- Dữ liệu gốc: 1,210,271 users, 249,274 products, 2,023,070 ratings

**Bước 2: Lọc dữ liệu theo số ratings tối thiểu**

**Tiêu chí lọc:**
- Users: Giữ lại users có **ít nhất 5 ratings**
- Products: Giữ lại products có **ít nhất 5 ratings**
- Lọc lặp đi lặp lại cho đến khi không còn users/products nào bị loại

**Lý do:**
- Users $<5$ ratings: Không đủ để phân tích behavior pattern
- Products $<5$ ratings: Cold start problem, ít tín hiệu để collaborative filtering

**Kết quả sau lọc:**
- Users: 22,480 (giữ lại 1.9%)
- Products: 12,153 (giữ lại 4.9%)
- Ratings: 199,177 (giữ lại 9.8%)

**Bước 3: Tạo Index Mappings**

Chuyển string IDs sang integer indices (0-based) cho NumPy arrays:
- User IDs $\rightarrow$ User indices [0, 22,479]
- Product IDs $\rightarrow$ Product indices [0, 12,152]

Cơ chế:
```python
unique_users = np.unique(filtered_users)
user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
user_indices = np.array([user_to_idx[u] for u in filtered_users])
```

Tương tự cho products.

**Bước 4: Chia Train/Test**

**Phương pháp:** Temporal split (chia theo thời gian)

**Cơ chế:**
```python
# 1. Sort theo timestamp
sorted_indices = np.argsort(timestamps)

# 2. Tính split point
split_idx = int(len(ratings) * 0.8)

# 3. Chia
train_indices = sorted_indices[:split_idx]
test_indices = sorted_indices[split_idx:]
```

**Đặc điểm:**
- Không random shuffle
- Đảm bảo tính temporal consistency (test set chứa ratings mới hơn train set)
- Mô phỏng real-world scenario (dự đoán tương lai từ quá khứ)
- Tỷ lệ: 80% train, 20% test

**Kết quả:**
- Train: 159,342 ratings
- Test: 39,835 ratings

**Bước 5: Tính thống kê cho users và products**

Tính các features:
- Số ratings cho mỗi user/product
- Average rating cho mỗi user/product
- Rating variance

**Bước 6: Xây dựng user-item matrix**

Tạo dense matrix kích thước (n_users × n_products):
```python
train_matrix = np.zeros((n_users, n_products))
for user, product, rating in zip(train_users, train_products, train_ratings):
    train_matrix[user, product] = rating
```

Kết quả:
- Shape: $(22{,}480 \times 12{,}153)$
- Non-zero entries: 159,342
- Sparsity: $99.94\%$

#### Output Files

| File | Nội Dung |
|------|----------|
| `preprocessed_data.npz` | `train_users`, `train_products`, `train_ratings`, `test_users`, `test_products`, `test_ratings`, `n_users`, `n_products` |
| `id_mappings.npz` | `user_to_idx`, `idx_to_user`, `product_to_idx`, `idx_to_product` |
| `user_stats.npy` | Thống kê users (avg rating, count, variance) |
| `product_stats.npy` | Thống kê products (avg rating, count, variance) |

---

### Notebook 03: Modeling (Xây Dựng Models)

**Mục đích:** Train 4 recommendation models (Popularity, ItemCF, SVD, ALS) và so sánh hiệu năng qua metrics (Precision, Recall, F1, NDCG, Coverage).

#### Các Phương Pháp

##### 1. Popularity Recommender

**Cơ chế:**
- Recommend các sản phẩm phổ biến nhất (nhiều ratings nhất)
- Không cá nhân hóa (tất cả users nhận cùng recommendations)

**Công thức:**
```
score(item) = count(ratings for item)
```

**Ưu điểm:**
- Đơn giản, nhanh
- Giải quyết cold start problem

**Nhược điểm:**
- Không cá nhân hóa
- Coverage thấp (chỉ recommend popular items)

**Training time:** ~0.03s

---

##### 2. ItemCF (Item-based Collaborative Filtering)

**Cơ chế:**
- "Users thích item A cũng thích item B"
- Tính similarity giữa items dựa trên user ratings
- Recommend items tương tự với items user đã thích

**Công thức:**

1. **Item similarity** (Cosine similarity):

$$\text{sim}(i, j) = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\| \times \|\mathbf{r}_j\|}$$

Trong đó $\mathbf{r}_i$, $\mathbf{r}_j$ là rating vectors của item $i$ và $j$

2. **Prediction score**:

$$\text{score}(u, i) = \frac{\sum_{j \in N(i)} \text{sim}(i, j) \times r_{uj}}{\sum_{j \in N(i)} |\text{sim}(i, j)|}$$

Với $j \in$ top-K similar items mà user $u$ đã rate

**Tối ưu hóa:**
- Pre-compute toàn bộ item-item similarity matrix khi training
- Sử dụng vectorized matrix operations (NumPy)
- Top-K filtering với `np.argpartition` (nhanh hơn full sort)

**Tham số:**
- k = 20 (số neighbors)

**Ưu điểm:**
- Coverage cao (recommend nhiều items khác nhau)
- Stable (item similarity ít thay đổi)

**Nhược điểm:**
- Training chậm (tính similarity matrix)
- Memory intensive (lưu n_items × n_items matrix)

**Training time:** ~36.75s

---

##### 3. SVD (Singular Value Decomposition)

**Cơ chế:**
- Matrix factorization: phân rã user-item matrix thành 2 latent factor matrices
- Giảm chiều dữ liệu (dimensionality reduction)

**Công thức:**

$$R \approx U \times \Sigma \times V^T$$

Trong đó:
- $R$: user-item matrix $(22{,}480 \times 12{,}153)$
- $U$: user factors $(22{,}480 \times 50)$
- $\Sigma$: singular values $(50)$
- $V^T$: item factors $(50 \times 12{,}153)$

**Implementation:**
- Randomized SVD (Halko et al. 2011)
- Power iteration để tính singular vectors
- Không dùng sklearn (pure NumPy)

**Tham số:**
- n_components = 50 (latent factors)
- n_iterations = 5 (power iterations)

**Ưu điểm:**
- Xử lý tốt sparsity
- Tìm latent patterns

**Nhược điểm:**
- Training chậm
- Overfitting risk

**Training time:** ~221.09s

---

##### 4. ALS (Alternating Least Squares)

**Cơ chế:**
- Matrix factorization với implicit feedback
- Xen kẽ optimize user factors và item factors

**Công thức:**

1. **Objective function**:

$$\min_{U, V} \sum_{(u,i) \in \text{observed}} c_{ui}(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda(\|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2)$$

2. **Update rules** (với Conjugate Gradient):

$$\mathbf{u}_u = \arg\min_{\mathbf{u}_u} \sum_{i} c_{ui}(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda\|\mathbf{u}_u\|^2$$

$$\mathbf{v}_i = \arg\min_{\mathbf{v}_i} \sum_{u} c_{ui}(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda\|\mathbf{v}_i\|^2$$

**Tối ưu hóa:**
- Conjugate Gradient thay vì Cholesky decomposition
- Pre-compute $Y^T Y$ để tránh materialized large matrices
- Complexity: $O(N^2)$ thay vì $O(N^3)$

**Tham số:**
- n_factors = 50
- n_iterations = 10
- lambda_reg = 0.01

**Ưu điểm:**
- Tốt cho implicit feedback
- Scalable

**Nhược điểm:**
- Nhiều hyperparameters
- Cần tune cẩn thận

**Training time:** ~154.88s

---

#### Metrics Đánh Giá

**Evaluation set:**
- 15,422 test users có ít nhất 1 relevant item (rating $\geq$ 4)
- $K = 10$ (Top-10 recommendations)

**Metrics:**

1. **Precision@K**: Tỷ lệ items được recommend là relevant

$$\text{Precision@K} = \frac{|\text{Recommended} \cap \text{Relevant}|}{K}$$

2. **Recall@K**: Tỷ lệ relevant items được recommend

$$\text{Recall@K} = \frac{|\text{Recommended} \cap \text{Relevant}|}{|\text{Relevant}|}$$

3. **F1@K**: Harmonic mean của Precision và Recall

$$\text{F1@K} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

4. **NDCG@K**: Normalized Discounted Cumulative Gain (xét thứ tự ranking)

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$$

Trong đó $\text{rel}_i = 1$ nếu item thứ $i$ là relevant, $0$ nếu không.

5. **Coverage**: Tỷ lệ unique items được recommend

$$\text{Coverage} = \frac{|\bigcup \text{Recommended items}|}{|\text{All items}|}$$

---

#### Kết Quả

**Bảng tổng hợp:**

| Model | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Coverage | N_users |
|-------|--------------|-----------|-------|---------|----------|---------|
| Popularity | 0.0037 | 0.0210 | 0.0060 | 0.0123 | 0.08% | 15,422 |
| ItemCF | 0.0091 | 0.0436 | 0.0140 | 0.0308 | 97.02% | 15,418 |
| SVD | 0.0158 | 0.0634 | 0.0229 | 0.0439 | 4.67% | 15,418 |
| **ALS** | **0.0200** | **0.0892** | **0.0299** | **0.0611** | 8.20% | 15,422 |

**Model tốt nhất theo từng metric:**

| Metric | Model | Score |
|--------|-------|-------|
| Precision@10 | **ALS** | 0.0200 |
| Recall@10 | **ALS** | 0.0892 |
| F1@10 | **ALS** | 0.0299 |
| NDCG@10 | **ALS** | 0.0611 |
| Coverage | **ItemCF** | 0.9702 |

**Phân tích:**

**ALS - Overall Winner**
- Thắng tuyệt đối về accuracy metrics (Precision, Recall, F1, NDCG)
- Precision gấp 5.4x so với Popularity
- Recall đạt 8.92% - cao nhất
- NDCG cao nhất (0.0611) cho thấy ranking quality tốt nhất
- Coverage: 8.20%, cân bằng giữa accuracy và diversity

**ItemCF - Coverage Champion**
- Coverage 97.02% - gần như recommend tất cả items
- Phù hợp cho discovery & diversity
- Recall tốt (4.36%), vượt cả Popularity và SVD
- Tốt cho cold start items

**SVD - Balanced Approach**
- Performance ở giữa ItemCF và ALS
- Recall: 6.34%, NDCG: 0.0439
- Tốt cho dimensionality reduction
- Coverage: 4.67%

**Popularity - Baseline**
- Worst performer về accuracy
- Coverage cực thấp (0.08%) - chỉ recommend popular items
- Nhưng: Nhanh nhất, đơn giản, dễ implement

#### Output

- File: `results/model_recommendations.npz`
- Chứa: Dictionary `results` với metrics cho từng model

---

---

## Tác Giả

**Phạm Phú Hòa**  
MSSV: 23122030  
Email: phamhoa23us@gmail.com / 23122030@student.hcmus.edu.vn
Trường: Đại học Khoa học Tự nhiên - ĐHQG TP.HCM

---

## Tham Khảo

1. **ALS:**
   - Hu, Koren, Volinsky (2008). "Collaborative Filtering for Implicit Feedback Datasets"
   - Takács, Tikk (2012). "Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering"

2. **SVD:**
   - Halko, Martinsson, Tropp (2011). "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions"

3. **Collaborative Filtering:**
   - Sarwar et al. (2001). "Item-based collaborative filtering recommendation algorithms"
   - Koren, Bell, Volinsky (2009). "Matrix Factorization Techniques for Recommender Systems"

---

## License

This project is developed for **learning and research purposes**.

### Dataset License
This project uses **Amazon Product Data** strictly for research and educational purposes.  
All rights to the dataset belong to their respective owners and **are not** covered by this project's license.

### Project License — CC0 1.0 Universal (Public Domain Dedication)

CC0 1.0 Universal

Statement of Purpose

The person who associated a work with this deed has dedicated the work to the public domain by waiving all of their rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.

You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

In no event will the licensors be liable for any damages arising from the use of this software.

For more information, please refer to: [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)

