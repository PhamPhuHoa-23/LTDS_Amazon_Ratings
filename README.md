# Hệ Thống Gợi Ý Sản Phẩm Amazon Beauty

**Đồ án môn:** CSC17104 - Lập Trình Cho Khoa Học Dữ Liệu  
**Sinh viên:** Phạm Phú Hòa  
**MSSV:** 23122030  
**Năm học:** 2025-2026

---

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Tập Dữ Liệu](#tập-dữ-liệu)
3. [Phương Pháp](#phương-pháp)
4. [Cài Đặt](#cài-đặt)
5. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
6. [Kết Quả](#kết-quả)
7. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
8. [Thách Thức và Giải Pháp](#thách-thức-và-giải-pháp)
9. [Hướng Phát Triển](#hướng-phát-triển)
10. [Tác Giả](#tác-giả)
11. [Tham Khảo](#tham-khảo)
12. [Giấy Phép](#giấy-phép)

---

## Giới Thiệu

### Mô Tả Bài Toán

Xây dựng hệ thống gợi ý sản phẩm mỹ phẩm để dự đoán và đề xuất các sản phẩm phù hợp với từng người dùng dựa trên lịch sử đánh giá của họ và người dùng khác.

### Động Lực và Ứng Dụng Thực Tế

**Vấn đề:**
- Người dùng thường bị quá tải thông tin khi có quá nhiều sản phẩm trên nền tảng thương mại điện tử
- Khó tìm được sản phẩm phù hợp với sở thích cá nhân
- Nhà bán hàng cần công cụ để tăng doanh số và giữ chân khách hàng

**Ứng dụng:**
- Thương mại điện tử: Gợi ý sản phẩm cá nhân hóa (Amazon, Shopee, Lazada)
- Tăng trải nghiệm người dùng và tỷ lệ chuyển đổi
- Bán chéo và bán thêm sản phẩm
- Giảm thời gian tìm kiếm sản phẩm

### Mục Tiêu Cụ Thể

1. **Khám phá dữ liệu:** Phân tích đặc điểm của tập dữ liệu đánh giá sản phẩm mỹ phẩm Amazon
2. **Xử lý dữ liệu:** Làm sạch, lọc và chuẩn bị dữ liệu cho mô hình
3. **Xây dựng mô hình:** Cài đặt 4 thuật toán gợi ý từ đầu bằng NumPy:
   - Dựa trên độ phổ biến (Popularity-based)
   - Lọc cộng tác dựa trên sản phẩm (Item-based Collaborative Filtering)
   - Phân rã giá trị đơn (SVD - Singular Value Decomposition)
   - Bình phương nhỏ nhất xen kẽ (ALS - Alternating Least Squares)
4. **Đánh giá và so sánh:** Sử dụng các độ đo chuẩn (Precision, Recall, F1, NDCG, Coverage)
5. **Cài đặt thuần NumPy:** Không sử dụng thư viện học máy/khoa học dữ liệu như pandas, scikit-learn

---

## Tập Dữ Liệu

### Nguồn Dữ Liệu

**Tập dữ liệu:** Amazon Beauty Ratings  
**Nguồn:** [Amazon Product Data (Kaggle)](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)  
**Tệp tin:** `ratings_Beauty.csv`  
**Khoảng thời gian:** 1998-10-19 đến 2014-07-23 (15.8 năm)

### Mô Tả Các Trường Dữ Liệu

Tập dữ liệu bao gồm 4 trường thông tin:

| Trường | Kiểu Dữ Liệu | Mô Tả |
|---------|--------------|-------|
| `user_id` | Chuỗi | Mã định danh người dùng (ví dụ: A3SGXH7AUHU8GW) |
| `product_id` | Chuỗi | Mã định danh sản phẩm/ASIN (ví dụ: B001MA0QY2) |
| `rating` | Số thực | Điểm đánh giá từ 1.0 đến 5.0 sao |
| `timestamp` | Số nguyên | Thời điểm đánh giá (Unix timestamp) |

### Kích Thước và Đặc Điểm Dữ Liệu

**Thống kê dữ liệu gốc:**

| Chỉ Số | Giá Trị |
|---------|---------|
| Tổng số đánh giá | 2,023,070 |
| Số lượng người dùng | 1,210,271 |
| Số lượng sản phẩm | 249,274 |
| Điểm đánh giá trung bình | 4.149 / 5.0 |
| Trung vị điểm đánh giá | 5.0 |
| Độ lệch chuẩn | 1.312 |

**Phân bố điểm đánh giá:**

| Điểm | Số Lượng | Tỷ Lệ |
|--------|----------|-------|
| 1 sao | 183,784 | 9.1% |
| 2 sao | 113,034 | 5.6% |
| 3 sao | 169,791 | 8.4% |
| 4 sao | 307,740 | 15.2% |
| 5 sao | 1,248,721 | **61.7%** |

**Nhận xét:** Dữ liệu có xu hướng nghiêng về điểm đánh giá cao, với hơn 61% là 5 sao, cho thấy xu hướng tích cực phổ biến trong thương mại điện tử.

**Dữ liệu sau tiền xử lý:**

Sau khi lọc (chỉ giữ người dùng và sản phẩm có ít nhất 5 đánh giá):

| Chỉ Số | Giá Trị | Tỷ Lệ Giữ Lại |
|---------|---------|----------------|
| Người dùng | 22,480 | 1.9% |
| Sản phẩm | 12,153 | 4.9% |
| Đánh giá | 199,177 | 9.8% |
| Độ thưa | 99.94% | - |

---

## Phương Pháp

### Quy Trình Xử Lý Dữ Liệu

**Bước 1: Tải dữ liệu gốc**
- Đầu vào: `data/raw/ratings_Beauty.csv`
- Định dạng: user_id, product_id, rating, timestamp
- Dữ liệu gốc: 1,210,271 người dùng, 249,274 sản phẩm, 2,023,070 đánh giá

**Bước 2: Lọc dữ liệu**

Tiêu chí lọc:
- Người dùng: Giữ lại người dùng có **ít nhất 5 đánh giá**
- Sản phẩm: Giữ lại sản phẩm có **ít nhất 5 đánh giá**
- Lọc lặp đi lặp lại cho đến khi không còn người dùng/sản phẩm nào bị loại

Lý do:
- Người dùng $<5$ đánh giá: Không đủ để phân tích mẫu hành vi
- Sản phẩm $<5$ đánh giá: Vấn đề khởi động lạnh, ít tín hiệu cho lọc cộng tác

**Bước 3: Tạo ánh xạ chỉ số**

Chuyển mã định danh chuỗi sang chỉ số số nguyên (bắt đầu từ 0) cho mảng NumPy:
- Mã người dùng $\rightarrow$ Chỉ số người dùng [0, 22,479]
- Mã sản phẩm $\rightarrow$ Chỉ số sản phẩm [0, 12,152]

**Bước 4: Chia tập huấn luyện và kiểm tra**

Phương pháp: Chia theo thời gian

Cơ chế:
```python
# 1. Sắp xếp theo thời gian
sorted_indices = np.argsort(timestamps)

# 2. Tính điểm chia
split_idx = int(len(ratings) * 0.8)

# 3. Chia
train_indices = sorted_indices[:split_idx]
test_indices = sorted_indices[split_idx:]
```

Đặc điểm:
- Không xáo trộn ngẫu nhiên
- Đảm bảo tính nhất quán về thời gian (tập kiểm tra chứa đánh giá mới hơn tập huấn luyện)
- Mô phỏng tình huống thực tế (dự đoán tương lai từ quá khứ)
- Tỷ lệ: 80% huấn luyện, 20% kiểm tra

Kết quả:
- Huấn luyện: 159,342 đánh giá
- Kiểm tra: 39,835 đánh giá

**Bước 5: Xây dựng ma trận người dùng-sản phẩm**

Tạo ma trận dày kích thước (số_người_dùng × số_sản_phẩm):
```python
train_matrix = np.zeros((n_users, n_products))
for user, product, rating in zip(train_users, train_products, train_ratings):
    train_matrix[user, product] = rating
```

Kết quả:
- Kích thước: $(22{,}480 \times 12{,}153)$
- Phần tử khác không: 159,342
- Độ thưa: $99.94\%$

### Thuật Toán Sử Dụng

#### 1. Gợi Ý Dựa Trên Độ Phổ Biến (Popularity Recommender)

**Cơ chế:**
- Gợi ý các sản phẩm phổ biến nhất (nhiều đánh giá nhất)
- Không cá nhân hóa (tất cả người dùng nhận cùng gợi ý)

**Công thức:**
$$\text{score}(\text{sản phẩm}) = \text{số lượng đánh giá}$$

**Ưu điểm:**
- Đơn giản, nhanh
- Giải quyết vấn đề khởi động lạnh

**Nhược điểm:**
- Không cá nhân hóa
- Độ bao phủ thấp (chỉ gợi ý sản phẩm phổ biến)

**Thời gian huấn luyện:** ~0.03 giây

---

#### 2. Lọc Cộng Tác Dựa Trên Sản Phẩm (ItemCF)

**Cơ chế:**
- "Người dùng thích sản phẩm A cũng thích sản phẩm B"
- Tính độ tương tự giữa các sản phẩm dựa trên đánh giá của người dùng
- Gợi ý các sản phẩm tương tự với sản phẩm người dùng đã thích

**Công thức:**

1. **Độ tương tự sản phẩm** (Cosine similarity):

$$\text{sim}(i, j) = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\| \times \|\mathbf{r}_j\|}$$

Trong đó $\mathbf{r}_i$, $\mathbf{r}_j$ là vector đánh giá của sản phẩm $i$ và $j$

2. **Điểm dự đoán**:

$$\text{score}(u, i) = \frac{\sum_{j \in N(i)} \text{sim}(i, j) \times r_{uj}}{\sum_{j \in N(i)} |\text{sim}(i, j)|}$$

Với $j \in$ top-K sản phẩm tương tự mà người dùng $u$ đã đánh giá

**Cài đặt bằng NumPy:**
- Tính trước toàn bộ ma trận độ tương tự sản phẩm-sản phẩm khi huấn luyện
- Sử dụng các phép toán ma trận vector hóa (NumPy)
- Lọc top-K với `np.argpartition` (nhanh hơn sắp xếp đầy đủ)

**Tham số:**
- k = 20 (số lượng láng giềng)

**Thời gian huấn luyện:** ~36.75 giây

---

#### 3. Phân Rã Giá Trị Đơn (SVD)

**Cơ chế:**
- Phân rã ma trận: phân rã ma trận người dùng-sản phẩm thành 2 ma trận nhân tố ẩn
- Giảm chiều dữ liệu

**Công thức:**

$$R \approx U \times \Sigma \times V^T$$

Trong đó:
- $R$: ma trận người dùng-sản phẩm $(22{,}480 \times 12{,}153)$
- $U$: nhân tố người dùng $(22{,}480 \times 50)$
- $\Sigma$: giá trị đơn $(50)$
- $V^T$: nhân tố sản phẩm $(50 \times 12{,}153)$

**Cài đặt bằng NumPy:**
- SVD ngẫu nhiên (Randomized SVD - Halko et al. 2011)
- Lặp lũy thừa để tính vector đơn
- Không dùng sklearn (thuần NumPy)

**Tham số:**
- n_components = 50 (nhân tố ẩn)
- n_iterations = 5 (số lần lặp lũy thừa)

**Thời gian huấn luyện:** ~221.09 giây

---

#### 4. Bình Phương Nhỏ Nhất Xen Kẽ (ALS)

**Cơ chế:**
- Phân rã ma trận với phản hồi ngầm
- Xen kẽ tối ưu hóa nhân tố người dùng và nhân tố sản phẩm

**Công thức:**

1. **Hàm mục tiêu**:

$$\min_{U, V} \sum_{(u,i) \in \text{quan sát}} c_{ui}(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda(\|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2)$$

2. **Quy tắc cập nhật** (với Conjugate Gradient):

$$\mathbf{u}_u = \arg\min_{\mathbf{u}_u} \sum_{i} c_{ui}(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda\|\mathbf{u}_u\|^2$$

$$\mathbf{v}_i = \arg\min_{\mathbf{v}_i} \sum_{u} c_{ui}(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda\|\mathbf{v}_i\|^2$$

**Cài đặt bằng NumPy:**
- Conjugate Gradient thay vì phân rã Cholesky
- Tính trước $Y^T Y$ để tránh tạo ma trận lớn
- Độ phức tạp: $O(N^2)$ thay vì $O(N^3)$

**Tham số:**
- n_factors = 50
- n_iterations = 10
- lambda_reg = 0.01

**Thời gian huấn luyện:** ~154.88 giây

---

### Độ Đo Đánh Giá

**Tập đánh giá:**
- 15,422 người dùng kiểm tra có ít nhất 1 sản phẩm liên quan (điểm $\geq$ 4)
- $K = 10$ (Top-10 gợi ý)

**Các độ đo:**

1. **Precision@K**: Tỷ lệ sản phẩm được gợi ý là liên quan

$$\text{Precision@K} = \frac{|\text{Đã gợi ý} \cap \text{Liên quan}|}{K}$$

2. **Recall@K**: Tỷ lệ sản phẩm liên quan được gợi ý

$$\text{Recall@K} = \frac{|\text{Đã gợi ý} \cap \text{Liên quan}|}{|\text{Liên quan}|}$$

3. **F1@K**: Trung bình điều hòa của Precision và Recall

$$\text{F1@K} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

4. **NDCG@K**: Lợi ích tích lũy chiết khấu chuẩn hóa (xét thứ tự xếp hạng)

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$$

Trong đó $\text{rel}_i = 1$ nếu sản phẩm thứ $i$ liên quan, $0$ nếu không.

5. **Coverage**: Tỷ lệ sản phẩm duy nhất được gợi ý

$$\text{Coverage} = \frac{|\bigcup \text{Các sản phẩm đã gợi ý}|}{|\text{Tất cả sản phẩm}|}$$

---

## Cài Đặt

### Yêu Cầu Hệ Thống

- **Python:** 3.8 trở lên
- **Thư viện:** NumPy, Matplotlib, Seaborn
- **RAM:** Tối thiểu 4GB (khuyến nghị 8GB)
- **Dung lượng:** khoảng 500MB (bao gồm tập dữ liệu)

### Các Bước Cài Đặt

**Bước 1:** Sao chép kho mã

```bash
git clone https://github.com/PhamPhuHoa-23/LTDS_Amazon_Ratings.git
cd LTDS_Amazon_Ratings
```

**Bước 2:** Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

**Bước 3:** Tải tập dữ liệu

1. Tải tệp `ratings_Beauty.csv` từ [Kaggle](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)
2. Đặt tệp vào thư mục `data/raw/`:
   ```
   data/
   └── raw/
       └── ratings_Beauty.csv
   ```

**Bước 4:** Kiểm tra cài đặt

```bash
# Kiểm tra các mô-đun
python src/models.py
```

---

## Hướng Dẫn Sử Dụng

### Chạy Notebooks

**Quan trọng:** Chạy notebooks theo **thứ tự** 01 $\rightarrow$ 02 $\rightarrow$ 03 vì mỗi notebook sử dụng kết quả của notebook trước.

```bash
# Mở Jupyter Notebook hoặc VS Code
jupyter notebook notebooks/

# Hoặc sử dụng VS Code Notebook
code notebooks/
```

**Thứ tự chạy:**
1. `01_data_exploration.ipynb` - Phân tích dữ liệu (khoảng 1 phút)
2. `02_preprocessing.ipynb` - Tiền xử lý (khoảng 1 phút)
3. `03_modeling.ipynb` - Huấn luyện và đánh giá mô hình (khoảng 15 phút)

### Chạy Mô-đun Độc Lập

```bash
# Kiểm tra cài đặt mô hình
python src/models.py

# Kiểm tra hàm trực quan hóa
python src/visualization.py
```

---

## Kết Quả

### Bảng Tổng Hợp

| Mô Hình | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Coverage | Số người dùng |
|---------|--------------|-----------|-------|---------|----------|---------------|
| Popularity | 0.0037 | 0.0210 | 0.0060 | 0.0123 | 0.08% | 15,422 |
| ItemCF | 0.0091 | 0.0436 | 0.0140 | 0.0308 | 97.02% | 15,418 |
| SVD | 0.0158 | 0.0634 | 0.0229 | 0.0439 | 4.67% | 15,418 |
| **ALS** | **0.0200** | **0.0892** | **0.0299** | **0.0611** | 8.20% | 15,422 |

### Mô Hình Tốt Nhất Theo Từng Độ Đo

| Độ Đo | Mô Hình | Điểm Số |
|-------|---------|---------|
| Precision@10 | **ALS** | 0.0200 |
| Recall@10 | **ALS** | 0.0892 |
| F1@10 | **ALS** | 0.0299 |
| NDCG@10 | **ALS** | 0.0611 |
| Coverage | **ItemCF** | 0.9702 |

### Phân Tích Kết Quả

**ALS - Người Chiến Thắng Tổng Thể**
- Thắng tuyệt đối về các độ đo độ chính xác (Precision, Recall, F1, NDCG)
- Precision gấp 5.4 lần so với Popularity
- Recall đạt 8.92% - cao nhất
- NDCG cao nhất (0.0611) cho thấy chất lượng xếp hạng tốt nhất
- Coverage: 8.20%, cân bằng giữa độ chính xác và tính đa dạng

**ItemCF - Nhà Vô Địch Độ Bao Phủ**
- Coverage 97.02% - gần như gợi ý tất cả sản phẩm
- Phù hợp cho khám phá và tính đa dạng
- Recall tốt (4.36%), vượt cả Popularity và SVD
- Tốt cho các sản phẩm mới

**SVD - Phương Pháp Cân Bằng**
- Hiệu năng ở giữa ItemCF và ALS
- Recall: 6.34%, NDCG: 0.0439
- Tốt cho giảm chiều dữ liệu
- Coverage: 4.67%

**Popularity - Mô Hình Cơ Sở**
- Hiệu năng kém nhất về độ chính xác
- Coverage cực thấp (0.08%) - chỉ gợi ý sản phẩm phổ biến
- Nhưng: Nhanh nhất, đơn giản, dễ cài đặt

### Trực Quan Hóa

Các biểu đồ so sánh được tạo tự động trong notebook 03:
- Biểu đồ cột so sánh 5 độ đo (Precision, Recall, F1, NDCG, Coverage)
- Ma trận so sánh hiệu năng các mô hình

---

## Cấu Trúc Dự Án

```
LTDS_Amazon_Ratings/
├── data/
│   ├── raw/
│   │   └── ratings_Beauty.csv          # Tập dữ liệu gốc (tải từ Kaggle)
│   └── processed/                       # Dữ liệu đã xử lý (tự động tạo)
│       ├── exploration_outputs.npz      # Kết quả phân tích khám phá
│       ├── preprocessed_data.npz        # Tập huấn luyện/kiểm tra
│       ├── id_mappings.npz              # Ánh xạ mã người dùng/sản phẩm
│       ├── user_stats.npy               # Thống kê người dùng
│       └── product_stats.npy            # Thống kê sản phẩm
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Phân tích dữ liệu
│   ├── 02_preprocessing.ipynb          # Tiền xử lý
│   └── 03_modeling.ipynb               # Xây dựng và đánh giá mô hình
│
├── src/                                 # Mô-đun Python có thể tái sử dụng
│   ├── __init__.py
│   ├── data_processing.py              # Tải, lọc, tạo đặc trưng
│   ├── models.py                        # Các thuật toán gợi ý
│   ├── evaluation.py                    # Độ đo đánh giá
│   └── visualization.py                 # Vẽ biểu đồ
│
├── results/
│   └── model_recommendations.npz        # Kết quả đánh giá mô hình
│
├── requirements.txt                     # Thư viện phụ thuộc
└── README.md
```

### Chức Năng Từng Tệp/Thư Mục

**Thư mục `data/`:**
- `raw/`: Lưu trữ tập dữ liệu gốc chưa xử lý
- `processed/`: Lưu trữ dữ liệu đã xử lý, ánh xạ, và thống kê (định dạng `.npz` và `.npy` nén)

**Thư mục `notebooks/`:**
- `01_data_exploration.ipynb`: Phân tích khám phá dữ liệu, trả lời 5 câu hỏi nghiên cứu
- `02_preprocessing.ipynb`: Lọc, ánh xạ, chia tập, lưu kết quả
- `03_modeling.ipynb`: Huấn luyện 4 mô hình, đánh giá, so sánh, trực quan hóa

**Thư mục `src/`:**
- `data_processing.py`: Các hàm tải, lọc, chia tập dữ liệu
- `models.py`: Cài đặt 4 thuật toán gợi ý bằng NumPy thuần
- `evaluation.py`: Các hàm tính độ đo (Precision, Recall, F1, NDCG, Coverage)
- `visualization.py`: Các hàm vẽ biểu đồ

**Thư mục `results/`:**
- Lưu trữ kết quả đánh giá mô hình dạng `.npz`

---

## Thách Thức và Giải Pháp

### 1. Dữ Liệu Cực Kỳ Thưa (Sparsity > 99.9%)

**Thách thức:**
- Ma trận người dùng-sản phẩm có độ thưa 99.999329% (chỉ 0.000671% có dữ liệu)
- Hầu hết người dùng chỉ đánh giá rất ít sản phẩm
- Hầu hết sản phẩm chỉ được đánh giá bởi rất ít người dùng
- Khó khăn trong việc tìm mẫu hành vi và mối tương quan

**Giải pháp:**
- **Lọc dữ liệu:** Chỉ giữ người dùng và sản phẩm có ít nhất 5 đánh giá, giảm độ thưa xuống 99.94%
- **Phân rã ma trận:** Sử dụng SVD và ALS để tìm các nhân tố ẩn, giảm chiều dữ liệu
- **Lọc cộng tác:** ItemCF sử dụng độ tương tự cosine để tìm sản phẩm liên quan
- **Lưu trữ thưa:** Sử dụng định dạng `.npz` nén để tiết kiệm bộ nhớ

### 2. Thời Gian Suy Luận Lâu Của UserCF

**Thách thức:**
- Phương pháp lọc cộng tác dựa trên người dùng (UserCF) đã được cài đặt
- Thời gian suy luận cực kỳ chậm do:
  - Phải tính độ tương tự giữa 22,480 người dùng (hơn 250 triệu cặp)
  - Ma trận độ tương tự người dùng-người dùng quá lớn (22,480 × 22,480)
  - Mỗi lần gợi ý phải tìm láng giềng gần nhất trong không gian lớn

**Giải pháp:**
- **Không sử dụng UserCF:** Quyết định loại bỏ UserCF khỏi đánh giá cuối cùng
- **Tập trung vào ItemCF:** Item-based có thể tính trước và lưu lại (12,153 × 12,153 sản phẩm)
- **Sử dụng ALS:** Phương pháp phân rã ma trận hiệu quả hơn cho số lượng người dùng lớn
- **Tối ưu hóa:** Nếu cần UserCF, có thể dùng kỹ thuật LSH (Locality Sensitive Hashing) hoặc approximate nearest neighbors

### 3. Kết Quả Recall@10 Chỉ Đạt Khoảng 9%

**Thách thức:**
- Mô hình tốt nhất (ALS) chỉ đạt Recall@10 = 8.92%
- Nghĩa là trong số các sản phẩm mà người dùng thực sự thích, hệ thống chỉ gợi ý được khoảng 9%
- Precision@10 cũng thấp (2%)
- Nguyên nhân:
  - Dữ liệu quá thưa
  - Hành vi người dùng đa dạng và khó dự đoán
  - Chỉ sử dụng thông tin đánh giá (không có nội dung sản phẩm, hình ảnh, văn bản)
  - Các sản phẩm mới hoặc ít phổ biến khó được gợi ý

**Giải pháp hiện tại:**
- Chấp nhận kết quả này như baseline cho hệ thống gợi ý chỉ dựa trên collaborative filtering
- Tập trung vào việc cài đặt đúng thuật toán bằng NumPy thuần
- So sánh tương đối giữa các mô hình (ALS vẫn tốt hơn các mô hình khác đáng kể)

**Hướng cải thiện:**
- Xem phần "Hướng Phát Triển" bên dưới

---

## Hướng Phát Triển

### 1. Cải Thiện Độ Chính Xác

- **Kết hợp nội dung (Hybrid Recommender):**
  - Thêm thông tin văn bản (tiêu đề, mô tả sản phẩm)
  - Sử dụng hình ảnh sản phẩm (deep learning)
  - Kết hợp collaborative filtering + content-based filtering

- **Kỹ thuật ensemble:**
  - Kết hợp dự đoán từ nhiều mô hình (ALS + ItemCF + SVD)
  - Weighted average hoặc stacking

### 2. Tối Ưu Hóa Hiệu Năng

- **Điều chỉnh siêu tham số:**
  - Grid search hoặc random search cho ALS (số nhân tố, lambda, số vòng lặp)
  - Tối ưu k cho ItemCF
  - Thử các phương pháp khác nhau cho SVD

- **Thuật toán nâng cao:**
  - Neural Collaborative Filtering (NCF)
  - Deep learning models (AutoEncoder, VAE)
  - Graph-based methods (GraphSAGE, LightGCN)

### 3. Mở Rộng Hệ Thống

- **Triển khai thực tế:**
  - Xây dựng API REST với FastAPI
  - Triển khai trên cloud (AWS, GCP, Azure)
  - Xây dựng giao diện web đơn giản

- **Cập nhật thời gian thực:**
  - Online learning để cập nhật mô hình với dữ liệu mới
  - Incremental training cho ALS

### 4. Đánh Giá Toàn Diện Hơn

- **Thêm độ đo:**
  - Đa dạng (Diversity): Đo mức độ khác biệt giữa các gợi ý
  - Mới lạ (Novelty): Đo mức độ ngạc nhiên của người dùng
  - Serendipity: Gợi ý bất ngờ nhưng hữu ích

- **Kiểm tra A/B:**
  - Thử nghiệm với người dùng thật
  - So sánh hiệu quả kinh doanh (click-through rate, conversion rate)

---

## Tác Giả

**Phạm Phú Hòa**  
Mã số sinh viên: 23122030  
Email: 23122030@student.hcmus.edu.vn

Trường Đại học Khoa học Tự nhiên - Đại học Quốc gia Thành phố Hồ Chí Minh

---

## Tham Khảo

### Bài Báo Khoa Học

1. **ALS (Alternating Least Squares):**
   - Hu, Y., Koren, Y., & Volinsky, C. (2008). "Collaborative Filtering for Implicit Feedback Datasets". IEEE International Conference on Data Mining.
   - Takács, G., & Tikk, D. (2012). "Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering". RecSys.

2. **SVD (Singular Value Decomposition):**
   - Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions". SIAM Review.

3. **Collaborative Filtering:**
   - Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). "Item-based collaborative filtering recommendation algorithms". WWW.
   - Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems". IEEE Computer.

### Tập Dữ Liệu

- Amazon Product Data: [Trang web của Julian McAuley](http://jmcauley.ucsd.edu/data/amazon/)
- Bài báo: "Image-based recommendations on styles and substitutes" (SIGIR 2015)

### Công Cụ và Thư Viện

- Tài liệu NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/

---

## Giấy Phép

Dự án này được phát triển cho **mục đích học tập và nghiên cứu**.

### Giấy Phép Tập Dữ Liệu

Dự án này sử dụng **Amazon Product Data** hoàn toàn cho mục đích nghiên cứu và giáo dục.  
Tất cả quyền đối với tập dữ liệu thuộc về chủ sở hữu tương ứng và **không** được bao phủ bởi giấy phép của dự án này.

### Giấy Phép Dự Án - CC0 1.0 Universal (Cống Hiến Phạm Vi Công Cộng)

**CC0 1.0 Universal**

**Tuyên Bố Mục Đích**

Người liên kết tác phẩm này với văn bản pháp lý đã cống hiến tác phẩm cho phạm vi công cộng bằng cách từ bỏ tất cả quyền của họ đối với tác phẩm trên toàn thế giới theo luật bản quyền, bao gồm tất cả các quyền liên quan và lân cận, trong phạm vi được pháp luật cho phép.

Bạn có thể sao chép, chỉnh sửa, phân phối và thực hiện tác phẩm, ngay cả cho mục đích thương mại, tất cả mà không cần xin phép.

Trong mọi trường hợp, người cấp giấy phép sẽ không chịu trách nhiệm về bất kỳ thiệt hại nào phát sinh từ việc sử dụng phần mềm này.

Để biết thêm thông tin, vui lòng tham khảo: [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)
