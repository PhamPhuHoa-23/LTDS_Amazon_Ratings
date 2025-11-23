

### Notebook 03: Modeling & Evaluation

**File**: `notebooks/03_modeling.ipynb`  
**Thời gian chạy**: ~7-8 phút (includes training 4 models + evaluation)  
**Mục đích**: Implement và compare 4 recommendation algorithms from scratch using pure NumPy

**Outputs**:
- Trained model parameters (saved to `data/processed/model_*.npz`)
- Evaluation metrics (Precision@10, Recall@10, F1@10, NDCG@10, Coverage)
- Comparison results

---

#### Load Processed Data

**Input Files** (from Notebook 02):
`python
# Load main data
data = np.load('data/processed/preprocessed_data.npz')
train_users = data['train_users']       # (159342,)
train_products = data['train_products'] # (159342,)
train_ratings = data['train_ratings']   # (159342,)
test_users = data['test_users']         # (39835,)
test_products = data['test_products']   # (39835,)
test_ratings = data['test_ratings']     # (39835,)
n_users = int(data['n_users'])          # 22480
n_products = int(data['n_products'])    # 12153
`

**Summary**: Notebook 03 section added successfully
