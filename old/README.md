# Federated Learning với CNN-GRU Model

Hệ thống Federated Learning sử dụng model CNN-GRU cho phát hiện xâm nhập IoT.

## Cấu trúc Project

```
ai4fids_project/
├── model.py                          # Model CNN-GRU (PyTorch)
├── Fed.py                            # Federated Learning framework (FedAvg, FedProx)
├── step1_prepare_chunks.py           # Tiền xử lý data thành chunks
├── step2_create_federated_splits.py  # Tạo Non-IID splits cho clients
├── run_federated_training.py         # Script chạy training
└── data/
    ├── IoT_Dataset_2023/             # Data thô (CSV files)
    ├── preprocessed_chunks/          # Data đã xử lý (từ step 1)
    └── federated_splits/             # Data đã chia cho clients (từ step 2)
```

## Yêu cầu

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

## Cách chạy

### **BƯỚC 1: Chuẩn bị Data (nếu chưa có)**

Nếu bạn đã có data trong `./data/federated_splits/`, **BỎ QUA** bước 1 và 2.

```bash
# Step 1: Tiền xử lý data thô
python step1_prepare_chunks.py
```

Kết quả:
- Tạo preprocessed chunks trong `./data/preprocessed_chunks/`
- Lưu scaler và label encoder

---

### **BƯỚC 2: Tạo Federated Splits**

```bash
# Step 2: Tạo Non-IID splits cho clients
python step2_create_federated_splits.py
```

Kết quả:
- Tạo client data: `client_0_data.npz`, `client_1_data.npz`, ...
- Lưu trong `./data/federated_splits/`
- Tạo visualization về phân phối data

---

### **BƯỚC 3: Chạy Federated Learning** ⭐

```bash
# Chạy training với FedAvg hoặc FedProx
python run_federated_training.py
```

**Cấu hình training** (sửa trong `run_federated_training.py`):

```python
CONFIG = {
    # Data
    'data_dir': './data/federated_splits',
    'num_clients': 5,

    # Model
    'input_shape': (46,),      # Số features (kiểm tra từ data của bạn)
    'num_classes': 2,          # Binary classification

    # Training
    'algorithm': 'fedavg',     # 'fedavg' hoặc 'fedprox'
    'num_rounds': 50,          # Số rounds
    'local_epochs': 5,         # Số epochs mỗi client
    'learning_rate': 0.001,
    'batch_size': 64,
    'client_fraction': 1.0,    # Tỉ lệ clients tham gia mỗi round

    # FedProx
    'mu': 0.01,                # Proximal term (chỉ dùng cho FedProx)

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
```

Kết quả:
- Global model: `./results/global_model.pth`
- Training history: `./results/training_history.pkl`
- Biểu đồ: `./results/training_history.png`

---

## Sử dụng Model trong Code

### **1. Import model từ model.py**

```python
from model import CNN_GRU_Model, build_cnn_gru_model

# Tạo model
model = build_cnn_gru_model(input_shape=(100,), num_classes=2)
```

### **2. Sử dụng trong Federated Learning**

```python
from Fed import FederatedServer, FederatedClient
from model import build_cnn_gru_model

# Tạo model
global_model = build_cnn_gru_model(input_shape=(46,), num_classes=2)

# Tạo server
server = FederatedServer(
    model=global_model,
    clients=clients,
    test_loader=test_loader,
    device='cuda'
)

# Train với FedAvg
history = server.train_fedavg(
    num_rounds=50,
    num_epochs=5,
    learning_rate=0.001,
    client_fraction=1.0
)

# Hoặc train với FedProx
history = server.train_fedprox(
    num_rounds=50,
    num_epochs=5,
    mu=0.01,
    learning_rate=0.001,
    client_fraction=1.0
)
```

---

## Chi tiết Model CNN-GRU

**Kiến trúc:**
- **CNN Module**: 3 Conv1D blocks (64 → 128 → 256 filters)
- **GRU Module**: 2 GRU layers (128 → 64 units)
- **MLP Module**: 2 Dense layers (256 → 128)
- **Output**: Softmax layer

**Input shape**: `(batch_size, seq_length)` hoặc `(batch_size, seq_length, 1)`

**Ví dụ:**
```python
import torch
from model import build_cnn_gru_model

# Tạo model
model = build_cnn_gru_model(input_shape=(100,), num_classes=2)

# Test forward pass
batch_size = 4
seq_length = 100
x = torch.randn(batch_size, seq_length)

# Forward
output = model(x)  # Shape: (batch_size, num_classes)
print(output.shape)  # torch.Size([4, 2])
```

---

## Federated Learning Algorithms

### **FedAvg (Federated Averaging)**
- Weighted average của model parameters
- Weight theo số lượng samples của mỗi client

### **FedProx (Federated Proximal)**
- Thêm proximal term: `mu/2 * ||w - w_global||^2`
- Tốt hơn cho Non-IID data và heterogeneous clients

---

## Tips & Troubleshooting

### **1. Kiểm tra input_shape**

Trước khi train, kiểm tra số features trong data:

```python
import numpy as np

# Load client data
data = np.load('./data/federated_splits/client_0_data.npz')
X_train = data['X_train']

print(f"X_train shape: {X_train.shape}")
# Output: (samples, features, 1) hoặc (samples, features)

# Input shape cho model = (số features,)
input_shape = (X_train.shape[1],)
```

### **2. GPU Memory Issues**

Nếu bị out of memory:
- Giảm `batch_size` (ví dụ: 32 thay vì 64)
- Giảm `num_clients` tham gia mỗi round
- Sử dụng `client_fraction < 1.0`

### **3. Thay đổi algorithm**

```python
# FedAvg (đơn giản hơn)
CONFIG['algorithm'] = 'fedavg'

# FedProx (tốt hơn cho Non-IID data)
CONFIG['algorithm'] = 'fedprox'
CONFIG['mu'] = 0.01  # Thử 0.001 - 0.1
```

---

## Tham khảo

- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", 2017
- **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks", 2020

---

## License

MIT License
