# Quick Start Guide - Federated Learning với CNN-GRU

## 🚀 Chạy nhanh (Hoàn toàn tự động)

### **Bước 1: Đảm bảo bạn đã có data**

Kiểm tra folder `./data/federated_splits/` có các file:
```bash
ls ./data/federated_splits/
# Phải có: client_0_data.npz, client_1_data.npz, ...
```

### **Bước 2: Cấu hình trong run_federated_training.py**

Mở file [run_federated_training.py](run_federated_training.py) và chỉnh sửa CONFIG:

```python
CONFIG = {
    'data_dir': './data/federated_splits',  # Đường dẫn đến federated data
    'output_dir': './results',              # Đường dẫn lưu kết quả
    'num_clients': 5,                       # Số lượng clients

    # Model params - TỰ ĐỘNG PHÁT HIỆN (không cần sửa)
    'input_shape': None,   # ✅ Tự động detect
    'num_classes': None,   # ✅ Tự động detect

    # Training params
    'algorithm': 'fedavg',       # 'fedavg' hoặc 'fedprox'
    'num_rounds': 50,            # Số rounds training
    'local_epochs': 5,           # Số epochs mỗi client
    'learning_rate': 0.001,      # Learning rate
    'batch_size': 64,            # Batch size
    'client_fraction': 1.0,      # Tỉ lệ clients tham gia mỗi round

    # FedProx specific
    'mu': 0.01,                  # Proximal term (chỉ dùng cho FedProx)

    # Device - TỰ ĐỘNG DÙNG GPU
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'force_gpu': True,           # ✅ Bắt buộc dùng GPU (set False để cho phép CPU)

    # Visualization
    'eval_every': 1,             # Đánh giá sau mỗi round
}
```

### **Bước 3: Chạy training** ⭐

```bash
python run_federated_training.py
```

**Script sẽ tự động:**
1. ✅ Kiểm tra và setup GPU
2. ✅ Phát hiện `input_shape` và `num_classes` từ data
3. ✅ Load data cho tất cả clients
4. ✅ Khởi tạo model CNN-GRU trên GPU
5. ✅ Train với FedAvg/FedProx
6. ✅ Lưu kết quả và visualizations

---

## 📊 Output

Sau khi chạy xong, bạn sẽ có trong `./results/`:
- `global_model.pth` - Model đã train
- `training_history.png` - Biểu đồ loss & accuracy
- `training_history.pkl` - Training history data
- `config.pkl` - Configuration đã dùng
- `data_statistics.json` - Thống kê data

---

## 🎮 GPU Monitoring

Script sẽ tự động hiển thị:
```
🔧 KIỂM TRA GPU
================================================================================
CUDA Available: True
CUDA Version: 12.1
PyTorch Version: 2.x.x
Number of GPUs: 1

GPU 0:
  Name: NVIDIA GeForce RTX 3080
  Memory: 10.00 GB
  Compute Capability: 8.6

✅ Sử dụng GPU: NVIDIA GeForce RTX 3080
================================================================================
```

Trong quá trình training, sẽ hiển thị GPU memory:
```
🎮 GPU Memory Status:
  • Allocated: 1234.56 MB
  • Cached: 2345.67 MB
  • Max Allocated: 3456.78 MB
```

---

## 📂 Tự động phát hiện tham số

Script sẽ tự động phát hiện từ data:

```
📂 TỰ ĐỘNG PHÁT HIỆN THAM SỐ DỮ LIỆU
================================================================================
→ Đường dẫn dữ liệu: ./data/federated_splits
→ Chế độ: Non-IID Safe (quét tất cả 5 clients)

✅ Từ client 0:
  • INPUT_FEATURES = 46
  • INPUT_SHAPE = (46,)

📊 Đang quét 5 clients:
  • Client 0: 10000 train, 3000 test, 2 unique labels
  • Client 1: 10000 train, 3000 test, 2 unique labels
  ...

✅ Tổng hợp:
  • NUM_CLASSES = 2
  • INPUT_FEATURES = 46
  • INPUT_SHAPE = (46,)
  • Tổng train samples = 50,000
  • Tổng test samples = 15,000
================================================================================
```

---

## 🛠️ Tùy chỉnh Training

### **Thử nghiệm với FedProx** (tốt hơn cho Non-IID data)

```python
CONFIG = {
    'algorithm': 'fedprox',
    'mu': 0.01,  # Thử 0.001, 0.01, 0.1
    ...
}
```

### **Giảm GPU memory usage**

```python
CONFIG = {
    'batch_size': 32,        # Giảm từ 64 xuống 32
    'client_fraction': 0.5,  # Chỉ train 50% clients mỗi round
    ...
}
```

### **Training nhanh hơn**

```python
CONFIG = {
    'num_rounds': 20,        # Giảm số rounds
    'local_epochs': 3,       # Giảm local epochs
    'eval_every': 5,         # Evaluate ít hơn
    ...
}
```

### **Cho phép chạy trên CPU** (nếu không có GPU)

```python
CONFIG = {
    'force_gpu': False,      # ⚠️  Chậm hơn rất nhiều!
    ...
}
```

---

## 🔥 Nếu chưa có data

Chạy từng bước để chuẩn bị data:

```bash
# Bước 1: Tiền xử lý data thô từ CSV
python step1_prepare_chunks.py

# Bước 2: Tạo federated splits (Non-IID)
python step2_create_federated_splits.py

# Bước 3: Train
python run_federated_training.py
```

---

## 🧪 Test model sau khi train

```python
import torch
from model import build_cnn_gru_model
import numpy as np

# Load model
input_shape = (46,)  # Thay bằng số features của bạn
num_classes = 2

model = build_cnn_gru_model(input_shape, num_classes)
model.load_state_dict(torch.load('./results/global_model.pth'))
model.eval()

# Test với data mới
X_test = np.random.randn(10, 46)  # 10 samples, 46 features
X_test_tensor = torch.from_numpy(X_test).float()

# Predict
with torch.no_grad():
    output = model(X_test_tensor)
    predictions = output.argmax(dim=1)
    probabilities = torch.softmax(output, dim=1)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

---

## ❌ Troubleshooting

### **Lỗi: "Client data not found"**
→ Chạy `python step2_create_federated_splits.py` trước

### **Lỗi: "GPU required but not available"**
→ Sửa `force_gpu = False` trong CONFIG hoặc cài đặt CUDA PyTorch

### **Lỗi: "CUDA out of memory"**
→ Giảm `batch_size` xuống 32 hoặc 16
→ Hoặc giảm `client_fraction`

### **Model accuracy quá thấp**
→ Tăng `num_rounds` (ví dụ: 100)
→ Tăng `learning_rate` (ví dụ: 0.01)
→ Thử `algorithm = 'fedprox'` với `mu = 0.01`

### **Training quá chậm**
→ Kiểm tra có đang dùng GPU không (xem log)
→ Giảm `local_epochs` xuống 3
→ Tăng `eval_every` lên 5 hoặc 10

---

## 📈 Kết quả mong đợi

Với data IoT và model CNN-GRU:
- **Training time**: ~5-30 phút (tùy GPU và số rounds)
- **Accuracy**: 85-95% (tùy data quality)
- **GPU memory**: ~1-3 GB

---

Xong! 🎉 Chỉ cần chạy `python run_federated_training.py`
