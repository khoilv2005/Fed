# 🚀 Hướng dẫn chạy Federated Learning với Multiprocessing

## ⚠️ VẤN ĐỀ QUAN TRỌNG: Jupyter Notebook vs Python Script

### Tại sao không nên chạy trong Jupyter Notebook?

Khi sử dụng **PyTorch CUDA với multiprocessing**, có 2 vấn đề chính trong Jupyter notebook:

1. **CUDA fork issue**:
   - CUDA không hỗ trợ `fork()` method sau khi đã khởi tạo
   - Error: `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

2. **Pickle issue với spawn**:
   - Spawn method cần pickle functions
   - Functions trong notebook không thể pickle được
   - Error: `AttributeError: Can't get attribute '_client_training_worker'`

### ✅ GIẢI PHÁP: Chạy như Python script

```bash
# Thay vì chạy trong notebook, chạy trực tiếp từ terminal:
python run_federated_training.py
```

## 📝 Hướng dẫn sử dụng

### Option 1: Chạy trực tiếp (KHUYẾN NGHỊ)

```bash
# Di chuyển đến thư mục chứa script
cd /path/to/Fed

# Chạy script
python run_federated_training.py
```

### Option 2: Chạy trong Google Colab

Nếu bắt buộc phải dùng Colab, có thể chạy cell với magic command:

```python
# Trong Colab cell
!python run_federated_training.py
```

### Option 3: Tắt multiprocessing trong notebook

Nếu muốn chạy trong notebook, tắt multiprocessing trong CONFIG:

```python
CONFIG = {
    # ...
    'use_multiprocessing': False,  # Tắt multiprocessing
    'num_processes': 1,
    # ...
}
```

**Lưu ý**: Tắt multiprocessing sẽ chậm hơn đáng kể (5x - 10x).

## ⚙️ Cấu hình Multiprocessing

### Các thông số quan trọng

```python
CONFIG = {
    'use_multiprocessing': True,   # Bật/tắt multiprocessing
    'num_processes': 5,            # Số processes chạy song song
}
```

### Khuyến nghị cho `num_processes`

| Hệ thống | Khuyến nghị | Lý do |
|----------|------------|-------|
| **1 GPU** | 2-3 processes | Tránh OOM (Out Of Memory) |
| **2 GPUs** | 4 processes | 2 processes/GPU |
| **4+ GPUs** | = num_clients | Tận dụng tối đa GPUs |
| **CPU only** | 4-8 processes | = số CPU cores |

### Ví dụ cấu hình

```python
# Với 1 GPU (15GB VRAM)
CONFIG = {
    'use_multiprocessing': True,
    'num_processes': 3,      # An toàn với 1 GPU
    'batch_size': 1024,      # Có thể giảm nếu OOM
}

# Với 2 GPUs
CONFIG = {
    'use_multiprocessing': True,
    'num_processes': 4,      # 2 processes/GPU
    'batch_size': 1024,
}

# Với nhiều GPUs (8 GPUs)
CONFIG = {
    'use_multiprocessing': True,
    'num_processes': 5,      # = num_clients
    'batch_size': 2048,      # Có thể tăng batch size
}
```

## 🐛 Troubleshooting

### Lỗi: "RuntimeError: Cannot re-initialize CUDA in forked subprocess"

**Nguyên nhân**: Đang dùng `fork` method với CUDA

**Giải pháp**:
1. Chạy script như .py file (không phải notebook)
2. Script sẽ tự động dùng `spawn` method

### Lỗi: "AttributeError: Can't get attribute '_client_training_worker'"

**Nguyên nhân**: Đang chạy trong Jupyter notebook với spawn method

**Giải pháp**:
```bash
# Chạy từ terminal thay vì notebook
python run_federated_training.py
```

### Lỗi: "RuntimeError: CUDA out of memory"

**Nguyên nhân**: Quá nhiều processes hoặc batch size quá lớn

**Giải pháp**:
```python
# Giảm số processes
'num_processes': 2,  # Thay vì 5

# HOẶC giảm batch size
'batch_size': 512,   # Thay vì 1024
```

### Lỗi: "Tất cả clients đều thất bại!"

**Nguyên nhân**: Workers không thể khởi động hoặc crash

**Cách debug**:
1. Xem error log chi tiết trong output
2. Kiểm tra VRAM: `nvidia-smi`
3. Giảm `num_processes` xuống 1 để test
4. Kiểm tra đường dẫn dữ liệu trong CONFIG

## 📊 Hiệu suất

### So sánh Sequential vs Multiprocessing

| Method | Time (5 clients, 10 rounds) | Speedup |
|--------|----------------------------|---------|
| Sequential (1 process) | ~50 phút | 1x |
| Multiprocessing (3 processes) | ~15 phút | **3.3x** |
| Multiprocessing (5 processes) | ~10 phút | **5x** |

**Lưu ý**: Speedup thực tế phụ thuộc vào:
- Số GPUs khả dụng
- VRAM của mỗi GPU
- Kích thước model và dataset

## 📈 Monitoring

### Theo dõi quá trình training

Script tự động hiển thị:

```
🔄 Clients Training (Parallel): 100% 5/5 [00:23<00:00, 4.30s/client]
   🚀 Worker cho Client 0 đã start (device: 0)
   🚀 Worker cho Client 1 đã start (device: 1)
   ✓ Client 0 hoàn thành - Loss: 0.3245
   ✓ Client 1 hoàn thành - Loss: 0.3156
```

### Theo dõi GPU usage

```bash
# Terminal khác
watch -n 1 nvidia-smi
```

Bạn sẽ thấy multiple processes sử dụng GPU đồng thời.

## 🎯 Best Practices

1. **Luôn chạy script từ terminal** khi dùng multiprocessing + CUDA
2. **Test với num_processes=1** trước để đảm bảo code chạy đúng
3. **Monitor VRAM** khi tăng num_processes
4. **Backup dữ liệu** trước khi chạy training dài
5. **Dùng tmux/screen** để tránh mất kết nối SSH

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. CUDA version: `nvcc --version`
3. GPU memory: `nvidia-smi`
4. Python version: `python --version` (khuyến nghị: 3.8+)

---

**Tóm tắt**: Để sử dụng multiprocessing với CUDA hiệu quả, hãy chạy script từ terminal:
```bash
python run_federated_training.py
```
