# 🎯 Configuration Guide for Kaggle_build.py

## 📝 Easy Configuration Options

### 🎯 **DEFAULT_CONFIG Section**
Located at the top of `Kaggle_build.py`:

```python
DEFAULT_CONFIG = {
    # Data paths
    'input_dir': './data/federated_splits',    # ← Thay đổi input directory
    'output_dir': './kaggle_results',         # ← Thay đổi output directory
    
    # Federated learning parameters
    'num_clients': 5,
    'num_rounds': 10,
    'local_epochs': 5,
    'algorithm': 'fedprox',  # 'fedavg' or 'fedprox'
    'method': 'auto',       # 'sequential', 'joblib', or 'auto'
    
    # Training parameters
    'base_learning_rate': 0.001,
    'mu': 0.01,           # FedProx proximal term
}
```

## 🚀 **Usage Methods**

### **Method 1: Edit DEFAULT_CONFIG (Easiest)**
```python
# Chỉnh sửa trực tiếp trong file
DEFAULT_CONFIG['input_dir'] = '/path/to/your/data'
DEFAULT_CONFIG['num_clients'] = 10
DEFAULT_CONFIG['num_rounds'] = 20

# Run với default values
from Kaggle_build import run_kaggle_optimized_federated_learning
results = run_kaggle_optimized_federated_learning()
```

### **Method 2: Override Parameters**
```python
from Kaggle_build import run_kaggle_optimized_federated_learning

# Override specific parameters
results = run_kaggle_optimized_federated_learning(
    data_dir='/path/to/your/data',  # Override input_dir
    num_clients=10,                 # Override num_clients
    num_rounds=20,                  # Override num_rounds
    # Uses DEFAULT_CONFIG for other parameters
)
```

### **Method 3: Complete Custom Configuration**
```python
from Kaggle_build import run_kaggle_optimized_federated_learning

# Custom all parameters
results = run_kaggle_optimized_federated_learning(
    data_dir='/path/to/your/data',
    num_clients=10,
    num_rounds=20,
    local_epochs=10,
    algorithm='fedavg',
    method='joblib'
)
```

## 📁 **Directory Structure Examples**

### **Kaggle/Colab Structure:**
```
/content/
├── Kaggle_build.py
├── data/
│   └── federated_splits/
│       ├── client_0_data.npz
│       ├── client_1_data.npz
│       └── ...
└── kaggle_results/
    └── kaggle_optimized_20251115_214500.json
```

### **Local Structure:**
```
d:/Project/ai4fids_project/
├── Kaggle_build.py
├── data/
│   └── federated_splits/
│       ├── client_0_data.npz
│       └── ...
└── results/
    └── kaggle_results/
        └── kaggle_optimized_20251115_214500.json
```

## ⚙️ **Parameter Explanations**

### **Data Paths:**
- `input_dir`: Directory containing client_*.npz files
- `output_dir`: Directory for saving results

### **Federated Learning:**
- `num_clients`: Number of federated clients
- `num_rounds`: Number of communication rounds
- `local_epochs`: Local training epochs per client
- `algorithm`: 'fedavg' or 'fedprox'
- `method`: 'sequential', 'joblib', or 'auto'

### **Training:**
- `base_learning_rate`: Base learning rate (will be scaled by GPU)
- `mu`: FedProx proximal term coefficient

## 🎯 **Quick Start Examples**

### **Example 1: Kaggle Free Tier (T4)**
```python
# Edit DEFAULT_CONFIG
DEFAULT_CONFIG['input_dir'] = '/kaggle/input/your-dataset'
DEFAULT_CONFIG['num_clients'] = 3  # Smaller for free tier
DEFAULT_CONFIG['num_rounds'] = 5   # Faster training

# Run
results = run_kaggle_optimized_federated_learning()
```

### **Example 2: Colab Pro (A100)**
```python
# Edit DEFAULT_CONFIG
DEFAULT_CONFIG['input_dir'] = '/content/drive/MyDrive/federated_data'
DEFAULT_CONFIG['num_clients'] = 10  # Larger for Pro
DEFAULT_CONFIG['num_rounds'] = 20   # More rounds

# Run
results = run_kaggle_optimized_federated_learning()
```

### **Example 3: Local Development**
```python
# Edit DEFAULT_CONFIG
DEFAULT_CONFIG['input_dir'] = './data/federated_splits'
DEFAULT_CONFIG['output_dir'] = './results'
DEFAULT_CONFIG['num_clients'] = 5
DEFAULT_CONFIG['method'] = 'sequential'  # More stable locally

# Run
results = run_kaggle_optimized_federated_learning()
```

## 🔧 **Advanced Configuration**

### **GPU-Specific Settings:**
The system automatically detects GPU and optimizes:
- **T4**: Mixed precision enabled, batch_size=384
- **A100**: Mixed precision enabled, batch_size=1024
- **L4**: Mixed precision enabled, batch_size=512
- **P100**: Mixed precision disabled, batch_size=256

### **Method Selection:**
- `auto`: Automatically chooses best method
- `sequential`: Always stable, compatible everywhere
- `joblib`: Parallel processing (Colab recommended)

## 📊 **Output Files**

Results are saved to `{output_dir}/kaggle_optimized_{timestamp}.json` containing:
- GPU information and optimization config
- Training configuration
- Round-by-round results
- Performance metrics
- Final summary

## 🚨 **Common Issues & Solutions**

### **Issue: "Client data not found"**
```python
# Check your input directory structure
DEFAULT_CONFIG['input_dir'] = '/correct/path/to/data'
```

### **Issue: "Out of memory"**
```python
# Reduce batch size (system will auto-adjust if needed)
# Or use sequential method
DEFAULT_CONFIG['method'] = 'sequential'
```

### **Issue: "Joblib not available"**
```python
# Install joblib or use sequential
!pip install joblib
# Or
DEFAULT_CONFIG['method'] = 'sequential'
```

## 🎉 **Success Tips**

1. **Start with DEFAULT_CONFIG** - it's optimized for most cases
2. **Check GPU detection** - system will print GPU info
3. **Monitor performance** - results show speed and efficiency
4. **Use auto method** - lets system choose optimal approach
5. **Save results** - automatic timestamping prevents overwrites

Happy Federated Learning! 🚀
