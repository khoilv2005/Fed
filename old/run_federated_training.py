import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import pickle
import json
import copy
from collections import OrderedDict
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm.auto import tqdm
import time

# === THÊM THƯ VIỆN CHO BIỂU ĐỒ VÀ KẾT QUẢ ===
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support
)
import pandas as pd
# ============================================

# Import từ 2 file .py kia
try:
    from old.Fed import FederatedServer, FederatedClient
    from old.model import CNN_GRU_Model, build_cnn_gru_model
except ImportError as e:
    print("LỖI: Không tìm thấy 'Fed.py' hoặc 'model.py'.")
    print(f"Chi tiết: {e}")
    print("Hãy đảm bảo cả 3 file (.py) đều nằm trong cùng một thư mục.")
    # Định nghĩa tạm thời để Colab không báo lỗi ngay
    class FederatedClient: pass
    class FederatedServer: pass
    class CNN_GRU_Model(nn.Module): pass
    def build_cnn_gru_model(shape, classes): pass
    # Code sẽ crash sau đó, nhưng đây là để debug import
    
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 💡 BƯỚC 1: CẤU HÌNH CHÍNH 💡
# ============================================================================

CONFIG = {
    # ⬇️ CHỈNH ĐƯỜNG DẪN NÀY (TRÊN COLAB CẦN MOUNT DRIVE TRƯỚC) ⬇️
    'data_dir': '/content/drive/MyDrive/Fed-Data/5-Client',
    'output_dir': '/content/drive/MyDrive/Fed-Data/5-Client/Results', # Lưu kết quả
    
    'num_clients': 5,

    # Model params (sẽ được tự động phát hiện từ data)
    'input_shape': None,  # Tự động phát hiện
    'num_classes': None,  # Tự động phát hiện

    # Training params
    'algorithm': 'fedprox',     # 'fedavg' hoặc 'fedprox'
    'num_rounds': 4,            # 4 Vòng
    'local_epochs': 1,          # 1 Epoch/Vòng
    'learning_rate': 0.001,
    'batch_size': 2048,         # Batch size lớn (GPU T4 15GB chạy tốt)
    'client_fraction': 1.0,     # Tỉ lệ clients tham gia mỗi round

    # FedProx specific
    'mu': 0.01,  # Proximal term coefficient

    # Device - Luôn ưu tiên GPU
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'force_gpu': True,  # Set False nếu muốn cho phép chạy trên CPU

    # Multiprocessing
    'use_multiprocessing': True,  # ✅ Đặt True để chạy song song
    'num_processes': 2,           # Số processes (Colab free chỉ có 2 CPU cores)

    # Visualization
    'eval_every': 1,  # Đánh giá sau mỗi round
}

# === TẠO THƯ MỤC OUTPUT ===
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(CONFIG['output_dir'], f"run_{TIMESTAMP}_{CONFIG['algorithm']}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG['output_dir'] = OUTPUT_DIR # Cập nhật config với đường dẫn mới

# ============================================================================
# 💡 BƯỚC 2: HÀM KIỂM TRA GPU 💡
# ============================================================================

def check_and_setup_gpu(config: Dict) -> str:
    """
    Kiểm tra GPU và trả về torch.device phù hợp.
    """
    logger.info("\n" + "="*80)
    logger.info("🔧 KIỂM TRA THIẾT BỊ GPU/CPU")
    logger.info("="*80)

    cuda_available = torch.cuda.is_available()
    logger.info(f"   - CUDA khả dụng: {cuda_available}")

    if cuda_available:
        device = torch.device('cuda')
        logger.info(f"   - Phiên bản CUDA: {torch.version.cuda}")
        logger.info(f"   - Số lượng GPU: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"\n   ➤ GPU {i}:")
            logger.info(f"      - Tên: {torch.cuda.get_device_name(i)}")
            logger.info(f"      - Bộ nhớ: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"      - Compute capability: {props.major}.{props.minor}")
        
        logger.info(f"\n✅ Sử dụng thiết bị: {torch.cuda.get_device_name(0)}")
    else:
        if config.get('force_gpu', False):
            logger.error("\n❌ LỖI: Không phát hiện GPU nhưng force_gpu=True")
            raise RuntimeError("Yêu cầu GPU nhưng không có GPU khả dụng.")
        else:
            device = torch.device('cpu')
            logger.warning(f"\n⚠️ Cảnh báo: Không có GPU, hệ thống sẽ chạy trên CPU (chậm hơn).")

    logger.info("="*80)
    # Trả về string, không phải torch.device, để dễ dàng serialize
    return 'cuda' if cuda_available else 'cpu'

# ============================================================================
# 💡 BƯỚC 3: HÀM TỰ ĐỘNG PHÁT HIỆN THAM SỐ DỮ LIỆU 💡
# ============================================================================

def auto_detect_data_parameters(data_dir, num_clients):
    """
    Tự động phát hiện input_shape và num_classes.
    """
    logger.info("\n" + "="*80)
    logger.info("📂 TỰ ĐỘNG PHÁT HIỆN THAM SỐ DỮ LIỆU")
    logger.info("="*80)
    logger.info(f"→ Thư mục dữ liệu: {data_dir}")
    logger.info(f"→ Số lượng client (dự kiến): {num_clients}")

    try:
        all_labels = []
        data_stats = {}

        # Lấy kích thước input từ client_0
        client_0_path = os.path.join(data_dir, "client_0_data.npz")
        if not os.path.exists(client_0_path):
            raise FileNotFoundError(f"Không tìm thấy file: {client_0_path}")

        with np.load(client_0_path) as data:
            x_train_sample = data['X_train']
            input_features = x_train_sample.shape[1]
            input_shape = (input_features,)
            logger.info(f"\n✅ Thông tin từ client 0:")
            logger.info(f"   - Số đặc trưng (INPUT_FEATURES): {input_features}")
            logger.info(f"   - input_shape: {input_shape}")

        logger.info(f"\n→ Đang quét dữ liệu của {num_clients} client để thống kê nhãn...")
        total_train = 0
        total_test = 0

        for i in range(num_clients):
            file_path = os.path.join(data_dir, f"client_{i}_data.npz")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

            with np.load(file_path) as data:
                x_train = data['X_train']; y_train = data['y_train']
                x_test = data['X_test']; y_test = data['y_test']

                all_labels.append(y_train)
                all_labels.append(y_test) # Quét cả nhãn test
                
                unique_labels, counts = np.unique(y_train, return_counts=True)
                total_train += len(x_train)
                total_test += len(x_test)

                data_stats[i] = {
                    'train_samples': int(len(x_train)),
                    'test_samples': int(len(x_test)),
                    'unique_labels': int(len(unique_labels)),
                    'label_distribution': {str(k): int(v) for k, v in zip(unique_labels, counts)}
                }
                logger.info(f"   - Client {i}: {len(x_train):,} train, {len(x_test):,} test, {len(unique_labels)} nhãn")

        combined_labels = np.concatenate(all_labels)
        num_classes = len(np.unique(combined_labels))

        logger.info("\n📊 Tổng hợp toàn bộ dữ liệu:")
        logger.info(f"   - Số lớp (num_classes): {num_classes}")
        logger.info(f"   - Tổng số mẫu train: {total_train:,}")
        logger.info(f"   - Tổng số mẫu test:  {total_test:,}")
        logger.info("="*80)

        return input_shape, num_classes, data_stats

    except FileNotFoundError as e:
        logger.error("\n" + "="*80 + f"\n❌ LỖI: KHÔNG TÌM THẤY TỆP DỮ LIỆU\nĐường dẫn: {e.filename}\n" + "="*80)
        raise
    except KeyError as e:
        logger.error("\n" + "="*80 + f"\n❌ LỖI: THIẾU KEY TRONG FILE .NPZ\nKey: {e}\n" + "="*80)
        raise

# ============================================================================
# 💡 BƯỚC 4: HÀM LOAD DỮ LIỆU 💡
# ============================================================================

class NumpyDataset(TensorDataset):
    """Dataset tiện dụng để wrap numpy array thành TensorDataset."""
    def __init__(self, X, y):
        if len(X.shape) == 3:
            X = X.squeeze(-1)  # (N, F, 1) -> (N, F)

        X = X.astype(np.float32)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y).long()
        super().__init__(X_tensor, y_tensor)


def load_federated_data(data_dir, num_clients, batch_size, device='cpu'):
    """
    Load dữ liệu federated cho tất cả client.
    """
    logger.info("\n" + "="*80)
    logger.info("📥 LOADING FEDERATED DATA")
    logger.info("="*80)
    logger.info(f"→ Thiết bị hiện dùng: {device}")
    logger.info(f"→ Số lượng client: {num_clients}\n")

    train_loaders = []
    test_loaders = []

    for client_id in range(num_clients):
        data_path = os.path.join(data_dir, f'client_{client_id}_data.npz')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy dữ liệu của client {client_id} tại: {data_path}")

        data = np.load(data_path)
        X_train = data['X_train']; y_train = data['y_train']
        X_test = data['X_test']; y_test = data['y_test']

        logger.info(f"   - Client {client_id}: X_train {X_train.shape}, X_test {X_test.shape}")

        train_dataset = NumpyDataset(X_train, y_train)
        test_dataset = NumpyDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size*2, shuffle=False, drop_last=False # Batch size test lớn hơn
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    logger.info(f"\n✅ Đã load dữ liệu cho {num_clients} client.")
    logger.info("="*80)

    return train_loaders, test_loaders

# ============================================================================
# 💡 BƯỚC 5: HÀM KHỞI TẠO HỆ THỐNG 💡
# ============================================================================

def initialize_federated_system(
    config: Dict,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader]
):
    """
    Khởi tạo global model, clients, và server (đã sửa lỗi RAM)
    """
    logger.info("\n" + "="*80)
    logger.info("🏗️  INITIALIZING FEDERATED SYSTEM")
    logger.info("="*80)
    
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    device = config['device']
    num_clients = config['num_clients']

    # Tạo global model
    logger.info(f"\n→ Khởi tạo mô hình toàn cục (global model)...")
    logger.info(f"   - Input shape: {input_shape}")
    logger.info(f"   - Số lớp: {num_classes}")
    logger.info(f"   - Thiết bị: {device}")

    global_model = build_cnn_gru_model(input_shape, num_classes)
    global_model = global_model.to(device)
    logger.info(f"   - Mô hình đã được chuyển sang thiết bị: {device}")

    if device == 'cuda':
        logger.info(f"   - Xác nhận tham số đầu tiên của mô hình đang ở: {next(global_model.parameters()).device}")

    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    logger.info(f"   - Tổng số tham số: {total_params:,}")
    logger.info(f"   - Số tham số trainable: {trainable_params:,}")

    # Tạo clients (Chỉ cần thiết cho chế độ Tuần Tự)
    clients = []
    if not config['use_multiprocessing']:
        logger.info(f"\n→ Khởi tạo {num_clients} client (chế độ tuần tự)...")
        for client_id in range(num_clients):
            client_model = CNN_GRU_Model(input_shape, num_classes)
            client_model.load_state_dict(global_model.state_dict())

            client = FederatedClient(
                client_id=client_id,
                model=client_model,
                train_loader=train_loaders[client_id],
                test_loader=test_loaders[client_id],
                device=device
            )
            clients.append(client)
            logger.info(f"   - Client {client_id}: khởi tạo thành công trên thiết bị {device}")
    else:
        logger.info(f"\n→ Bỏ qua khởi tạo Client (chế độ song song)...")
        
    
    logger.info("\n→ Gán danh sách test loader cho server (tránh lỗi RAM)...")
    
    # Tạo server
    logger.info("\n→ Khởi tạo server...")
    server = FederatedServer(
        model=global_model,
        clients=clients, # List này sẽ rỗng nếu dùng multiprocessing
        client_test_loaders=test_loaders, # ✅ SỬA: Dùng list test loader
        device=device
    )

    logger.info("\n✅ Hệ thống Federated đã được khởi tạo hoàn chỉnh.")
    logger.info("="*80)

    return server, clients

# ============================================================================
# 💡 BƯỚC 6: CÁC HÀM HỖ TRỢ MULTIPROCESSING 💡
# ============================================================================

# (Cần import lại bên trong worker)
def _client_training_worker(args_tuple):
    """
    Hàm worker (helper) để chạy trong một process riêng biệt.
    """
    # Import lại các thư viện cần thiết cho process này
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from collections import OrderedDict
    from tqdm.auto import tqdm as _tqdm
    import time as _time
    import os # Thêm os
    
    # === BẮT ĐẦU ĐỊNH NGHĨA LẠI (Bắt buộc cho 'spawn') ===
    # (Đây là bản copy-paste của class model)
    class CNN_GRU_Model_Worker(nn.Module):
        def __init__(self, input_shape, num_classes=2):
            super(CNN_GRU_Model_Worker, self).__init__()
            if isinstance(input_shape, tuple): seq_length = input_shape[0]
            else: seq_length = input_shape
            self.input_shape = input_shape; self.num_classes = num_classes
            def conv_output_shape(L_in, kernel_size=1, stride=1, padding=0, dilation=1):
                if padding == 1 and kernel_size == 3: L_out_conv = L_in
                else: L_out_conv = (L_in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
                return L_out_conv
            def pool_output_shape(L_in, kernel_size=2, stride=2, padding=0, dilation=1): return (L_in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
            self.conv1 = nn.Conv1d(1, 64, 3, padding=1); self.bn1 = nn.BatchNorm1d(64); self.pool1 = nn.MaxPool1d(2, stride=2); self.dropout_cnn1 = nn.Dropout(0.2)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1); self.bn2 = nn.BatchNorm1d(128); self.pool2 = nn.MaxPool1d(2, stride=2); self.dropout_cnn2 = nn.Dropout(0.2)
            self.conv3 = nn.Conv1d(128, 256, 3, padding=1); self.bn3 = nn.BatchNorm1d(256); self.pool3 = nn.MaxPool1d(2, stride=2); self.dropout_cnn3 = nn.Dropout(0.3)
            cnn_output_length = seq_length
            cnn_output_length = pool_output_shape(conv_output_shape(cnn_output_length, kernel_size=3, padding=1))
            cnn_output_length = pool_output_shape(conv_output_shape(cnn_output_length, kernel_size=3, padding=1))
            cnn_output_length = pool_output_shape(conv_output_shape(cnn_output_length, kernel_size=3, padding=1))
            self.cnn_output_size = 256 * cnn_output_length
            self.gru1 = nn.GRU(1, 128, batch_first=True); self.gru2 = nn.GRU(128, 64, batch_first=True); self.gru_output_size = 64
            concat_size = self.cnn_output_size + self.gru_output_size
            self.dense1 = nn.Linear(concat_size, 256); self.bn_mlp1 = nn.BatchNorm1d(256); self.dropout1 = nn.Dropout(0.4)
            self.dense2 = nn.Linear(256, 128); self.bn_mlp2 = nn.BatchNorm1d(128); self.dropout2 = nn.Dropout(0.3)
            self.output = nn.Linear(128, num_classes); self.relu = nn.ReLU()
        
        def forward(self, x):
            if len(x.shape) == 2: x = x.unsqueeze(-1)
            batch_size = x.size(0); x_cnn = x.permute(0, 2, 1)
            x_cnn = self.dropout_cnn1(self.pool1(self.relu(self.bn1(self.conv1(x_cnn)))))
            x_cnn = self.dropout_cnn2(self.pool2(self.relu(self.bn2(self.conv2(x_cnn)))))
            x_cnn = self.dropout_cnn3(self.pool3(self.relu(self.bn3(self.conv3(x_cnn)))))
            cnn_output = x_cnn.view(batch_size, -1); x_gru = x; x_gru, _ = self.gru1(x_gru); x_gru, _ = self.gru2(x_gru)
            gru_output = x_gru[:, -1, :]; concatenated = torch.cat([cnn_output, gru_output], dim=1)
            x = self.dense1(concatenated); 
            if x.shape[0] > 1: x = self.bn_mlp1(x)
            x = self.relu(x); x = self.dropout1(x)
            x = self.dense2(x); 
            if x.shape[0] > 1: x = self.bn_mlp2(x)
            x = self.relu(x); x = self.dropout2(x)
            return self.output(x)

    class NumpyDataset_Worker(TensorDataset):
        def __init__(self, X, y):
            if len(X.shape) == 3: X = X.squeeze(-1)
            X = X.astype(np.float32)
            X_tensor = torch.from_numpy(X)
            y_tensor = torch.from_numpy(y).long()
            super().__init__(X_tensor, y_tensor)
    # === KẾT THÚC ĐỊNH NGHĨA LẠI ===
    
    try:
        # Giải nén arguments
        (client_id, model_state_dict, client_data_path, device_id, 
         config) = args_tuple
        
        # Lấy config
        num_epochs = config['local_epochs']
        learning_rate = config['learning_rate']
        algorithm = config['algorithm']
        mu = config['mu']
        batch_size = config['batch_size']
        
        # Set device
        if device_id != 'cpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')
        
        # ✅ SỬA LỖI RAM: Worker tự load data
        with np.load(client_data_path) as data:
            X_train = data['X_train']
            y_train = data['y_train']
            
        # Tạo DataLoader
        train_dataset = NumpyDataset_Worker(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        
        # Tạo model và load state
        model = CNN_GRU_Model_Worker(config['input_shape'], config['num_classes'])
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        
        # Huấn luyện
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Dùng Adam
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            # Đặt position=client_id để các thanh progress bar xếp chồng lên nhau
            pbar = _tqdm(
                train_loader,
                desc=f"[Worker Client {client_id} (GPU {device_id})] Epoch {epoch+1}/{num_epochs}",
                unit="batch",
                leave=False,
                position=client_id 
            )

            for data, target in pbar:
                batch_start = _time.time()

                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                ce_loss = criterion(output, target)
                
                # FedProx proximal term
                if algorithm == 'fedprox' and model_state_dict is not None:
                    proximal_term = 0.0
                    # ✅ SỬA LỖI: Dùng named_parameters và so sánh với state_dict
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            global_param = model_state_dict[name].to(device)
                            proximal_term += torch.sum((param - global_param) ** 2)
                    loss = ce_loss + (mu / 2) * proximal_term
                    prox_val = float(((mu / 2) * proximal_term).detach().cpu().item())
                else:
                    loss = ce_loss
                    prox_val = 0.0
                
                loss.backward()
                optimizer.step()
                
                batch_time = _time.time() - batch_start

                epoch_loss += ce_loss.item() * data.size(0)
                epoch_samples += data.size(0)

                pbar.set_postfix({
                    "ce": f"{ce_loss.item():.4f}",
                    "prox": f"{prox_val:.2e}",
                    "bt": f"{batch_time*1000:.0f}ms"
                })
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        avg_loss = total_loss / max(1, total_samples)
        
        # Trả về kết quả (chuyển params về CPU)
        return {
            'client_id': client_id,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'num_samples': len(X_train),
            'loss': avg_loss
        }
    
    except Exception as e:
        print(f"❌ Lỗi trong worker client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_models_fedavg_parallel(client_results: List[Dict]) -> OrderedDict:
    """
    ✅ ĐÃ SỬA LỖI: FedAvg aggregation từ client results (Fix lỗi Dtype)
    (Chạy trên CPU)
    """
    total_samples = sum(r['num_samples'] for r in client_results)
    
    aggregated_params = OrderedDict()
    first_state = client_results[0]['model_state_dict']
    
    for key in first_state.keys():
        aggregated_params[key] = torch.zeros_like(first_state[key])
    
    for result in client_results:
        weight = result['num_samples'] / max(1, total_samples)
        state_dict = result['model_state_dict']
        
        for key in aggregated_params.keys():
            param = state_dict[key]
            
            # Chỉ aggregate các tham số float
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                weight_tensor = torch.tensor(weight, dtype=param.dtype, device=param.device)
                aggregated_params[key] += weight_tensor * param
            else:
                # Giữ lại giá trị của client đầu tiên (ví dụ: num_batches_tracked)
                if result['client_id'] == client_results[0]['client_id']:
                    aggregated_params[key] = param
    
    return aggregated_params


def train_round_multiprocessing(
    server, # Dùng để set params
    config,
    # train_loaders, # KHÔNG DÙNG NỮA
    device='cuda'
):
    """
    ✅ ĐÃ SỬA LỖI: Train 1 round với multiprocessing (Fix lỗi RAM OOM + TypeError)
    """
    # Lấy global model state (trên CPU)
    global_state_dict = {k: v.cpu() for k, v in server.get_global_params().items()}
    
    # Phân bổ GPU (nếu có)
    if device == 'cuda' and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_ids = [i % num_gpus for i in range(config['num_clients'])]
        if num_gpus > 1:
            logger.info(f"   • Phân bổ {config['num_clients']} clients cho {num_gpus} GPUs (round-robin).")
            logger.info(f"   • GPU mapping: {device_ids}")
        else:
            logger.info(f"   • Tất cả {config['num_clients']} clients sẽ dùng chung 1 GPU (cuda:0).")
    else:
        # Nếu chỉ có 1 GPU hoặc CPU
        device_ids = ['cpu'] * config['num_clients']
        logger.info(f"   • Tất cả {config['num_clients']} clients sẽ chạy trên CPU.")

    
    # Chuẩn bị arguments cho mỗi client
    args_list = []
    for client_id in range(config['num_clients']):
        client_data_path = os.path.join(config['data_dir'], f"client_{client_id}_data.npz")
        
        args_list.append(
            (
                client_id,
                global_state_dict,
                client_data_path, # ✅ SỬA LỖI RAM: Chỉ truyền đường dẫn
                device_ids[client_id],
                config # Truyền toàn bộ config
            )
        )
    
    # Train song song
    logger.info(f"   • Bắt đầu train {config['num_clients']} clients song song với {config['num_processes']} processes...")
    
    mp_context = mp.get_context('spawn') # 'spawn' là bắt buộc cho CUDA
    logger.info(f"   • Đang khởi tạo process pool...")
    
    results = []
    # Dùng pool.map để chạy song song
    with mp_context.Pool(processes=config['num_processes']) as pool:
        logger.info(f"   • Pool đã được tạo, bắt đầu submit {len(args_list)} tasks...")
        
        # Dùng imap_unordered để có progress bar
        pbar = tqdm(
            pool.imap_unordered(_client_training_worker, args_list),
            total=len(args_list),
            desc="🔄 Clients Training (Parallel)",
            unit="client"
        )
        for res in pbar:
            results.append(res)
    
    logger.info(f"\n   • Đã nhận kết quả từ {len(results)} clients.")
    
    # Lọc các kết quả lỗi (None)
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        raise RuntimeError("Tất cả clients đều thất bại!")
    
    # Aggregate models (trên CPU)
    logger.info(f"   • Đang aggregate models từ {len(results)} clients...")
    aggregated_params_cpu = aggregate_models_fedavg_parallel(results)
    
    # Cập nhật global model (chuyển params về lại GPU)
    aggregated_params_gpu = OrderedDict(
        (k, v.to(device)) for k, v in aggregated_params_cpu.items()
    )
    server.set_global_params(aggregated_params_gpu)
    
    # Tính avg loss
    avg_loss = np.mean([r['loss'] for r in results])
    
    return {
        'train_loss': avg_loss,
        'num_clients': len(results)
    }

# ============================================================================
# 💡 BƯỚC 9: HÀM HUẤN LUYỆN CHÍNH 💡
# ============================================================================

def train_federated(server, config, train_loaders=None):
    """
    Điều phối quá trình huấn luyện (chọn tuần tự hoặc song song)
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 BẮT ĐẦU HUẤN LUYỆN FEDERATED")
    logger.info("="*80)

    algorithm = config['algorithm']
    num_rounds = config['num_rounds']
    local_epochs = config['local_epochs']
    learning_rate = config['learning_rate']
    client_fraction = config['client_fraction']
    eval_every = config['eval_every']
    device = config['device']
    use_multiprocessing = config.get('use_multiprocessing', False)
    
    history = server.history # Lấy history từ server

    # In Cấu hình
    logger.info(f"\n📋 Cấu hình huấn luyện:")
    logger.info(f"   - Thuật toán: {algorithm.upper()}")
    logger.info(f"   - Số round: {num_rounds}")
    logger.info(f"   - Số epoch local: {local_epochs}")
    logger.info(f"   - Learning rate: {learning_rate}")
    logger.info(f"   - Batch size: {config['batch_size']}")
    logger.info(f"   - Tỉ lệ client mỗi round: {client_fraction}")
    logger.info(f"   - Thiết bị: {device}")
    logger.info(f"   - Chạy song song (Multiprocessing): {use_multiprocessing}")
    if use_multiprocessing:
        logger.info(f"   - Số Processes: {config['num_processes']}")
    if algorithm == 'fedprox':
        logger.info(f"   - Mu (proximal term): {config['mu']}")
    
    # Đánh giá ban đầu (Round 0)
    if eval_every > 0:
        logger.info("\n📊 Đánh giá mô hình toàn cục (chưa huấn luyện)...")
        eval_result = server.evaluate_global()
        history['train_loss'].append(None) # Chưa train
        history['test_accuracy'].append(eval_result['accuracy'])
        history['test_loss'].append(eval_result['loss'])
        logger.info(f"✅ Round 0 (Init): Test Acc: {eval_result['accuracy']*100:.2f}% | Test Loss: {eval_result['loss']:.4f}")

    # === BẮT ĐẦU CÁC VÒNG HUẤN LUYỆN ===
    round_iter = tqdm(
        range(num_rounds),
        desc="Global Rounds",
        unit="round"
    )

    for round_idx in round_iter:
        logger.info(f"\n{'='*60}")
        logger.info(f"📍 ROUND {round_idx+1}/{num_rounds}")
        logger.info(f"{'='*60}")
        
        # Chọn chế độ chạy (Song song hoặc Tuần tự)
        if use_multiprocessing:
            # ===== CHẠY SONG SONG (MULTIPROCESSING) =====
            logger.info("\n🔥 Chế độ: MULTIPROCESSING - Clients chạy song song")
            
            # Cập nhật config cho worker
            worker_config = config.copy()
            worker_config['local_epochs'] = local_epochs
            worker_config['learning_rate'] = learning_rate
            worker_config['algorithm'] = algorithm
            worker_config['mu'] = config.get('mu', 0.01)

            # ✅ SỬA LỖI TypeError: Xóa các keyword arguments không hợp lệ
            round_result = train_round_multiprocessing(
                server=server,
                config=worker_config,
                # train_loaders=train_loaders, # <-- ĐÃ XÓA (Gây lỗi)
                device=device
            )
        
        else:
            # ===== CHẠY TUẦN TỰ (SEQUENTIAL) =====
            logger.info("\n📝 Chế độ: SEQUENTIAL - Clients chạy lần lượt (chậm)...")
            if algorithm == 'fedavg':
                round_result = server.train_round_fedavg(
                    num_epochs=local_epochs,
                    learning_rate=learning_rate,
                    client_fraction=client_fraction,
                    verbose=1 # Bật verbose cho client
                )
            elif algorithm == 'fedprox':
                round_result = server.train_round_fedprox(
                    num_epochs=local_epochs,
                    mu=config['mu'],
                    learning_rate=learning_rate,
                    client_fraction=client_fraction,
                    verbose=1 # Bật verbose cho client
                )
            else:
                raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")

        # Đánh giá sau mỗi vòng
        if (round_idx + 1) % eval_every == 0:
            logger.info(f"\n📊 Đang đánh giá mô hình toàn cục trên test set...")
            eval_result = server.evaluate_global()

            history['train_loss'].append(round_result['train_loss'])
            history['test_accuracy'].append(eval_result['accuracy'])
            history['test_loss'].append(eval_result['loss'])

            logger.info(f"\n✅ Round {round_idx+1}/{num_rounds} Summary:")
            logger.info(f"   • Train Loss (Avg): {round_result['train_loss']:.4f}")
            logger.info(f"   • Test Accuracy: {eval_result['accuracy']*100:.2f}%")
            logger.info(f"   • Test Loss: {eval_result['loss']:.4f}")
            
            round_iter.set_postfix({
                "algo": algorithm,
                "train_loss": f"{round_result['train_loss']:.4f}",
                "test_acc": f"{eval_result['accuracy']*100:.2f}%"
            })

    if history['test_accuracy']:
        logger.info(f"\n✅ Huấn luyện {algorithm.upper()} hoàn tất.")
        logger.info(f"   → Độ chính xác cuối cùng trên test: {history['test_accuracy'][-1]*100:.2f}%")
    else:
        logger.info("\n⚠️ Không có kết quả test nào được ghi nhận.")

    return history

# ============================================================================
# 💡 BƯỚC 10: HÀM VẼ BIỂU ĐỒ & LƯU KẾT QUẢ 💡
# ============================================================================

def plot_training_history(history, save_path):
    """
    Vẽ biểu đồ train_loss, test_loss và test_accuracy theo round.
    """
    logger.info("\n" + "="*80)
    logger.info("📊 ĐANG VẼ BIỂU ĐỒ KẾT QUẢ HUẤN LUYỆN")
    logger.info("="*80)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Lấy số vòng (rounds), bao gồm cả round 0
        rounds = range(len(history['test_loss'])) 
        train_loss = history['train_loss'] # train_loss[0] là None
        test_loss = history['test_loss']
        test_acc = history['test_accuracy']

        # Loss
        ax1 = axes[0]
        # Bỏ qua điểm None đầu tiên của train_loss
        ax1.plot(rounds[1:], train_loss[1:], label='Train Loss (Trung bình Client)', marker='o', linewidth=2)
        ax1.plot(rounds, test_loss, label='Test Loss (Toàn cục)', marker='s', linewidth=2)
        ax1.set_xlabel('Round', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Train Loss & Test Loss', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(rounds)

        # Accuracy
        ax2 = axes[1]
        test_acc_pct = [acc * 100 for acc in test_acc]
        ax2.plot(rounds, test_acc_pct, label='Test Accuracy (%)', marker='o', linewidth=2, color='green')
        ax2.set_xlabel('Round', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Độ chính xác trên Test Set Toàn cục', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.set_xticks(rounds)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Đã lưu biểu đồ lịch sử huấn luyện tại: {save_path}")
        plt.show() # Hiển thị plot trong notebook
    
    except Exception as e:
        logger.warning(f"⚠️ Lỗi khi vẽ biểu đồ: {e}")
    finally:
        plt.close() # Đảm bảo đóng plot
        logger.info("="*80)


def evaluate_and_save_results(server, history, config, output_dir, data_stats, training_duration, start_time, end_time):
    """
    Đánh giá cuối cùng, lưu model, history, config và tạo các báo cáo.
    """
    logger.info("\n" + "="*80)
    logger.info("💾 BƯỚC 9 & 10: ĐÁNH GIÁ CUỐI CÙNG VÀ LƯU KẾT QUẢ")
    logger.info("="*80)
    
    device = config['device']

    # 1. Lưu Model
    model_path = os.path.join(output_dir, 'global_model.pth')
    torch.save(server.global_model.state_dict(), model_path)
    logger.info(f"✅ Đã lưu mô hình toàn cục: {model_path}")

    # 2. Lưu History
    history_path = os.path.join(output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"✅ Đã lưu lịch sử huấn luyện: {history_path}")

    # 3. Lưu Config
    config_path = os.path.join(output_dir, 'config.json')
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device'])
    config_to_save['input_shape'] = str(config['input_shape']) # Chuyển tuple sang str
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    logger.info(f"✅ Đã lưu cấu hình: {config_path}")

    # 4. Lưu Thống kê Dữ liệu
    stats_path = os.path.join(output_dir, 'data_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(data_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Đã lưu thống kê dữ liệu: {stats_path}")

    # 5. Vẽ Biểu đồ
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # 6. Đánh giá cuối cùng (Lấy dự đoán)
    logger.info("\n→ Đang tạo dự đoán (predictions) trên Test Set toàn cục...")
    all_y_true = []
    all_y_pred = []
    server.global_model.eval()
    
    # Dùng list test loader từ server
    pbar_predict = tqdm(
        server.client_test_loaders,
        desc="[Predict] Lấy dự đoán từ Test Set",
        unit="loader",
        leave=False
    )

    with torch.no_grad():
        for loader in pbar_predict: 
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = server.global_model(data)
                pred = output.argmax(dim=1)
                
                all_y_true.append(target.cpu().numpy())
                all_y_pred.append(pred.cpu().numpy())
                
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    logger.info("✅ Đã tạo dự đoán xong.")

    # 7. In và Lưu Báo cáo Phân loại
    logger.info("\n" + "="*80)
    logger.info("📄 CLASSIFICATION REPORT")
    logger.info("="*80)
    class_labels = [str(i) for i in range(config['num_classes'])]
    report = classification_report(
        y_true, 
        y_pred, 
        labels=range(config['num_classes']),
        target_names=class_labels,
        zero_division=0,
        digits=4
    )
    print(report) # In ra màn hình
    
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n" + "="*80 + "\n\n" + report)
    logger.info(f"\n💾 Đã lưu report: {report_path}")

    # 8. Vẽ và Lưu Ma trận Nhầm lẫn
    logger.info("\n→ Đang tạo ma trận nhầm lẫn (Confusion Matrix)...")
    cm = confusion_matrix(y_true, y_pred, labels=range(config['num_classes']))
    
    show_labels = (config['num_classes'] <= 40)
    fig_size = max(12, config['num_classes'] * 0.4)
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    
    sns.heatmap(
        cm, 
        annot=show_labels, 
        fmt='d', 
        cmap='Blues',
        cbar=True,
        xticklabels=class_labels if show_labels else False,
        yticklabels=class_labels if show_labels else False
    )
    
    final_test_accuracy = history['test_accuracy'][-1]
    
    plt.title(f'Confusion Matrix - Final Global Model\n'
              f'Test Accuracy: {final_test_accuracy:.4f} ({final_test_accuracy*100:.2f}%)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    if show_labels:
        plt.xticks(rotation=90); plt.yticks(rotation=0)
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Đã lưu confusion matrix: {cm_path}")
    plt.show()
    plt.close()

    # 9. Lưu Metrics chi tiết (F1, Precision, Recall)
    logger.info("\n→ Đang lưu metrics chi tiết (F1, Precision, Recall)...")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        labels=range(config['num_classes']),
        average=None,
        zero_division=0
    )
    detailed_metrics = {
        'class': class_labels,
        'precision': precision, 'recall': recall, 'f1_score': f1, 'support': support
    }
    df_metrics = pd.DataFrame(detailed_metrics)
    csv_path = os.path.join(output_dir, "detailed_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"✅ Đã lưu detailed metrics: {csv_path}")

    logger.info("\n" + "="*80)
    logger.info("📊 TOP 5 CLASSES PERFORMANCE:")
    logger.info("="*80)
    df_sorted = df_metrics.sort_values('f1_score', ascending=False)
    print("\nTop 5 Best Classes (by F1-score):")
    print(df_sorted.head(5).to_string(index=False))
    print("\nTop 5 Worst Classes (by F1-score):")
    print(df_sorted.tail(5).to_string(index=False))
    logger.info("="*80)
    
    # 10. Tạo Summary Report
    logger.info("\n" + "="*80)
    logger.info("📝 BƯỚC 10: TẠO SUMMARY REPORT")
    logger.info("="*80)
    summary_path = os.path.join(OUTPUT_DIR, "SUMMARY_REPORT.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n" + " "*20 + "FEDERATED LEARNING SUMMARY REPORT\n" + "="*80 + "\n\n")
        f.write("📅 THỜI GIAN:\n")
        f.write(f"  • Bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  • Kết thúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  • Tổng thời gian: {training_duration:.2f}s ({training_duration/60:.2f} phút)\n\n")
        f.write("⚙️  CẤU HÌNH:\n")
        f.write(f"  • Chiến lược: {config['algorithm'].upper()}\n")
        if config['algorithm'] == 'fedprox':
            f.write(f"  • Mu (proximal): {config['mu']}\n")
        f.write(f"  • Số clients: {config['num_clients']}\n")
        f.write(f"  • Số rounds: {config['num_rounds']}\n")
        f.write(f"  • Epochs/round: {config['local_epochs']}\n")
        f.write(f"  • Batch size: {config['batch_size']}\n")
        f.write(f"  • Learning rate: {config['learning_rate']}\n")
        f.write(f"  • Input features: {config['input_shape'][0]}\n")
        f.write(f"  • Num classes: {config['num_classes']}\n")
        f.write(f"  • Chạy song song: {config['use_multiprocessing']}\n")
        
        f.write("\n📊 KẾT QUẢ CUỐI CÙNG (TỔNG HỢP TỪ TEST SET):\n")
        if history['test_accuracy']:
            final_acc = history['test_accuracy'][-1]
            final_loss = history['test_loss'][-1]
            f.write(f"  • Final Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
            f.write(f"  • Final Test Loss: {final_loss:.4f}\n")
        
        f.write("\n📁 OUTPUT FILES:\n")
        f.write(f"  • Thư mục: {OUTPUT_DIR}\n")
        f.write(f"  • Model: global_model.pth\n")
        f.write(f"  • History: training_history.pkl\n")
        f.write(f"  • Plots: training_history.png\n")
        f.write(f"  • Report: classification_report.txt\n")
        f.write(f"  • Metrics: detailed_metrics.csv\n")
        f.write(f"  • Config: config.json\n")
        
        f.write("\n" + "="*80 + "\n" + "✅ HUẤN LUYỆN THÀNH CÔNG!\n" + "="*80 + "\n")

    logger.info(f"✅ Đã tạo summary report: {summary_path}")

# ============================================================================
# 💡 BƯỚC 11: HÀM MAIN (ĐỂ CHẠY) 💡
# ============================================================================

def main():
    config = CONFIG
    start_time = datetime.now()

    logger.info("="*80)
    logger.info("🤖 FEDERATED LEARNING VỚI MÔ HÌNH CNN-GRU (IoT IDS)")
    logger.info("="*80)

    try:
        # Bước 1: Kiểm tra thiết bị
        device = check_and_setup_gpu(config)
        config['device'] = device

        # Bước 2: Tự động phát hiện tham số dữ liệu
        input_shape, num_classes, data_stats = auto_detect_data_parameters(
            data_dir=config['data_dir'],
            num_clients=config['num_clients']
        )
        config['input_shape'] = input_shape
        config['num_classes'] = num_classes

        # In cấu hình cuối cùng
        logger.info("\n" + "="*80)
        logger.info("⚙️  CẤU HÌNH CUỐI CÙNG")
        logger.info("="*80)
        # Chuyển config sang string để in
        config_str = json.dumps(config, indent=2, default=str)
        print(config_str) # Dùng print để hiển thị đẹp
        logger.info("="*80)

        # Bước 3: Load dữ liệu
        train_loaders, test_loaders = load_federated_data(
            data_dir=config['data_dir'],
            num_clients=config['num_clients'],
            batch_size=config['batch_size'],
            device=config['device']
        )
        
        # Nếu dùng multiprocessing, chúng ta cần giữ chỗ (placeholder)
        # vì không thể truyền DataLoader qua các process
        client_train_loaders = train_loaders
        if config['use_multiprocessing']:
            client_train_loaders = [None] * config['num_clients'] # Chỉ là giữ chỗ


        # Bước 4: Khởi tạo hệ thống federated
        server, clients = initialize_federated_system(
            config=config, # Truyền config vào
            train_loaders=client_train_loaders, # Sẽ là [None, None,...] nếu dùng multiprocessing
            test_loaders=test_loaders
        )

        # Bước 5: Huấn luyện federated
        history = train_federated(
            server, 
            config, 
            train_loaders=train_loaders # Truyền Dataloader thật (chỉ dùng nếu multiprocessing)
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("🏁 HUẤN LUYỆN HOÀN TẤT!")
        logger.info("="*80)
        logger.info(f"⏱️  Thời gian huấn luyện: {training_duration:.2f} giây ({training_duration/60:.2f} phút)")

        # Bước 6: Lưu kết quả
        evaluate_and_save_results(
            server, history, config, 
            config['output_dir'], data_stats,
            training_duration, start_time, end_time
        )
        
        logger.info("\n" + "="*80)
        logger.info("🎉 🎉 🎉  HOÀN TẤT TẤT CẢ CÁC BƯỚC  🎉 🎉 🎉")
        logger.info("="*80)

    except Exception as e:
        logger.error("\n❌ ĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH CHẠY SCRIPT")
        logger.error(f"Chi tiết lỗi: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Thiết lập 'spawn' là phương thức bắt đầu cho multiprocessing
    # BẮT BUỘC phải đặt trong block if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        # Bỏ qua lỗi nếu nó đã được set
        if "context has already been set" not in str(e):
            logger.warning(f"Không thể set 'spawn' start method: {e}")
        
    main()