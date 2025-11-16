################################################################################
#                                                                              #
#  SCRIPT HUẤN LUYỆN FEDERATED LEARNING (FLOWER + PYTORCH)                      #
#  Framework: Flower (Tự động dùng Ray cho chạy song song)                      #
#  Mô hình: 🌟 PyTorch CNN-GRU (Tự code) 🌟                                  #
#  Chiến lược: 🌟 FedProx (Của Flower) 🌟                                      #
#  Tính năng: ⚡ TỐI ƯU GPU (Tự động) + BÁO CÁO F1-Score                         #
#                                                                              #
################################################################################

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import os
import logging
import pickle
import json
import copy
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm.auto import tqdm
import time
import torch.multiprocessing as mp # Import để set 'spawn'

# === THÊM THƯ VIỆN CHO BIỂU ĐỒ VÀ KẾT QUẢ ===
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support
)
import pandas as pd
# ============================================

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 💡 BƯỚC 1: CẤU HÌNH CHÍNH 💡
# ============================================================================

CONFIG = {
    # ⬇️ CHỈNH ĐƯỜNG DẪN NÀY (TRÊN COLAB CẦN MOUNT DRIVE TRƯỚC) ⬇️
    'data_dir': '/kaggle/input/fed-5clients',
    'output_dir': './results',
    
    'num_clients': 5,

    # Model params (sẽ được tự động phát hiện từ data)
    'input_shape': None,  
    'num_classes': None,  

    # Training params
    'algorithm': 'fedprox',     # 'fedavg' hoặc 'fedprox'
    'num_rounds': 1,            # 1 Vòng
    'local_epochs': 1,          # 1 Epoch/Vòng
    'learning_rate': 0.001,
    'batch_size': 1024,         
    'client_fraction': 1.0,     # 1.0 = chọn tất cả 5 client

    # FedProx specific
    'mu': 0.01,  # Proximal term coefficient

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'force_gpu': True,
    
    # Visualization
    'eval_every': 1,  # Đánh giá sau mỗi round
}

# === TẠO THƯ MỤC OUTPUT ===
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(CONFIG['output_dir'], f"run_{TIMESTAMP}_{CONFIG['algorithm']}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG['output_dir'] = OUTPUT_DIR 


# ============================================================================
# 💡 BƯỚC 2: ĐỊNH NGHĨA MÔ HÌNH CNN-GRU (PyTorch) 💡
# (Nội dung từ model.py của bạn)
# ============================================================================

class CNN_GRU_Model(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(CNN_GRU_Model, self).__init__()

        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
        else:
            seq_length = input_shape

        self.input_shape = input_shape
        self.num_classes = num_classes
        
        def conv_output_shape(L_in, kernel_size=1, stride=1, padding=0, dilation=1):
            if padding == 1 and kernel_size == 3: L_out_conv = L_in
            else: L_out_conv = (L_in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
            return L_out_conv
        def pool_output_shape(L_in, kernel_size=2, stride=2, padding=0, dilation=1):
            return (L_in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1

        # ===== CNN MODULE =====
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn3 = nn.Dropout(0.3)

        cnn_output_length = seq_length
        cnn_output_length = pool_output_shape(conv_output_shape(cnn_output_length, kernel_size=3, padding=1))
        cnn_output_length = pool_output_shape(conv_output_shape(cnn_output_length, kernel_size=3, padding=1))
        cnn_output_length = pool_output_shape(conv_output_shape(cnn_output_length, kernel_size=3, padding=1))
        self.cnn_output_size = 256 * cnn_output_length
        
        # ===== GRU MODULE =====
        self.gru1 = nn.GRU(input_size=1, hidden_size=128, batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, batch_first=True)
        self.gru_output_size = 64

        # ===== MLP MODULE =====
        concat_size = self.cnn_output_size + self.gru_output_size
        self.dense1 = nn.Linear(concat_size, 256)
        self.bn_mlp1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(256, 128)
        self.bn_mlp2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

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

def build_cnn_gru_model(input_shape, num_classes=2):
    """Hàm tiện ích để khởi tạo model CNN-GRU."""
    model = CNN_GRU_Model(input_shape, num_classes)
    logger.info(f"✅ Khởi tạo mô hình CNN-GRU (PyTorch) thành công")
    logger.info(f"   - Kích thước input: {input_shape}")
    logger.info(f"   - Số lớp (num_classes): {num_classes}")
    return model

# ============================================================================
# 💡 BƯỚC 3: HÀM LOAD DỮ LIỆU 💡
# ============================================================================

class NumpyDataset(TensorDataset):
    """Dataset tiện dụng để wrap numpy array thành TensorDataset."""
    def __init__(self, X, y):
        if len(X.shape) == 3: X = X.squeeze(-1)
        X = X.astype(np.float32)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y).long()
        super().__init__(X_tensor, y_tensor)

def load_data_for_client(data_dir, client_id, batch_size):
    """
    Load data CHỈ CHO 1 client (dùng trong client_fn)
    """
    data_path = os.path.join(data_dir, f'client_{client_id}_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy dữ liệu của client {client_id} tại: {data_path}")
        
    data = np.load(data_path)
    X_train = data['X_train']; y_train = data['y_train']
    X_test = data['X_test']; y_test = data['y_test']
    
    train_dataset = NumpyDataset(X_train, y_train)
    test_dataset = NumpyDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
    
    logger.info(f"   - [client_fn(cid={client_id})] Đã load {len(train_dataset):,} train, {len(test_dataset):,} test.")
    return train_loader, test_loader, len(train_dataset)

def load_global_test_set(data_dir, num_clients, batch_size):
    """
    ✅ SỬA LỖI RAM OOM: Tạo 1 DataLoader gộp (dùng để đánh giá)
    """
    logger.info("\n→ Tạo global test loader (gộp test của tất cả client)...")
    all_X_test = []
    all_y_test = []
    
    for client_id in range(num_clients):
        data_path = os.path.join(data_dir, f'client_{client_id}_data.npz')
        with np.load(data_path) as data:
            all_X_test.append(data['X_test'])
            all_y_test.append(data['y_test'])
        
    X_test_global = np.concatenate(all_X_test, axis=0)
    y_test_global = np.concatenate(all_y_test, axis=0)
    
    global_test_dataset = NumpyDataset(X_test_global, y_test_global)
    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size * 2, shuffle=False)
    
    logger.info(f"   - Kích thước global test set: {len(global_test_dataset):,} mẫu.")
    return global_test_loader

# ============================================================================
# 💡 BƯỚC 4: ĐỊNH NGHĨA FLOWER CLIENT (PyTorch) 💡
# (Đây là logic "tự code" của bạn, được bọc trong Flower)
# ============================================================================

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader, device):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        """Lấy tham số model (weights) và chuyển sang NumPy"""
        # logger.info(f"[Client {self.cid}] Đang gửi tham số (get_parameters)")
        # Chuyển state_dict (OrderedDict) thành list các mảng NumPy
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Nhận tham số (weights) từ server và cập nhật model"""
        # logger.info(f"[Client {self.cid}] Đang nhận tham số (set_parameters)")
        # Chuyển list mảng NumPy trở lại thành state_dict
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Huấn luyện model trên dữ liệu local (FedAvg hoặc FedProx)
        """
        logger.info(f"[Client {self.cid}] Bắt đầu huấn luyện (fit)...")
        self.set_parameters(parameters) # Nhận model mới từ server
        
        # Lấy tham số từ server
        algorithm = config.get('algorithm', 'fedavg')
        epochs = config.get('local_epochs', 1)
        learning_rate = config.get('learning_rate', 0.001)
        mu = config.get('mu', 0.01)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Lưu global params (nếu là FedProx)
        global_params_dict = None
        if algorithm == 'fedprox':
            # ✅ SỬA LỖI: Sử dụng state_dict().keys() để mapping đúng với parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)
            global_params_dict = {
                k: torch.tensor(v).to(self.device)
                for k, v in params_dict
            }

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            pbar = tqdm(
                self.trainloader,
                desc=f"[Client {self.cid}] Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=False,
                position=int(self.cid) # Vị trí thanh progress
            )
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                ce_loss = criterion(output, target)
                loss = ce_loss # Mặc định là FedAvg

                if algorithm == 'fedprox':
                    # === ✅ SỬA LỖI: Logic "tự code" FedProx ===
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            global_param = global_params_dict[name]
                            proximal_term += torch.sum((param - global_param) ** 2)
                    
                    proximal_term = (mu / 2) * proximal_term
                    loss += proximal_term
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += ce_loss.item() * data.size(0) # Chỉ log CE loss
                epoch_samples += data.size(0)
                
                pbar.set_postfix({
                    "ce_loss": f"{ce_loss.item():.4f}",
                    "loss": f"{loss.item():.4f}"
                })

            avg_loss = epoch_loss / max(1, epoch_samples)
            # Không in log ở đây, để pbar tự xử lý

        # Trả về model đã huấn luyện (dưới dạng NumPy) và số mẫu
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"avg_loss": avg_loss}

    def evaluate(self, parameters, config):
        """
        Flower sẽ KHÔNG gọi hàm này nếu 'evaluate_fn' được định nghĩa ở server
        """
        return 0.0, 0, {"accuracy": 0.0}

# ============================================================================
# 💡 BƯỚC 5: HÀM TỰ ĐỘNG PHÁT HIỆN THAM SỐ DỮ LIỆU 💡
# ============================================================================

def auto_detect_data_parameters(data_dir, num_clients):
    logger.info("\n" + "="*80)
    logger.info("📂 TỰ ĐỘNG PHÁT HIỆN THAM SỐ DỮ LIỆU")
    logger.info("="*80)
    logger.info(f"→ Thư mục dữ liệu: {data_dir}")
    logger.info(f"→ Số lượng client (dự kiến): {num_clients}")

    try:
        all_labels = []
        data_stats = {}
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
                all_labels.append(y_train); all_labels.append(y_test) # Quét cả train và test
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
# 💡 BƯỚC 6: HÀM VẼ BIỂU ĐỒ & LƯU KẾT QUẢ 💡
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
        
        # Lấy dữ liệu từ history.losses_distributed (list of tuples (round, loss))
        # Bỏ qua round 0 nếu có (vì ta có evaluate_fn ở round 0)
        if history.losses_distributed[0][0] == 0:
             rounds = [r for r, _ in history.losses_distributed][1:]
             train_loss = [l for _, l in history.losses_distributed][1:]
        else:
             rounds = [r for r, _ in history.losses_distributed]
             train_loss = [l for _, l in history.losses_distributed]
        
        # Lấy dữ liệu từ history.metrics_centralized (dict of lists)
        # Bỏ qua round 0 (init)
        rounds_eval = [r for r, _ in history.metrics_centralized['accuracy']]
        test_acc = [a for _, a in history.metrics_centralized['accuracy']]
        test_loss = [l for _, l in history.metrics_centralized['test_loss']]


        # Loss
        ax1 = axes[0]
        ax1.plot(rounds, train_loss, label='Train Loss (Trung bình Client)', marker='o', linewidth=2)
        ax1.plot(rounds_eval, test_loss, label='Test Loss (Toàn cục)', marker='s', linewidth=2)
        ax1.set_xlabel('Round', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Train Loss & Test Loss', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(rounds_eval)

        # Accuracy
        ax2 = axes[1]
        test_acc_pct = [acc * 100 for acc in test_acc]
        ax2.plot(rounds_eval, test_acc_pct, label='Test Accuracy (%)', marker='o', linewidth=2, color='green')
        ax2.set_xlabel('Round', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Độ chính xác trên Test Set Toàn cục', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.set_xticks(rounds_eval)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Đã lưu biểu đồ lịch sử huấn luyện tại: {save_path}")
        plt.show() # Hiển thị plot trong notebook
    
    except Exception as e:
        logger.warning(f"⚠️ Lỗi khi vẽ biểu đồ: {e}")
        logger.warning(f"History object (metrics_centralized): {history.metrics_centralized}")
        logger.warning(f"History object (losses_distributed): {history.losses_distributed}")
    finally:
        plt.close() # Đảm bảo đóng plot
        logger.info("="*80)


def evaluate_and_save_results(
    server_model, history, config, output_dir, data_stats, 
    training_duration, start_time, end_time
):
    """
    Đánh giá cuối cùng, lưu model, history, config và tạo các báo cáo.
    """
    logger.info("\n" + "="*80)
    logger.info("💾 BƯỚC 9 & 10: ĐÁNH GIÁ CUỐI CÙNG VÀ LƯU KẾT QUẢ")
    logger.info("="*80)
    
    device = config['device']

    # 1. Lưu Model
    model_path = os.path.join(output_dir, 'global_model.pth')
    torch.save(server_model.state_dict(), model_path)
    logger.info(f"✅ Đã lưu mô hình toàn cục: {model_path}")

    # 2. Lưu History (từ Flower)
    history_path = os.path.join(output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"✅ Đã lưu lịch sử huấn luyện: {history_path}")

    # 3. Lưu Config
    config_path = os.path.join(output_dir, 'config.json')
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device'])
    config_to_save['input_shape'] = str(config['input_shape']) 
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
    global_test_loader = load_global_test_set(
        config['data_dir'], config['num_clients'], config['batch_size']
    )
    
    all_y_true = []
    all_y_pred = []
    server_model.to(device) # Đảm bảo model trên đúng device
    server_model.eval()
    
    pbar_predict = tqdm(
        global_test_loader,
        desc="[Predict] Lấy dự đoán từ Test Set",
        unit="batch",
        leave=False
    )

    with torch.no_grad():
        for data, target in pbar_predict:
            data, target = data.to(device), target.to(device)
            output = server_model(data)
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
    
    final_test_accuracy = history.metrics_centralized['accuracy'][-1][1] # Lấy giá trị acc cuối
    
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
        f.write(f"  • Chạy song song: Flower (Ray)\n")
        
        f.write("\n📊 KẾT QUẢ CUỐI CÙNG (TỔNG HỢP TỪ TEST SET):\n")
        if history.metrics_centralized['accuracy']:
            final_acc = history.metrics_centralized['accuracy'][-1][1]
            final_loss = history.losses_distributed[-1][1]
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
        # Trả về string 'cuda'
        device_str = 'cuda'
        
    else:
        if config.get('force_gpu', False):
            logger.error("\n❌ LỖI: Không phát hiện GPU nhưng force_gpu=True")
            raise RuntimeError("Yêu cầu GPU nhưng không có GPU khả dụng.")
        else:
            device = torch.device('cpu')
            logger.warning(f"\n⚠️ Cảnh báo: Không có GPU, hệ thống sẽ chạy trên CPU (chậm hơn).")
            device_str = 'cpu'

    logger.info("="*80)
    return device_str # Trả về string

def main():
    config = CONFIG
    start_time = datetime.now()

    logger.info("="*80)
    logger.info("🤖 FEDERATED LEARNING VỚI MÔ HÌNH CNN-GRU (IoT IDS)")
    logger.info("="*80)

    try:
        # Bước 1: Kiểm tra thiết bị
        device_str = check_and_setup_gpu(config)
        config['device'] = device_str # Lưu string 'cuda' hoặc 'cpu'

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
        config_str = json.dumps(config, indent=2, default=str)
        print(config_str) # Dùng print để hiển thị đẹp
        logger.info("="*80)
        
        # --- Định nghĩa Client Function (client_fn) cho Flower ---
        # Hàm này sẽ được Ray gọi để tạo client trên một process/GPU riêng
        
        def client_fn(cid: str) -> fl.client.Client:
            """Tạo một Flower client (PyTorch)"""
            
            # 1. Tải dữ liệu cho client này
            trainloader, testloader, num_train = load_data_for_client(
                data_dir=config['data_dir'],
                client_id=int(cid),
                batch_size=config['batch_size']
            )
            
            # 2. Tạo model cho client này
            model = build_cnn_gru_model(
                input_shape=config['input_shape'],
                num_classes=config['num_classes']
            )
            # Chuyển model lên device (GPU/CPU)
            device = torch.device(config['device'])
            model.to(device)

            # 3. Tạo Flower client
            client = FlowerClient(
                cid=cid,
                model=model,
                trainloader=trainloader,
                testloader=testloader,
                device=config['device'] # Truyền 'cuda' hoặc 'cpu'
            )
            
            return client.to_client() # Chuyển đổi thành Flower Client
        
        # --- Định nghĩa Strategy (Chiến lược) cho Flower Server ---
        
        # Hàm này dùng để server đánh giá model toàn cục
        def get_evaluate_fn(global_test_loader, device_str):
            """
            Trả về một hàm đánh giá (evaluate_fn) cho server.
            Hàm này sẽ chạy trên server (hoặc 1 process riêng).
            """
            def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config_eval: Dict[str, fl.common.Scalar],
            ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
                
                device = torch.device(device_str)
                
                # Tạo model tạm thời để đánh giá
                model = build_cnn_gru_model(
                    input_shape=CONFIG['input_shape'],
                    num_classes=CONFIG['num_classes']
                )
                model.to(device)
                
                # Cập nhật model với tham số từ server
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
                
                model.eval()
                criterion = nn.CrossEntropyLoss()
                total_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    pbar_eval = tqdm(
                        global_test_loader,
                        desc=f"[Server Eval] Round {server_round}",
                        unit="batch",
                        leave=False
                    )
                    for data, target in pbar_eval:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        total_loss += loss.item() * data.size(0)
                        pred = output.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += data.size(0)
                        
                        if total > 0:
                            pbar_eval.set_postfix({
                                "acc": f"{correct / total * 100:.2f}%",
                                "loss": f"{total_loss / total:.4f}"
                            })

                accuracy = correct / total if total > 0 else 0.0
                avg_loss = total_loss / total if total > 0 else 0.0
                
                logger.info(f"✅ Round {server_round} (Server Eval): Test Acc: {accuracy*100:.2f}% | Test Loss: {avg_loss:.4f}")
                # Trả về loss (bắt buộc), và dict metrics
                return avg_loss, {"accuracy": accuracy, "test_loss": avg_loss}
            
            return evaluate

        # --- Khởi tạo Strategy ---
        logger.info("\n→ Khởi tạo chiến lược (Strategy)...")
        
        # Tạo Global Test Loader (dùng 1 lần cho server)
        global_test_loader = load_global_test_set(
            config['data_dir'], config['num_clients'], config['batch_size']
        )
        
        # Hàm gửi config cho client (để client biết epochs, lr, mu)
        def fit_config(server_round: int) -> Dict:
            return {
                "server_round": server_round,
                "local_epochs": config['local_epochs'],
                "learning_rate": config['learning_rate'],
                "algorithm": config['algorithm'],
                "mu": config['mu'],
                "batch_size": config['batch_size']
            }

        if config['algorithm'] == 'fedprox':
            strategy = fl.server.strategy.FedProx(
                fraction_fit=config['client_fraction'],
                fraction_evaluate=0.0, # Tắt client-side evaluation (dùng evaluate_fn thay thế)
                min_fit_clients=int(config['num_clients'] * config['client_fraction']),
                min_available_clients=config['num_clients'],
                evaluate_fn=get_evaluate_fn(global_test_loader, config['device']), # Đánh giá phía server
                on_fit_config_fn=fit_config, # Gửi config cho client
                proximal_mu=config['mu']
            )
        else: # fedavg
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=config['client_fraction'],
                fraction_evaluate=0.0, # Tắt client-side evaluation (dùng evaluate_fn thay thế)
                min_fit_clients=int(config['num_clients'] * config['client_fraction']),
                min_available_clients=config['num_clients'],
                evaluate_fn=get_evaluate_fn(global_test_loader, config['device']),
                on_fit_config_fn=fit_config
            )
        
        logger.info(f"✅ Strategy {config['algorithm'].upper()} đã được tạo.")

        # --- Cấu hình tài nguyên (Cho Ray chạy song song) ---
        client_resources = None
        if config['device'] == 'cuda':
            num_gpus_total = torch.cuda.device_count()
            # Chia GPU cho các client
            gpu_per_client = num_gpus_total / config['num_clients']
            
            # Colab free (T4) chỉ có 1 GPU, Colab Pro (A100) có 1 GPU
            # 1 CPU core cho mỗi client là đủ
            client_resources = {"num_cpus": 1, "num_gpus": gpu_per_client}
            
            logger.info(f"\n🖥️  GPU Mode: Cấu hình Ray cho {config['num_clients']} client song song.")
            logger.info(f"   - Tổng GPU: {num_gpus_total}")
            logger.info(f"   - CPU/client: {client_resources['num_cpus']}")
            logger.info(f"   - GPU/client: {client_resources['num_gpus']:.2f}")
        else:
            # Colab free chỉ có 2 CPU, chạy 5 client song song sẽ rất chậm
            client_resources = {"num_cpus": 1}
            logger.info(f"\n💻 CPU Mode: Cấu hình Ray cho 2 client song song (tối đa).")


        # Bước 5: Huấn luyện (Dùng Flower/Ray)
        logger.info("\n" + "="*80)
        logger.info("🚀 BẮT ĐẦU HUẤN LUYỆN FEDERATED (với Flower/Ray)")
        logger.info("="*80)
        
        # Thêm
        # logging.getLogger("flwr").setLevel(logging.DEBUG)

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config['num_clients'],
            config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
            strategy=strategy,
            client_resources=client_resources
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("🏁 HUẤN LUYỆN HOÀN TẤT!")
        logger.info("="*80)
        logger.info(f"⏱️  Thời gian huấn luyện: {training_duration:.2f} giây ({training_duration/60:.2f} phút)")

        # Bước 6: Lưu kết quả
        # Lấy model cuối cùng từ server (từ strategy)
        logger.info("→ Đang lấy mô hình toàn cục cuối cùng từ server...")
        server_model = build_cnn_gru_model(config['input_shape'], config['num_classes'])
        
        # Lấy tham số cuối cùng từ strategy
        final_params = strategy.get_parameters(config={})
        final_weights = fl.common.parameters_to_ndarrays(final_params) 
        
        params_dict = zip(server_model.state_dict().keys(), final_weights)
        state_dict = OrderedDict({k: torch.tensor(np.copy(v)) for k, v in params_dict})
        server_model.load_state_dict(state_dict, strict=True)
        logger.info("✅ Đã lấy mô hình thành công.")

        evaluate_and_save_results(
            server_model, history, config, 
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
        logger.info("✅ Đã set 'spawn' start method cho multiprocessing.")
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            logger.warning(f"Không thể set 'spawn' start method: {e}")
        
    main()