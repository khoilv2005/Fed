################################################################################
#                                                                              #
#  SCRIPT HUẤN LUYỆN FEDERATED LEARNING (FLOWER) HOÀN CHỈNH                   #
#  Framework: Tự build (PyTorch) - 1 FILE DUY NHẤT                             #
#  Mô hình: CNN-GRU ĐẦY ĐỦ (Full Model)                                        #
#  Chiến lược: 🌟 FedAvg & FedProx (Tự code) 🌟                                #
#  Tính năng: ⚡ TỐI ƯU GPU + SONG SONG (Multiprocessing) + BÁO CÁO F1-Score   #
#                                                                              #
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torch.multiprocessing as mp  # Thêm thư viện multiprocessing
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
from tqdm.auto import tqdm  # ✅ Thêm tqdm cho progress bar
import time  # ✅ Để đo thời gian mỗi batch


# Hàm check gpu
def check_and_setup_gpu(config: Dict) -> str:
    """
    Kiểm tra xem GPU (CUDA) có khả dụng hay không,
    và thiết lập thiết bị ('cuda' hoặc 'cpu') để sử dụng.
    """

    # Kiểm tra cấu hình và khả năng của hệ thống
    if config.get('force_gpu', False) and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning(
            "⚠️ LỖI CẤU HÌNH: Bạn yêu cầu 'force_gpu=True' nhưng không tìm thấy GPU (CUDA). "
            "Buộc phải chuyển sang chạy trên CPU."
        )
    elif torch.cuda.is_available() and config['device'] == 'cuda':
        device = 'cuda'
        # Thiết lập device để in ra thông tin GPU
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"✅ Đã phát hiện và sử dụng GPU/CUDA: {device_name}")
    else:
        # Nếu cấu hình là 'cpu' hoặc không có GPU
        device = 'cpu'
        logger.info("⚙️ Chạy trên CPU theo cấu hình hoặc do không tìm thấy GPU.")

    # Cập nhật cấu hình và trả về thiết bị
    config['device'] = device
    return device


# ============================================
# === THÊM THƯ VIỆN CHO BIỂU ĐỒ VÀ KẾT QUẢ ===
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
    'data_dir': '/content/drive/MyDrive/Fed-Data/5-Client',
    'output_dir': '/content/drive/MyDrive/Fed-Data/5-Client/Results',  # Lưu kết quả

    'num_clients': 5,

    # Model params (sẽ được tự động phát hiện từ data)
    'input_shape': None,  # Tự động phát hiện
    'num_classes': None,  # Tự động phát hiện

    # Training params
    'algorithm': 'fedavg',     # 'fedavg' hoặc 'fedprox'
    'num_rounds': 10,           # Giảm số round cho chạy thử nghiệm nhanh
    'local_epochs': 5,         # 1 Epoch/Vòng
    'learning_rate': 0.001,
    'batch_size': 1024,        # Batch size lớn (GPU 15GB ok)
    'client_fraction': 1.0,    # Tỉ lệ clients tham gia mỗi round

    # FedProx specific
    'mu': 0.01,  # Proximal term coefficient

    # Device - Luôn ưu tiên GPU
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'force_gpu': True,  # Set False nếu muốn cho phép chạy trên CPU

    # Multiprocessing
    'use_multiprocessing': False,  # Chạy clients song song
    'num_processes': 1,           # Giảm số processes cho chạy thử nghiệm nhanh

    # Visualization
    'eval_every': 1,
}

# === TẠO THƯ MỤC OUTPUT ===
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(CONFIG['output_dir'], f"run_{TIMESTAMP}_{CONFIG['algorithm']}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG['output_dir'] = OUTPUT_DIR  # Cập nhật config với đường dẫn mới

# ============================================================================
# 💡 BƯỚC 2: ĐỊNH NGHĨA MÔ HÌNH CNN-GRU 💡
# ============================================================================


class CNN_GRU_Model(nn.Module):
    """
    Mô hình CNN-GRU (CNN + GRU + MLP + Softmax) bằng PyTorch
    (Phiên bản TỐI ƯU TỐC ĐỘ, đã tắt recurrent_dropout)
    """
    def __init__(self, input_shape, num_classes=2):
        super(CNN_GRU_Model, self).__init__()

        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
        else:
            seq_length = input_shape

        self.input_shape = input_shape
        self.num_classes = num_classes

        # ===== CNN MODULE =====
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn1 = nn.Dropout(0.2)

        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn2 = nn.Dropout(0.2)

        # Conv Block 3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn3 = nn.Dropout(0.3)

        # Tính toán kích thước output của CNN
        def conv_output_shape(L_in, kernel_size=1, stride=1, padding=0, dilation=1):
            return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        cnn_output_length = seq_length
        cnn_output_length = conv_output_shape(cnn_output_length, kernel_size=3, stride=1, padding=1)  # conv1
        cnn_output_length = conv_output_shape(cnn_output_length, kernel_size=2, stride=2)            # pool1
        cnn_output_length = conv_output_shape(cnn_output_length, kernel_size=3, stride=1, padding=1)  # conv2
        cnn_output_length = conv_output_shape(cnn_output_length, kernel_size=2, stride=2)            # pool2
        cnn_output_length = conv_output_shape(cnn_output_length, kernel_size=3, stride=1, padding=1)  # conv3
        cnn_output_length = conv_output_shape(cnn_output_length, kernel_size=2, stride=2)            # pool3

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
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        batch_size = x.size(0)

        # ===== CNN =====
        x_cnn = x.permute(0, 2, 1)

        x_cnn = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.dropout_cnn1(x_cnn)

        x_cnn = self.pool2(self.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.dropout_cnn2(x_cnn)

        x_cnn = self.pool3(self.relu(self.bn3(self.conv3(x_cnn))))
        x_cnn = self.dropout_cnn3(x_cnn)

        cnn_output = x_cnn.view(batch_size, -1)

        # ===== GRU =====
        x_gru = x
        x_gru, _ = self.gru1(x_gru)
        x_gru, _ = self.gru2(x_gru)
        gru_output = x_gru[:, -1, :]

        # ===== CONCAT =====
        concatenated = torch.cat([cnn_output, gru_output], dim=1)

        # ===== MLP =====
        x = self.dense1(concatenated)
        if x.shape[0] > 1:  # BatchNorm yêu cầu batch_size > 1
            x = self.bn_mlp1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        if x.shape[0] > 1:
            x = self.bn_mlp2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        out = self.output(x)
        return out


def build_cnn_gru_model(input_shape, num_classes=2):
    """Hàm tiện ích để khởi tạo model CNN-GRU."""
    model = CNN_GRU_Model(input_shape, num_classes)
    print(f"\n✅ Khởi tạo mô hình CNN-GRU thành công")
    print(f"   - Kích thước input: {input_shape}")
    print(f"   - Số lớp (num_classes): {num_classes}")
    return model


# ============================================================================
# 💡 BƯỚC 3: ĐỊNH NGHĨA CLIENT FEDERATED 💡
# ============================================================================

class FederatedClient:
    """
    Mỗi client có dữ liệu riêng và huấn luyện mô hình local.
    """
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        device: str = 'cpu'
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(device)

    def get_model_params(self) -> OrderedDict:
        """Lấy tham số mô hình của client."""
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, params: OrderedDict):
        """Cập nhật tham số cho mô hình client."""
        self.model.load_state_dict(params)

    def train_fedavg(
        self,
        epochs: int,
        learning_rate: float = 0.01,
        verbose: int = 1
    ) -> Dict:
        """
        Huấn luyện local với FedAvg.
        Có progress bar chi tiết cho từng batch.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            if verbose:
                pbar = tqdm(
                    self.train_loader,
                    desc=f"[Client {self.client_id}] FedAvg Epoch {epoch+1}/{epochs}",
                    unit="batch",
                    leave=False
                )
            else:
                pbar = self.train_loader

            for batch_idx, (data, target) in enumerate(pbar):
                batch_start = time.time()

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start

                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)

                if verbose:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                        "bt": f"{batch_time*1000:.0f}ms"
                    })

            avg_loss = epoch_loss / max(1, epoch_samples)
            total_loss += epoch_loss
            total_samples += epoch_samples

            if verbose:
                print(f"\nClient {self.client_id} - Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        avg_total_loss = total_loss / max(1, total_samples)

        return {
            'client_id': self.client_id,
            'num_samples': total_samples // max(1, epochs),
            'loss': avg_total_loss
        }

    def train_fedprox(
        self,
        epochs: int,
        global_params: OrderedDict,
        mu: float = 0.01,
        learning_rate: float = 0.01,
        verbose: int = 0
    ) -> Dict:
        """
        Train model với FedProx:
        loss = CE + (mu/2) * ||w - w_global||^2
        Có progress bar hiển thị CE loss + Prox term.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            if verbose:
                pbar = tqdm(
                    self.train_loader,
                    desc=f"[Client {self.client_id}] FedProx Epoch {epoch+1}/{epochs}",
                    unit="batch",
                    leave=False
                )
            else:
                pbar = self.train_loader

            for batch_idx, (data, target) in enumerate(pbar):
                batch_start = time.time()

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)

                # Loss chuẩn (cross entropy)
                ce_loss = criterion(output, target)

                # Proximal Term
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        global_param = global_params[name].to(self.device)
                        proximal_term += torch.sum((param - global_param) ** 2)

                proximal_term = (mu / 2) * proximal_term
                loss = ce_loss + proximal_term

                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start

                epoch_loss += ce_loss.item() * data.size(0)  # chỉ log CE
                epoch_samples += data.size(0)

                if verbose:
                    prox_val = float(proximal_term.detach().cpu().item())
                    pbar.set_postfix({
                        "ce": f"{ce_loss.item():.4f}",
                        "prox": f"{prox_val:.2e}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                        "bt": f"{batch_time*1000:.0f}ms"
                    })

            avg_loss = epoch_loss / max(1, epoch_samples)
            total_loss += epoch_loss
            total_samples += epoch_samples

            if verbose:
                print(f"\nClient {self.client_id} - Epoch {epoch+1}/{epochs}, Avg CE Loss: {avg_loss:.4f}")

        avg_total_loss = total_loss / max(1, total_samples)

        return {
            'client_id': self.client_id,
            'num_samples': total_samples // max(1, epochs),
            'loss': avg_total_loss
        }

    def evaluate(self) -> Dict:
        """Đánh giá model trên test set của client."""
        if self.test_loader is None:
            return {'accuracy': 0.0, 'loss': 0.0, 'num_samples': 0}

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'num_samples': total
        }


# ============================================================================
# 💡 BƯỚC 4: ĐỊNH NGHĨA SERVER FEDERATED 💡
# ============================================================================

class FederatedServer:
    """
    Server quản lý global model và thực hiện aggregation.
    """
    def __init__(
        self,
        model: nn.Module,
        clients: List[FederatedClient],
        client_test_loaders: List[DataLoader],
        device: str = 'cpu'
    ):
        self.global_model = model
        self.clients = clients
        self.client_test_loaders = client_test_loaders
        self.device = device
        self.global_model.to(device)

        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': []
        }

    def get_global_params(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())

    def set_global_params(self, params: OrderedDict):
        self.global_model.load_state_dict(params)

    def distribute_model(self, client_list: List[FederatedClient]):
        """Gửi tham số mô hình toàn cục xuống các client được chọn."""
        global_params = self.get_global_params()
        for client in client_list:
            client.set_model_params(global_params)

    def aggregate_fedavg(self, client_results: List[Dict]) -> OrderedDict:
        """
        FedAvg aggregation (Fix dtype & BatchNorm).
        """
        total_samples = sum(result['num_samples'] for result in client_results)

        aggregated_params = self.get_global_params()

        # Đặt tất cả các tham số float về 0
        for key in aggregated_params.keys():
            if aggregated_params[key].dtype in [torch.float32, torch.float64, torch.float16]:
                aggregated_params[key] = torch.zeros_like(aggregated_params[key])

        # Weighted sum (chỉ cho các tham số float)
        for result in client_results:
            client_id = result['client_id']
            num_samples = result['num_samples']
            weight = num_samples / max(1, total_samples)

            client = self.clients[client_id]
            client_params = client.get_model_params()

            for key in aggregated_params.keys():
                param = client_params[key]
                if param.dtype in [torch.float32, torch.float64, torch.float16]:
                    weight_tensor = torch.tensor(weight, dtype=param.dtype, device=param.device)
                    if aggregated_params[key].device != param.device:
                        aggregated_params[key] = aggregated_params[key].to(param.device)

                    aggregated_params[key] += weight_tensor * param
                else:
                    # Giữ lại giá trị của client đầu tiên
                    if client_id == client_results[0]['client_id']:
                        aggregated_params[key] = param

        return aggregated_params

    def train_round_fedavg(
        self,
        num_epochs: int,
        learning_rate: float = 0.01,
        client_fraction: float = 1.0,
        verbose: int = 1
    ) -> Dict:
        """
        Thực hiện 1 round huấn luyện với FedAvg. (Tuần tự)
        """
        num_selected = max(1, int(len(self.clients) * client_fraction))
        selected_clients = np.random.choice(self.clients, num_selected, replace=False)

        if verbose:
            print(f"→ [Round] Chọn {len(selected_clients)} client để huấn luyện...")

        self.distribute_model(selected_clients)

        client_results = []
        for idx, client in enumerate(selected_clients):
            if verbose:
                num_batches = len(client.train_loader)
                print(f"\n→ Training Client {client.client_id} ({idx+1}/{num_selected}) - "
                      f"{len(client.train_loader.dataset):,} samples, {num_batches} batches")

            result = client.train_fedavg(
                epochs=num_epochs,
                learning_rate=learning_rate,
                verbose=verbose
            )
            client_results.append(result)
            if verbose:
                print(f"   ✓ Client {client.client_id} completed - Avg Loss: {result['loss']:.4f}")

        if verbose:
            print(f"\n→ [Round] Đang tổng hợp (aggregating) {len(client_results)} mô hình...")

        aggregated_params = self.aggregate_fedavg(client_results)
        self.set_global_params(aggregated_params)

        avg_loss = float(np.mean([r['loss'] for r in client_results])) if client_results else 0.0

        if verbose:
            print(f"→ [Round] Hoàn thành, Loss trung bình (train): {avg_loss:.4f}")

        return {'train_loss': avg_loss, 'num_clients': len(selected_clients)}

    def train_round_fedprox(
        self,
        num_epochs: int,
        mu: float = 0.01,
        learning_rate: float = 0.01,
        client_fraction: float = 1.0,
        verbose: int = 0
    ) -> Dict:
        """
        Thực hiện 1 round huấn luyện với FedProx. (Tuần tự)
        """
        num_selected = max(1, int(len(self.clients) * client_fraction))
        selected_clients = np.random.choice(self.clients, num_selected, replace=False)

        if verbose:
            print(f"→ [Round] Chọn {len(selected_clients)} client để huấn luyện (FedProx)...")

        global_params = self.get_global_params()
        self.distribute_model(selected_clients)

        client_results = []
        for idx, client in enumerate(selected_clients):
            if verbose:
                num_batches = len(client.train_loader)
                print(f"\n→ Training Client {client.client_id} ({idx+1}/{num_selected}) - "
                      f"{len(client.train_loader.dataset):,} samples, {num_batches} batches")

            result = client.train_fedprox(
                epochs=num_epochs,
                global_params=global_params,
                mu=mu,
                learning_rate=learning_rate,
                verbose=verbose
            )
            client_results.append(result)
            if verbose:
                print(f"   ✓ Client {client.client_id} completed - Avg CE Loss: {result['loss']:.4f}")

        if verbose:
            print(f"\n→ [Round] Đang tổng hợp (aggregating) {len(client_results)} mô hình...")

        aggregated_params = self.aggregate_fedavg(client_results)
        self.set_global_params(aggregated_params)

        avg_loss = float(np.mean([r['loss'] for r in client_results])) if client_results else 0.0

        if verbose:
            print(f"→ [Round] Hoàn thành, Loss CE trung bình (train): {avg_loss:.4f}")

        return {'train_loss': avg_loss, 'num_clients': len(selected_clients)}

    def evaluate_global(self) -> Dict:
        """
        Đánh giá global model trên TẤT CẢ test set của client (lặp qua từng client),
        có progress bar chi tiết.
        """
        if self.client_test_loaders is None:
            print("⚠️  [Server Evaluate] Không tìm thấy client_test_loaders.")
            return {'accuracy': 0.0, 'loss': 0.0}

        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for loader_idx, loader in enumerate(self.client_test_loaders):
            pbar = tqdm(
                loader,
                desc=f"[Eval] Client TestLoader {loader_idx}",
                unit="batch",
                leave=False
            )
            with torch.no_grad():
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    loss = criterion(output, target)

                    total_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += data.size(0)

                    if total > 0:
                        current_acc = correct / total
                        current_loss = total_loss / total
                        pbar.set_postfix({
                            "acc": f"{current_acc*100:.2f}%",
                            "loss": f"{current_loss:.4f}"
                        })

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }


# ============================================================================
# 💡 BƯỚC 5: HÀM TỰ ĐỘNG PHÁT HIỆN THAM SỐ DỮ LIỆU 💡
# ============================================================================

def auto_detect_data_parameters(data_dir, num_clients):
    """
    Tự động phát hiện input_shape và num_classes.
    """
    print("\n" + "="*80)
    print("📂 TỰ ĐỘNG PHÁT HIỆN THAM SỐ DỮ LIỆU")
    print("="*80)
    print(f"→ Thư mục dữ liệu: {data_dir}")
    print(f"→ Số lượng client (dự kiến): {num_clients}")

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
            print(f"\n✅ Thông tin từ client 0:")
            print(f"   - Số đặc trưng (INPUT_FEATURES): {input_features}")
            print(f"   - input_shape: {input_shape}")

        print(f"\n→ Đang quét dữ liệu của {num_clients} client để thống kê nhãn...")
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
                unique_labels, counts = np.unique(y_train, return_counts=True)
                total_train += len(x_train)
                total_test += len(x_test)

                data_stats[i] = {
                    'train_samples': int(len(x_train)),
                    'test_samples': int(len(x_test)),
                    'unique_labels': int(len(unique_labels)),
                    'label_distribution': {str(k): int(v) for k, v in zip(unique_labels, counts)}
                }
                print(f"   - Client {i}: {len(x_train)} mẫu train, {len(x_test)} mẫu test, {len(unique_labels)} nhãn")

        combined_labels = np.concatenate(all_labels)
        num_classes = len(np.unique(combined_labels))

        print("\n📊 Tổng hợp toàn bộ dữ liệu:")
        print(f"   - Số lớp (num_classes): {num_classes}")
        print(f"   - Tổng số mẫu train: {total_train:,}")
        print(f"   - Tổng số mẫu test:  {total_test:,}")
        print("="*80)

        return input_shape, num_classes, data_stats

    except FileNotFoundError as e:
        print("\n" + "="*80 +
              f"\n❌ LỖI: KHÔNG TÌM THẤY TỆP DỮ LIỆU\nĐường dẫn: {e.filename}\n" + "="*80)
        raise
    except KeyError as e:
        print("\n" + "="*80 +
              f"\n❌ LỖI: THIẾU KEY TRONG FILE .NPZ\nKey: {e}\n" + "="*80)
        raise


# ============================================================================
# 💡 BƯỚC 6: HÀM LOAD DỮ LIỆU 💡
# ============================================================================

class NumpyDataset(TensorDataset):
    """Dataset tiện dụng để wrap numpy array thành TensorDataset."""
    def __init__(self, X, y, device='cpu'):
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
    print("\n" + "="*80)
    print("📥 LOADING FEDERATED DATA")
    print("="*80)
    print(f"→ Thiết bị hiện dùng: {device}")
    print(f"→ Số lượng client: {num_clients}\n")

    train_loaders = []
    test_loaders = []

    for client_id in range(num_clients):
        data_path = os.path.join(data_dir, f'client_{client_id}_data.npz')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy dữ liệu của client {client_id} tại: {data_path}")

        data = np.load(data_path)
        X_train = data['X_train']; y_train = data['y_train']
        X_test = data['X_test']; y_test = data['y_test']

        print(f"   - Client {client_id}: X_train {X_train.shape}, X_test {X_test.shape}")

        train_dataset = NumpyDataset(X_train, y_train, device)
        test_dataset = NumpyDataset(X_test, y_test, device)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    print(f"\n✅ Đã load dữ liệu cho {num_clients} client.")
    print("="*80)

    return train_loaders, test_loaders


# ============================================================================
# 💡 BƯỚC 7: HÀM KHỞI TẠO HỆ THỐNG 💡
# ============================================================================

def initialize_federated_system(
    train_loaders,
    test_loaders,
    input_shape,
    num_classes,
    device='cpu'
):
    """
    Khởi tạo global model, clients, và server
    """
    print("\n" + "="*80)
    print("🏗️  INITIALIZING FEDERATED SYSTEM")
    print("="*80)

    num_clients = len(train_loaders)

    # Tạo global model
    print(f"\n→ Khởi tạo mô hình toàn cục (global model)...")
    print(f"   - Input shape: {input_shape}")
    print(f"   - Số lớp: {num_classes}")
    print(f"   - Thiết bị: {device}")

    global_model = build_cnn_gru_model(input_shape, num_classes)
    global_model = global_model.to(device)
    print(f"   - Mô hình đã được chuyển sang thiết bị: {device}")

    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)

    if device_type == 'cuda':
        print(f"   - Xác nhận tham số đầu tiên của mô hình đang ở: {next(global_model.parameters()).device}")

    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"   - Tổng số tham số: {total_params:,}")
    print(f"   - Số tham số trainable: {trainable_params:,}")

    # Tạo clients
    print(f"\n→ Khởi tạo {num_clients} client...")
    clients = []

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
        print(f"   - Client {client_id}: khởi tạo thành công trên thiết bị {device}")

    print("\n→ Gán danh sách test loader cho server (tránh lỗi RAM)...")

    # Tạo server
    print("\n→ Khởi tạo server...")
    server = FederatedServer(
        model=global_model,
        clients=clients,
        client_test_loaders=test_loaders,
        device=device
    )

    print("\n✅ Hệ thống Federated đã được khởi tạo hoàn chỉnh.")
    print("="*80)

    return server, clients


# ============================================================================
# 💡 BƯỚC 8: CÁC HÀM HỖ TRỢ MULTIPROCESSING 💡
# ============================================================================

def _client_training_worker(args_tuple):
    """
    Hàm worker (helper) để chạy trong một process riêng biệt.
    Có tqdm riêng cho từng worker.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from collections import OrderedDict
    from tqdm.auto import tqdm as _tqdm
    import time as _time

    class CNN_GRU_Model_Worker(nn.Module):
        def __init__(self, input_shape, num_classes=2):
            super(CNN_GRU_Model_Worker, self).__init__()
            if isinstance(input_shape, tuple):
                seq_length = input_shape[0]
            else:
                seq_length = input_shape
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(64)
            self.pool1 = nn.MaxPool1d(2)
            self.dropout_cnn1 = nn.Dropout(0.2)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm1d(128)
            self.pool2 = nn.MaxPool1d(2)
            self.dropout_cnn2 = nn.Dropout(0.2)
            self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.pool3 = nn.MaxPool1d(2)
            self.dropout_cnn3 = nn.Dropout(0.3)

            def conv_output_shape(L_in, kernel_size=1, stride=1, padding=0, dilation=1):
                return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

            cnn_output_length = seq_length
            for _ in range(3):
                cnn_output_length = conv_output_shape(cnn_output_length, kernel_size=2, stride=2)
            self.cnn_output_size = 256 * cnn_output_length
            self.gru1 = nn.GRU(1, 128, batch_first=True)
            self.gru2 = nn.GRU(128, 64, batch_first=True)
            self.gru_output_size = 64
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
            if len(x.shape) == 2:
                x = x.unsqueeze(-1)
            batch_size = x.size(0)
            x_cnn = x.permute(0, 2, 1)
            x_cnn = self.dropout_cnn1(self.pool1(self.relu(self.bn1(self.conv1(x_cnn)))))
            x_cnn = self.dropout_cnn2(self.pool2(self.relu(self.bn2(self.conv2(x_cnn)))))
            x_cnn = self.dropout_cnn3(self.pool3(self.relu(self.bn3(self.conv3(x_cnn)))))
            cnn_output = x_cnn.view(batch_size, -1)
            x_gru = x
            x_gru, _ = self.gru1(x_gru)
            x_gru, _ = self.gru2(x_gru)
            gru_output = x_gru[:, -1, :]
            concatenated = torch.cat([cnn_output, gru_output], dim=1)
            x = self.dense1(concatenated)
            if x.shape[0] > 1:
                x = self.bn_mlp1(x)
            x = self.relu(x)
            x = self.dropout1(x)
            x = self.dense2(x)
            if x.shape[0] > 1:
                x = self.bn_mlp2(x)
            x = self.relu(x)
            x = self.dropout2(x)
            return self.output(x)

    class NumpyDataset_Worker(TensorDataset):
        def __init__(self, X, y):
            if len(X.shape) == 3:
                X = X.squeeze(-1)
            X = X.astype(np.float32)
            X_tensor = torch.from_numpy(X)
            y_tensor = torch.from_numpy(y).long()
            super().__init__(X_tensor, y_tensor)

    try:
        (client_id, model_state_dict, train_data, device_id, config) = args_tuple

        num_epochs = config['local_epochs']
        learning_rate = config['learning_rate']
        algorithm = config['algorithm']
        mu = config['mu']
        batch_size = config['batch_size']

        if device_id != 'cpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')

        X_train, y_train = train_data

        train_dataset = NumpyDataset_Worker(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        model = CNN_GRU_Model_Worker(config['input_shape'], config['num_classes'])
        model.load_state_dict(model_state_dict)
        model = model.to(device)

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            pbar = _tqdm(
                train_loader,
                desc=f"[Worker Client {client_id}] Epoch {epoch+1}/{num_epochs}",
                unit="batch",
                leave=False
            )

            for data, target in pbar:
                batch_start = _time.time()

                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                ce_loss = criterion(output, target)

                if algorithm == 'fedprox' and model_state_dict is not None:
                    proximal_term = 0.0
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
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "bt": f"{batch_time*1000:.0f}ms"
                })

            total_loss += epoch_loss
            total_samples += epoch_samples

        avg_loss = total_loss / max(1, total_samples)

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
    FedAvg aggregation từ client results (Chạy trên CPU)
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

            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                weight_tensor = torch.tensor(weight, dtype=param.dtype, device=param.device)
                aggregated_params[key] += weight_tensor * param
            else:
                if result['client_id'] == client_results[0]['client_id']:
                    aggregated_params[key] = param

    return aggregated_params


def train_round_multiprocessing(
    server,
    config,
    train_loaders,
    device='cuda'
):
    """
    Train 1 round với multiprocessing - chạy nhiều client song song.
    Có tqdm cho danh sách clients.
    """
    global_state_dict = {k: v.cpu() for k, v in server.get_global_params().items()}

    client_data = []
    for client_id, train_loader in enumerate(train_loaders):
        X_list, y_list = [], []
        for X_batch, y_batch in train_loader:
            X_list.append(X_batch.cpu().numpy())
            y_list.append(y_batch.cpu().numpy())
        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)
        client_data.append((X_train, y_train))

    if device == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        device_ids = [i % num_gpus for i in range(config['num_clients'])]
        print(f"   • Phân bổ {config['num_clients']} clients cho {num_gpus} GPUs.")
    else:
        device_ids = [0 if device == 'cuda' else 'cpu'] * config['num_clients']

    args_list = [
        (
            client_id,
            global_state_dict,
            client_data[client_id],
            device_ids[client_id],
            config
        )
        for client_id in range(config['num_clients'])
    ]

    print(f"   • Bắt đầu train {config['num_clients']} clients song song với {config['num_processes']} processes...")

    mp_context = mp.get_context('spawn')
    results = []
    with mp_context.Pool(processes=config['num_processes']) as pool:
        for res in tqdm(
            pool.imap_unordered(_client_training_worker, args_list),
            total=len(args_list),
            desc="Clients (multiprocessing)",
            unit="client"
        ):
            results.append(res)

    results = [r for r in results if r is not None]

    if len(results) == 0:
        raise RuntimeError("Tất cả clients đều thất bại!")

    print(f"   • Đang aggregate models từ {len(results)} clients...")
    aggregated_params_cpu = aggregate_models_fedavg_parallel(results)

    aggregated_params_gpu = OrderedDict(
        (k, v.to(device)) for k, v in aggregated_params_cpu.items()
    )
    server.set_global_params(aggregated_params_gpu)

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
    Có tqdm cho vòng lặp rounds.
    """
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN FEDERATED")
    print("="*80)

    algorithm = config['algorithm']
    num_rounds = config['num_rounds']
    local_epochs = config['local_epochs']
    learning_rate = config['learning_rate']
    client_fraction = config['client_fraction']
    eval_every = config['eval_every']
    device = config['device']
    use_multiprocessing = config.get('use_multiprocessing', False)

    history = server.history

    print(f"\n📋 Cấu hình huấn luyện:")
    print(f"   - Thuật toán: {algorithm.upper()}")
    print(f"   - Số round: {num_rounds}")
    print(f"   - Số epoch local: {local_epochs}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Tỉ lệ client mỗi round: {client_fraction}")
    print(f"   - Thiết bị: {device}")
    print(f"   - Chạy song song (Multiprocessing): {use_multiprocessing}")
    if use_multiprocessing:
        print(f"   - Số Processes: {config['num_processes']}")
    if algorithm == 'fedprox':
        print(f"   - Mu (proximal term): {config['mu']}")

    if eval_every > 0:
        print("\n📊 Đánh giá mô hình toàn cục (chưa huấn luyện)...")
        eval_result = server.evaluate_global()
        history['train_loss'].append(None)
        history['test_accuracy'].append(eval_result['accuracy'])
        history['test_loss'].append(eval_result['loss'])
        print(f"✅ Round 0 (Init): Test Acc: {eval_result['accuracy']*100:.2f}% | Test Loss: {eval_result['loss']:.4f}")

    round_iter = tqdm(
        range(num_rounds),
        desc="Global Rounds",
        unit="round"
    )

    for round_idx in round_iter:
        print(f"\n{'='*60}")
        print(f"📍 ROUND {round_idx+1}/{num_rounds}")
        print(f"{'='*60}")

        if use_multiprocessing:
            if train_loaders is None:
                raise ValueError("train_loaders là bắt buộc khi dùng multiprocessing")

            worker_config = config.copy()
            worker_config['local_epochs'] = local_epochs
            worker_config['learning_rate'] = learning_rate
            worker_config['algorithm'] = algorithm
            worker_config['mu'] = config.get('mu', 0.01)

            round_result = train_round_multiprocessing(
                server=server,
                config=worker_config,
                train_loaders=train_loaders,
                device=device
            )

        else:
            print("📝 Chế độ: SEQUENTIAL - Clients chạy lần lượt (chậm)...")
            if algorithm == 'fedavg':
                round_result = server.train_round_fedavg(
                    num_epochs=local_epochs,
                    learning_rate=learning_rate,
                    client_fraction=client_fraction,
                    verbose=1
                )
            elif algorithm == 'fedprox':
                round_result = server.train_round_fedprox(
                    num_epochs=local_epochs,
                    mu=config['mu'],
                    learning_rate=learning_rate,
                    client_fraction=client_fraction,
                    verbose=1
                )
            else:
                raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")

        if (round_idx + 1) % eval_every == 0:
            print(f"\n📊 Đang đánh giá mô hình toàn cục trên test set...")
            eval_result = server.evaluate_global()

            history['train_loss'].append(round_result['train_loss'])
            history['test_accuracy'].append(eval_result['accuracy'])
            history['test_loss'].append(eval_result['loss'])

            print(f"\n✅ Round {round_idx+1}/{num_rounds} Summary:")
            print(f"   • Train Loss (Avg): {round_result['train_loss']:.4f}")
            print(f"   • Test Accuracy: {eval_result['accuracy']*100:.2f}%")
            print(f"   • Test Loss: {eval_result['loss']:.4f}")

            round_iter.set_postfix({
                "algo": algorithm,
                "train_loss": f"{round_result['train_loss']:.4f}",
                "test_acc": f"{eval_result['accuracy']*100:.2f}%"
            })

    if history['test_accuracy']:
        print(f"\n✅ Huấn luyện {algorithm.upper()} hoàn tất.")
        print(f"   → Độ chính xác cuối cùng trên test: {history['test_accuracy'][-1]*100:.2f}%")
    else:
        print("\n⚠️ Không có kết quả test nào được ghi nhận.")

    return history


# ============================================================================
# 💡 BƯỚC 10: HÀM VẼ BIỂU ĐỒ & LƯU KẾT QUẢ 💡
# ============================================================================

def plot_training_history(history, save_path):
    """
    Vẽ biểu đồ train_loss, test_loss và test_accuracy theo round.
    """
    print("\n" + "="*80)
    print("📊 ĐANG VẼ BIỂU ĐỒ KẾT QUẢ HUẤN LUYỆN")
    print("="*80)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        rounds = range(len(history['test_loss']))
        train_loss = history['train_loss']
        test_loss = history['test_loss']
        test_acc = history['test_accuracy']

        ax1 = axes[0]
        ax1.plot(rounds[1:], train_loss[1:], label='Train Loss (Trung bình Client)',
                 marker='o', linewidth=2)
        ax1.plot(rounds, test_loss, label='Test Loss (Toàn cục)',
                 marker='s', linewidth=2)
        ax1.set_xlabel('Round', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Train Loss & Test Loss', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(rounds)

        ax2 = axes[1]
        test_acc_pct = [acc * 100 for acc in test_acc]
        ax2.plot(rounds, test_acc_pct, label='Test Accuracy (%)',
                 marker='o', linewidth=2, color='green')
        ax2.set_xlabel('Round', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Độ chính xác trên Test Set Toàn cục', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.set_xticks(rounds)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu biểu đồ lịch sử huấn luyện tại: {save_path}")
        plt.show()

    except Exception as e:
        print(f"⚠️ Lỗi khi vẽ biểu đồ: {e}")
    finally:
        plt.close()
        print("="*80)


def evaluate_and_save_results(server, history, config, output_dir, data_stats, training_duration, start_time, end_time):
    """
    Đánh giá cuối cùng, lưu model, history, config và tạo các báo cáo.
    """
    print("\n" + "="*80)
    print("💾 BƯỚC 9 & 10: ĐÁNH GIÁ CUỐI CÙNG VÀ LƯU KẾT QUẢ")
    print("="*80)

    model_path = os.path.join(output_dir, 'global_model.pth')
    torch.save(server.global_model.state_dict(), model_path)
    print(f"✅ Đã lưu mô hình toàn cục: {model_path}")

    history_path = os.path.join(output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"✅ Đã lưu lịch sử huấn luyện: {history_path}")

    config_path = os.path.join(output_dir, 'config.json')
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device'])
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    print(f"✅ Đã lưu cấu hình: {config_path}")

    stats_path = os.path.join(output_dir, 'data_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(data_stats, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu thống kê dữ liệu: {stats_path}")

    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, plot_path)

    print("\n→ Đang tạo dự đoán (predictions) trên Test Set toàn cục...")
    all_y_true = []
    all_y_pred = []
    server.global_model.eval()

    for loader_idx, loader in enumerate(server.client_test_loaders):
        pbar = tqdm(
            loader,
            desc=f"[Predict] Client TestLoader {loader_idx}",
            unit="batch",
            leave=False
        )
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(server.device), target.to(server.device)
                output = server.global_model(data)
                pred = output.argmax(dim=1)

                all_y_true.append(target.cpu().numpy())
                all_y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    print("✅ Đã tạo dự đoán xong.")

    print("\n" + "="*80)
    print("📄 CLASSIFICATION REPORT")
    print("="*80)
    class_labels = [str(i) for i in range(config['num_classes'])]
    report = classification_report(
        y_true,
        y_pred,
        labels=range(config['num_classes']),
        target_names=class_labels,
        zero_division=0,
        digits=4
    )
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n" + "="*80 + "\n\n" + report)
    print(f"\n💾 Đã lưu report: {report_path}")

    print("\n→ Đang tạo ma trận nhầm lẫn (Confusion Matrix)...")
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
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu confusion matrix: {cm_path}")
    plt.show()
    plt.close()

    print("\n→ Đang lưu metrics chi tiết (F1, Precision, Recall)...")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(config['num_classes']),
        average=None,
        zero_division=0
    )
    detailed_metrics = {
        'class': class_labels,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }
    df_metrics = pd.DataFrame(detailed_metrics)
    csv_path = os.path.join(output_dir, "detailed_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"✅ Đã lưu detailed metrics: {csv_path}")

    print("\n" + "="*80)
    print("📊 TOP 5 CLASSES PERFORMANCE:")
    print("="*80)
    df_sorted = df_metrics.sort_values('f1_score', ascending=False)
    print("\nTop 5 Best Classes (by F1-score):")
    print(df_sorted.head(5).to_string(index=False))
    print("\nTop 5 Worst Classes (by F1-score):")
    print(df_sorted.tail(5).to_string(index=False))
    print("="*80)

    print("\n" + "="*80)
    print("📝 BƯỚC 10: TẠO SUMMARY REPORT")
    print("="*80)
    summary_path = os.path.join(OUTPUT_DIR, "SUMMARY_REPORT.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n" + " " * 20 + "FEDERATED LEARNING SUMMARY REPORT\n" +
                "="*80 + "\n\n")
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

    print(f"✅ Đã tạo summary report: {summary_path}")


# ============================================================================
# 💡 BƯỚC 11: HÀM MAIN 💡
# ============================================================================

def main():
    config = CONFIG
    start_time = datetime.now()

    print("="*80)
    print("🤖 FEDERATED LEARNING VỚI MÔ HÌNH CNN-GRU (IoT IDS)")
    print("="*80)

    try:
        device = check_and_setup_gpu(config)
        config['device'] = device

        input_shape, num_classes, data_stats = auto_detect_data_parameters(
            data_dir=config['data_dir'],
            num_clients=config['num_clients']
        )
        config['input_shape'] = input_shape
        config['num_classes'] = num_classes

        print("\n" + "="*80)
        print("⚙️  CẤU HÌNH CUỐI CÙNG")
        print("="*80)
        print(json.dumps(config, indent=2, default=str))
        print("="*80)

        train_loaders, test_loaders = load_federated_data(
            data_dir=config['data_dir'],
            num_clients=config['num_clients'],
            batch_size=config['batch_size'],
            device=config['device']
        )

        server, clients = initialize_federated_system(
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            input_shape=config['input_shape'],
            num_classes=config['num_classes'],
            device=config['device']
        )

        history = train_federated(server, config, train_loaders)

        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("🏁 HUẤN LUYỆN HOÀN TẤT!")
        print("="*80)
        print(f"⏱️  Thời gian huấn luyện: {training_duration:.2f} giây ({training_duration/60:.2f} phút)")

        evaluate_and_save_results(
            server, history, config,
            config['output_dir'], data_stats,
            training_duration, start_time, end_time
        )

        print("\n" + "="*80)
        print("🎉 🎉 🎉  HOÀN TẤT TẤT CẢ CÁC BƯỚC  🎉 🎉 🎉")
        print("="*80)

    except Exception as e:
        logger.error("\n❌ ĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH CHẠY SCRIPT")
        logger.error(f"Chi tiết lỗi: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
