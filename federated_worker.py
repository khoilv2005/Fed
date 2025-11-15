"""
Worker module for federated learning multiprocessing.
This file MUST be separate to work with spawn method in Jupyter/Kaggle.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm as _tqdm
import time as _time
import os


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


def client_training_worker(args_tuple):
    """
    Worker function for training a single client.
    MUST be at module level to be picklable by spawn.
    """
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    try:
        (client_id, model_state_dict, train_data, device_id, config) = args_tuple

        print(f"   🚀 Worker cho Client {client_id} đã start (device: {device_id})")

        num_epochs = config['local_epochs']
        learning_rate = config['learning_rate']
        algorithm = config['algorithm']
        mu = config['mu']
        batch_size = config['batch_size']

        # Thiết lập device cho worker process
        if device_id != 'cpu' and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if isinstance(device_id, int) and device_id < num_gpus:
                device = torch.device(f'cuda:{device_id}')
                torch.cuda.set_device(device)
            else:
                device = torch.device('cuda:0')
                torch.cuda.set_device(0)
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

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, total_samples)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'client_id': client_id,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'num_samples': len(X_train),
            'loss': avg_loss
        }

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ LỖI TRONG WORKER CLIENT {client_id}")
        print(f"{'='*60}")
        print(f"Device: {device_id}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return None
