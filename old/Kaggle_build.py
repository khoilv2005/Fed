################################################################################
#                                                                              #
#  KAGGLE/COLAB OPTIMIZED FEDERATED LEARNING                                   #
#  Tối ưu cho GPU T4, A100, L4 và các môi trường cloud                          #
#  Features: Auto GPU Detection, Mixed Precision, Dynamic Optimization          #
#                                                                              #
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp  # Mixed Precision
import numpy as np
import matplotlib.pyplot as plt
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
import warnings
warnings.filterwarnings('ignore')

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed, parallel_backend
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================

# 🎯 GPU DETECTION & OPTIMIZATION SYSTEM
# ============================================================================

class GPUOptimizer:
    """
    Auto-detect GPU type và optimize configuration accordingly
    """
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.optimization_config = self._get_optimization_config()
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU type và capabilities"""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'name': 'Unknown',
            'type': 'cpu',
            'memory_gb': 0,
            'tensor_cores': False,
            'compute_capability': (0, 0),
            'multiprocessor_count': 0,
            'optimal_batch_size': 512,
            'mixed_precision_benefit': 0.0
        }
        
        if not torch.cuda.is_available():
            return gpu_info
        
        # Get GPU name and properties
        gpu_name = torch.cuda.get_device_name(0)
        gpu_info['name'] = gpu_name
        gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Safely get multiprocessor_count (not available in all PyTorch versions)
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            if hasattr(gpu_props, 'multiprocessor_count'):
                gpu_info['multiprocessor_count'] = gpu_props.multiprocessor_count
            else:
                # Fallback: estimate based on GPU type
                gpu_info['multiprocessor_count'] = self._estimate_multiprocessors(gpu_name)
        except Exception:
            gpu_info['multiprocessor_count'] = 80  # Conservative fallback
        
        # Detect GPU type and capabilities
        gpu_name_lower = gpu_name.lower()
        
        if 'rtx 4060' in gpu_name_lower:
            gpu_info.update({
                'type': 'rtx_4060',
                'tensor_cores': True,
                'compute_capability': (8, 9),
                'optimal_batch_size': 512,
                'mixed_precision_benefit': 1.7
            })
        elif 'tesla t4' in gpu_name_lower:
            gpu_info.update({
                'type': 'tesla_t4',
                'tensor_cores': True,
                'compute_capability': (7, 5),
                'optimal_batch_size': 384,
                'mixed_precision_benefit': 1.8
            })
        elif 'tesla a100' in gpu_name_lower:
            gpu_info.update({
                'type': 'tesla_a100',
                'tensor_cores': True,
                'compute_capability': (8, 0),
                'optimal_batch_size': 1024,
                'mixed_precision_benefit': 2.0
            })
        elif 'tesla l4' in gpu_name_lower:
            gpu_info.update({
                'type': 'tesla_l4',
                'tensor_cores': True,
                'compute_capability': (8, 9),
                'optimal_batch_size': 512,
                'mixed_precision_benefit': 1.9
            })
        elif 'tesla p100' in gpu_name_lower:
            gpu_info.update({
                'type': 'tesla_p100',
                'tensor_cores': False,
                'compute_capability': (6, 0),
                'optimal_batch_size': 256,
                'mixed_precision_benefit': 0.3
            })
        elif 'rtx' in gpu_name_lower:
            # Generic RTX
            gpu_info.update({
                'type': 'rtx_generic',
                'tensor_cores': True,
                'compute_capability': (7, 5),
                'optimal_batch_size': 512,
                'mixed_precision_benefit': 1.5
            })
        else:
            # Unknown GPU - conservative settings
            gpu_info.update({
                'type': 'unknown',
                'tensor_cores': False,
                'optimal_batch_size': 256,
                'mixed_precision_benefit': 0.5
            })
        
        return gpu_info
    
    def _estimate_multiprocessors(self, gpu_name: str) -> int:
        """Estimate multiprocessor count based on GPU name"""
        gpu_name_lower = gpu_name.lower()
        
        if 'tesla a100' in gpu_name_lower:
            return 108
        elif 'tesla t4' in gpu_name_lower:
            return 80
        elif 'tesla l4' in gpu_name_lower:
            return 80
        elif 'tesla p100' in gpu_name_lower:
            return 56
        elif 'rtx 4090' in gpu_name_lower:
            return 128
        elif 'rtx 4080' in gpu_name_lower:
            return 112
        elif 'rtx 4070' in gpu_name_lower:
            return 96
        elif 'rtx 4060' in gpu_name_lower:
            return 80
        elif 'rtx 3090' in gpu_name_lower:
            return 104
        elif 'rtx 3080' in gpu_name_lower:
            return 104
        elif 'rtx 3070' in gpu_name_lower:
            return 88
        elif 'rtx 3060' in gpu_name_lower:
            return 80
        else:
            return 80  # Conservative estimate for unknown GPUs
    
    def _get_optimization_config(self) -> Dict:
        """Get optimization config based on GPU type"""
        gpu_type = self.gpu_info['type']
        
        base_config = {
            'use_mixed_precision': self.gpu_info['tensor_cores'],
            'batch_size': self.gpu_info['optimal_batch_size'],
            'num_workers': 2,
            'pin_memory': True,
            'gradient_accumulation_steps': 1,
            'learning_rate_scaling': 1.0
        }
        
        # GPU-specific optimizations
        if gpu_type == 'tesla_t4':
            base_config.update({
                'use_mixed_precision': True,  # MANDATORY for T4
                'batch_size': 384,
                'num_workers': 2,
                'gradient_accumulation_steps': 1,
                'learning_rate_scaling': 1.2
            })
        elif gpu_type == 'tesla_a100':
            base_config.update({
                'use_mixed_precision': True,
                'batch_size': 1024,
                'num_workers': 4,
                'gradient_accumulation_steps': 1,
                'learning_rate_scaling': 1.5
            })
        elif gpu_type == 'tesla_l4':
            base_config.update({
                'use_mixed_precision': True,
                'batch_size': 512,
                'num_workers': 3,
                'gradient_accumulation_steps': 1,
                'learning_rate_scaling': 1.3
            })
        elif gpu_type == 'tesla_p100':
            base_config.update({
                'use_mixed_precision': False,  # No Tensor Cores
                'batch_size': 256,
                'num_workers': 2,
                'gradient_accumulation_steps': 2,
                'learning_rate_scaling': 0.8
            })
        elif gpu_type == 'rtx_4060':
            base_config.update({
                'use_mixed_precision': True,
                'batch_size': 512,
                'num_workers': 2,
                'gradient_accumulation_steps': 1,
                'learning_rate_scaling': 1.0
            })
        
        # Memory-based adjustments
        if self.gpu_info['memory_gb'] < 8:
            base_config['batch_size'] = min(base_config['batch_size'], 128)
            base_config['gradient_accumulation_steps'] = 4
        elif self.gpu_info['memory_gb'] > 32:
            base_config['batch_size'] = min(base_config['batch_size'] * 2, 2048)
        
        return base_config
    
    def print_gpu_info(self):
        """Print detailed GPU information"""
        print("\n" + "="*80)
        print("🎯 GPU DETECTION & OPTIMIZATION REPORT")
        print("="*80)
        print(f"🔧 GPU Available: {self.gpu_info['available']}")
        if self.gpu_info['available']:
            print(f"📱 GPU Name: {self.gpu_info['name']}")
            print(f"🏷️  GPU Type: {self.gpu_info['type']}")
            print(f"💾 Memory: {self.gpu_info['memory_gb']:.1f} GB")
            print(f"⚡ Tensor Cores: {self.gpu_info['tensor_cores']}")
            print(f"🔢 Compute Capability: {self.gpu_info['compute_capability']}")
            print(f"🎯 Optimal Batch Size: {self.optimization_config['batch_size']}")
            print(f"🚀 Mixed Precision Benefit: {self.optimization_config['use_mixed_precision']}")
            print(f"📈 Expected Speedup: {self.gpu_info['mixed_precision_benefit']:.1f}x")
        print("="*80)
    
    def get_device(self) -> str:
        """Get optimal device string"""
        return 'cuda' if self.gpu_info['available'] else 'cpu'

# ============================================================================

# 🚀 OPTIMIZED CNN-GRU MODEL
# ============================================================================

class OptimizedCNN_GRU_Model(nn.Module):
    """
    CNN-GRU Model optimized for different GPU types
    """
    def __init__(self, input_shape, num_classes=2, gpu_optimizer=None):
        super(OptimizedCNN_GRU_Model, self).__init__()
        
        self.gpu_optimizer = gpu_optimizer
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
        else:
            seq_length = input_shape
        
        # Adaptive architecture based on GPU
        if gpu_optimizer and gpu_optimizer.gpu_info['memory_gb'] > 16:
            # Large memory GPUs - bigger model
            cnn_channels = [64, 128, 256, 512]
            gru_hidden = [256, 128]
            mlp_dims = [512, 256]
        else:
            # Standard model
            cnn_channels = [64, 128, 256]
            gru_hidden = [128, 64]
            mlp_dims = [256, 128]
        
        # CNN Module
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for i, out_channels in enumerate(cnn_channels):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(0.2 if i < len(cnn_channels)-1 else 0.3)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        # Calculate CNN output size
        def conv_output_shape(L_in, kernel_size=1, stride=1, padding=0, dilation=1):
            return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        
        cnn_output_length = seq_length
        for _ in cnn_channels:
            cnn_output_length = conv_output_shape(cnn_output_length, 3, 1, 1)  # conv
            cnn_output_length = conv_output_shape(cnn_output_length, 2, 2)      # pool
        
        self.cnn_output_size = cnn_channels[-1] * cnn_output_length
        
        # GRU Module
        self.gru_layers = nn.ModuleList()
        gru_input_size = 1
        
        for i, hidden_size in enumerate(gru_hidden):
            gru_layer = nn.GRU(gru_input_size, hidden_size, batch_first=True)
            self.gru_layers.append(gru_layer)
            gru_input_size = hidden_size
        
        self.gru_output_size = gru_hidden[-1]
        
        # MLP Module
        concat_size = self.cnn_output_size + self.gru_output_size
        
        mlp_layers = []
        in_dim = concat_size
        
        for i, out_dim in enumerate(mlp_dims):
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4 if i == 0 else 0.3)
            ])
            in_dim = out_dim
        
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        batch_size = x.size(0)
        
        # CNN forward
        x_cnn = x.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            x_cnn = conv_layer(x_cnn)
        cnn_output = x_cnn.view(batch_size, -1)
        
        # GRU forward
        x_gru = x
        for gru_layer in self.gru_layers:
            x_gru, _ = gru_layer(x_gru)
        gru_output = x_gru[:, -1, :]
        
        # Concatenate and MLP
        concatenated = torch.cat([cnn_output, gru_output], dim=1)
        output = self.mlp(concatenated)
        
        return output

# ============================================================================

# 🎯 OPTIMIZED FEDERATED CLIENT
# ============================================================================

class OptimizedFederatedClient:
    """
    Federated Client with GPU-specific optimizations
    """
    def __init__(self, client_id: int, model: nn.Module, train_data: tuple, 
                 gpu_optimizer: GPUOptimizer, device: str = 'cpu'):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.gpu_optimizer = gpu_optimizer
        self.device = device
        self.model.to(device)
        
        # Mixed precision setup
        self.use_mixed_precision = gpu_optimizer.optimization_config['use_mixed_precision']
        if self.use_mixed_precision:
            self.scaler = amp.GradScaler()
        
        # Performance tracking
        self.batch_times = []
        self.loss_history = []
    
    def create_data_loader(self, batch_size: int) -> DataLoader:
        """Create optimized data loader"""
        X_train, y_train = self.train_data
        
        # Convert to tensors
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train.astype(np.float32))
            y_train = torch.from_numpy(y_train).long()
        
        dataset = TensorDataset(X_train, y_train)
        
        # Optimized DataLoader settings
        loader_config = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': False,
            'pin_memory': self.gpu_optimizer.optimization_config['pin_memory'] and self.device == 'cuda',
            'num_workers': 0 if self.device == 'cuda' else self.gpu_optimizer.optimization_config['num_workers']
        }
        
        return DataLoader(dataset, **loader_config)
    
    def train_fedavg(self, epochs: int, learning_rate: float, 
                    global_params: Optional[OrderedDict] = None,
                    mu: float = 0.01, algorithm: str = 'fedavg') -> Dict:
        """
        Optimized training with mixed precision and performance tracking
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Get optimal batch size
        batch_size = self.gpu_optimizer.optimization_config['batch_size']
        gradient_accumulation_steps = self.gpu_optimizer.optimization_config['gradient_accumulation_steps']
        
        train_loader = self.create_data_loader(batch_size)
        
        total_loss = 0.0
        total_samples = 0
        batch_count = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            optimizer.zero_grad()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_start = time.time()
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Mixed precision forward pass
                if self.use_mixed_precision:
                    with amp.autocast():
                        output = self.model(data)
                        loss = criterion(output, target)
                        
                        # FedProx proximal term
                        if algorithm == 'fedprox' and global_params is not None:
                            proximal_term = 0.0
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    global_param = global_params[name].to(self.device)
                                    proximal_term += torch.sum((param - global_param) ** 2)
                            loss = loss + (mu / 2) * proximal_term
                    
                    # Mixed precision backward
                    scaled_loss = loss / gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard precision
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    if algorithm == 'fedprox' and global_params is not None:
                        proximal_term = 0.0
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                global_param = global_params[name].to(self.device)
                                proximal_term += torch.sum((param - global_param) ** 2)
                        loss = loss + (mu / 2) * proximal_term
                    
                    scaled_loss = loss / gradient_accumulation_steps
                    scaled_loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Performance tracking
                batch_time = time.time() - batch_start
                self.batch_times.append(batch_time)
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / max(1, epoch_samples)
            self.loss_history.append(avg_epoch_loss)
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Calculate metrics
        avg_total_loss = total_loss / max(1, total_samples)
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        samples_per_second = (total_samples / (sum(self.batch_times) if self.batch_times else 1))
        
        return {
            'client_id': self.client_id,
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'num_samples': total_samples,
            'loss': avg_total_loss,
            'avg_batch_time': avg_batch_time,
            'samples_per_second': samples_per_second,
            'total_batches': batch_count,
            'gpu_type': self.gpu_optimizer.gpu_info['type'],
            'mixed_precision': self.use_mixed_precision
        }

# ============================================================================

# 🚀 OPTIMIZED FEDERATED SERVER
# ============================================================================

class OptimizedFederatedServer:
    """
    Federated Server with GPU-specific optimizations and multiple training methods
    """
    def __init__(self, model_class, input_shape, num_classes, gpu_optimizer: GPUOptimizer):
        self.model_class = model_class
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.gpu_optimizer = gpu_optimizer
        self.device = gpu_optimizer.get_device()
        
        # Global model
        self.global_model = model_class(input_shape, num_classes, gpu_optimizer)
        self.global_model = self.global_model.to(self.device)
        
        # History
        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': [],
            'samples_per_second': [],
            'training_time': []
        }
    
    def get_global_params(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())
    
    def set_global_params(self, params: OrderedDict):
        self.global_model.load_state_dict(params)
    
    def train_round_sequential(self, client_data_list: List[tuple], 
                             config: Dict) -> Dict:
        """Sequential training optimized for stability"""
        print(f"→ [Sequential] Training {len(client_data_list)} clients...")
        
        global_params = self.get_global_params()
        results = []
        start_time = time.time()
        
        for client_id, train_data in enumerate(client_data_list):
            # Create client model
            client_model = self.model_class(self.input_shape, self.num_classes, self.gpu_optimizer)
            client_model.load_state_dict(global_params)
            
            client = OptimizedFederatedClient(
                client_id, client_model, train_data, self.gpu_optimizer, self.device
            )
            
            result = client.train_fedavg(
                epochs=config['local_epochs'],
                learning_rate=config['learning_rate'],
                global_params=global_params if config['algorithm'] == 'fedprox' else None,
                mu=config.get('mu', 0.01),
                algorithm=config['algorithm']
            )
            
            results.append(result)
            print(f"   ✓ Client {client_id} ({result['gpu_type']}) - "
                  f"Loss: {result['loss']:.4f}, "
                  f"Speed: {result['samples_per_second']:.1f} samples/s")
        
        # Aggregate results
        if results:
            aggregated_params = self._aggregate_fedavg(results)
            self.set_global_params(aggregated_params)
            
            training_time = time.time() - start_time
            avg_samples_per_second = np.mean([r['samples_per_second'] for r in results])
            
            return {
                'train_loss': float(np.mean([r['loss'] for r in results])),
                'num_clients': len(results),
                'training_time': training_time,
                'samples_per_second': avg_samples_per_second,
                'method': 'sequential'
            }
        else:
            raise RuntimeError("❌ Tất cả clients đều thất bại!")
    
    def train_round_joblib(self, client_data_list: List[tuple], 
                          config: Dict, n_jobs: int = 2) -> Dict:
        """Joblib parallel training"""
        if not JOBLIB_AVAILABLE:
            print("❌ Joblib không có sẵn. Chuyển sang sequential.")
            return self.train_round_sequential(client_data_list, config)
        
        print(f"→ [Joblib] Training {len(client_data_list)} clients with {n_jobs} workers...")
        
        global_params = self.get_global_params()
        start_time = time.time()
        
        # Prepare arguments for parallel execution
        args_list = [
            (client_id, global_params, train_data, config, self.gpu_optimizer, self.device)
            for client_id, train_data in enumerate(client_data_list)
        ]
        
        # Run parallel with joblib
        try:
            with parallel_backend('threading', n_jobs=n_jobs):
                results = Parallel()(
                    delayed(self._train_client_worker)(*args) for args in args_list
                )
            
            # Filter valid results
            valid_results = [r for r in results if r is not None]
            
            for result in valid_results:
                print(f"   ✓ Client {result['client_id']} ({result['gpu_type']}) - "
                      f"Loss: {result['loss']:.4f}, "
                      f"Speed: {result['samples_per_second']:.1f} samples/s")
            
            if valid_results:
                aggregated_params = self._aggregate_fedavg(valid_results)
                self.set_global_params(aggregated_params)
                
                training_time = time.time() - start_time
                avg_samples_per_second = np.mean([r['samples_per_second'] for r in valid_results])
                
                return {
                    'train_loss': float(np.mean([r['loss'] for r in valid_results])),
                    'num_clients': len(valid_results),
                    'training_time': training_time,
                    'samples_per_second': avg_samples_per_second,
                    'method': 'joblib'
                }
            else:
                raise RuntimeError("❌ Tất cả clients đều thất bại!")
                
        except Exception as e:
            print(f"❌ Joblib training failed: {e}")
            print("→ Chuyển sang sequential training...")
            return self.train_round_sequential(client_data_list, config)
    
    @staticmethod
    def _train_client_worker(client_id: int, global_params: OrderedDict, train_data: tuple,
                            config: Dict, gpu_optimizer: GPUOptimizer, device: str) -> Dict:
        """Worker function for parallel training"""
        try:
            # Import trong function để tránh pickling issues
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            import torch.cuda.amp as amp
            import numpy as np
            import time
            from collections import OrderedDict
            
            # Model class definition trong worker
            class OptimizedCNN_GRU_Model_Worker(nn.Module):
                def __init__(self, input_shape, num_classes=2, gpu_optimizer=None):
                    super(OptimizedCNN_GRU_Model_Worker, self).__init__()
                    
                    if isinstance(input_shape, tuple):
                        seq_length = input_shape[0]
                    else:
                        seq_length = input_shape
                    
                    # Standard model for worker
                    self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm1d(64)
                    self.pool1 = nn.MaxPool1d(2)
                    self.dropout1 = nn.Dropout(0.2)
                    
                    self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm1d(128)
                    self.pool2 = nn.MaxPool1d(2)
                    self.dropout2 = nn.Dropout(0.2)
                    
                    self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
                    self.bn3 = nn.BatchNorm1d(256)
                    self.pool3 = nn.MaxPool1d(2)
                    self.dropout3 = nn.Dropout(0.3)
                    
                    # Calculate CNN output size
                    def conv_output_shape(L_in, kernel_size=1, stride=1, padding=0, dilation=1):
                        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
                    
                    cnn_output_length = seq_length
                    cnn_output_length = conv_output_shape(cnn_output_length, 3, 1, 1)
                    cnn_output_length = conv_output_shape(cnn_output_length, 2, 2)
                    cnn_output_length = conv_output_shape(cnn_output_length, 3, 1, 1)
                    cnn_output_length = conv_output_shape(cnn_output_length, 2, 2)
                    cnn_output_length = conv_output_shape(cnn_output_length, 3, 1, 1)
                    cnn_output_length = conv_output_shape(cnn_output_length, 2, 2)
                    
                    self.cnn_output_size = 256 * cnn_output_length
                    
                    # GRU Module
                    self.gru1 = nn.GRU(1, 128, batch_first=True)
                    self.gru2 = nn.GRU(128, 64, batch_first=True)
                    self.gru_output_size = 64
                    
                    # MLP Module
                    concat_size = self.cnn_output_size + self.gru_output_size
                    self.dense1 = nn.Linear(concat_size, 256)
                    self.bn_mlp1 = nn.BatchNorm1d(256)
                    self.dropout4 = nn.Dropout(0.4)
                    self.dense2 = nn.Linear(256, 128)
                    self.bn_mlp2 = nn.BatchNorm1d(128)
                    self.dropout5 = nn.Dropout(0.3)
                    self.output = nn.Linear(128, num_classes)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    if len(x.shape) == 2:
                        x = x.unsqueeze(-1)
                    batch_size = x.size(0)
                    
                    # CNN
                    x_cnn = x.permute(0, 2, 1)
                    x_cnn = self.dropout1(self.pool1(self.relu(self.bn1(self.conv1(x_cnn)))))
                    x_cnn = self.dropout2(self.pool2(self.relu(self.bn2(self.conv2(x_cnn)))))
                    x_cnn = self.dropout3(self.pool3(self.relu(self.bn3(self.conv3(x_cnn)))))
                    cnn_output = x_cnn.view(batch_size, -1)
                    
                    # GRU
                    x_gru = x
                    x_gru, _ = self.gru1(x_gru)
                    x_gru, _ = self.gru2(x_gru)
                    gru_output = x_gru[:, -1, :]
                    
                    # Concat + MLP
                    concatenated = torch.cat([cnn_output, gru_output], dim=1)
                    x = self.dense1(concatenated)
                    if x.shape[0] > 1:
                        x = self.bn_mlp1(x)
                    x = self.relu(x)
                    x = self.dropout4(x)
                    x = self.dense2(x)
                    if x.shape[0] > 1:
                        x = self.bn_mlp2(x)
                    x = self.relu(x)
                    x = self.dropout5(x)
                    return self.output(x)
            
            # Force CPU for stability in parallel execution
            device = torch.device('cpu')
            
            # Create model and load global params
            model = OptimizedCNN_GRU_Model_Worker(config['input_shape'], config['num_classes'])
            model.load_state_dict(global_params)
            model = model.to(device)
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            # Mixed precision setup (disabled for CPU)
            use_mixed_precision = gpu_optimizer.optimization_config['use_mixed_precision'] and device.type == 'cuda'
            if use_mixed_precision:
                scaler = amp.GradScaler()
            
            # Create data loader
            X_train, y_train = train_data
            if isinstance(X_train, np.ndarray):
                X_train = torch.from_numpy(X_train.astype(np.float32))
                y_train = torch.from_numpy(y_train).long()
            
            dataset = TensorDataset(X_train, y_train)
            batch_size = gpu_optimizer.optimization_config['batch_size']
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            model.train()
            total_loss = 0.0
            total_samples = 0
            batch_times = []
            
            for epoch in range(config['local_epochs']):
                epoch_loss = 0.0
                epoch_samples = 0
                
                for data, target in train_loader:
                    batch_start = time.time()
                    
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # FedProx proximal term
                    if config['algorithm'] == 'fedprox' and global_params is not None:
                        proximal_term = 0.0
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                global_param = global_params[name].to(device)
                                proximal_term += torch.sum((param - global_param) ** 2)
                        loss = loss + (config.get('mu', 0.01) / 2) * proximal_term
                    
                    loss.backward()
                    optimizer.step()
                    
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    epoch_loss += loss.item() * data.size(0)
                    epoch_samples += data.size(0)
                
                total_loss += epoch_loss
                total_samples += epoch_samples
            
            avg_loss = total_loss / max(1, total_samples)
            avg_batch_time = np.mean(batch_times) if batch_times else 0
            samples_per_second = total_samples / (sum(batch_times) if batch_times else 1)
            
            return {
                'client_id': client_id,
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'num_samples': total_samples,
                'loss': avg_loss,
                'avg_batch_time': avg_batch_time,
                'samples_per_second': samples_per_second,
                'gpu_type': gpu_optimizer.gpu_info['type'],
                'mixed_precision': use_mixed_precision
            }
            
        except Exception as e:
            print(f"❌ Worker {client_id} Error: {e}")
            return None
    
    def _aggregate_fedavg(self, client_results: List[Dict]) -> OrderedDict:
        """FedAvg aggregation from client results"""
        if not client_results:
            raise ValueError("Không có client results để aggregate")
        
        total_samples = sum(result['num_samples'] for result in client_results)
        first_state = client_results[0]['model_state_dict']
        
        aggregated_params = OrderedDict()
        for key in first_state.keys():
            aggregated_params[key] = torch.zeros_like(first_state[key])
        
        # Weighted sum
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

# ============================================================================

# 🎯 MAIN KAGGLE/COLAB OPTIMIZED FUNCTION
# ============================================================================

def run_kaggle_optimized_federated_learning(data_dir: str, num_clients: int = 5, 
                                           num_rounds: int = 5, local_epochs: int = 5,
                                           algorithm: str = 'fedprox', method: str = 'auto',
                                           output_dir: str = './kaggle_results'):
    """
    Main function optimized for Kaggle/Colab environments
    
    Args:
        data_dir: Directory containing federated data
        num_clients: Number of clients
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs per round
        algorithm: 'fedavg' or 'fedprox'
        method: 'sequential', 'joblib', or 'auto'
    """
    
    print("="*80)
    print("🚀 KAGGLE/COLAB OPTIMIZED FEDERATED LEARNING")
    print("="*80)
    
    # 1. GPU Detection and Optimization
    gpu_optimizer = GPUOptimizer()
    gpu_optimizer.print_gpu_info()
    
    # 2. Auto-detect data parameters
    print("\n📂 Auto-detecting data parameters...")
    try:
        # Load client 0 data to detect parameters
        client_0_path = os.path.join(data_dir, "client_0_data.npz")
        if not os.path.exists(client_0_path):
            raise FileNotFoundError(f"Không tìm thấy file: {client_0_path}")
        
        with np.load(client_0_path) as data:
            x_train_sample = data['X_train']
            input_features = x_train_sample.shape[1]
            input_shape = (input_features,)
            
            # Detect num_classes from all clients
            all_labels = []
            for i in range(num_clients):
                client_path = os.path.join(data_dir, f"client_{i}_data.npz")
                with np.load(client_path) as client_data:
                    all_labels.append(client_data['y_train'])
            
            combined_labels = np.concatenate(all_labels)
            num_classes = len(np.unique(combined_labels))
        
        print(f"✅ Input shape: {input_shape}")
        print(f"✅ Num classes: {num_classes}")
        
    except Exception as e:
        print(f"❌ Error detecting data parameters: {e}")
        # Fallback to common values
        input_shape = (100,)  # Common feature size
        num_classes = 2       # Binary classification
    
    # 3. Load all client data
    print(f"\n📥 Loading data for {num_clients} clients...")
    client_data_list = []
    
    for client_id in range(num_clients):
        client_path = os.path.join(data_dir, f"client_{client_id}_data.npz")
        if not os.path.exists(client_path):
            print(f"⚠️ Client {client_id} data not found, skipping...")
            continue
        
        with np.load(client_path) as data:
            X_train = data['X_train']
            y_train = data['y_train']
            client_data_list.append((X_train, y_train))
            print(f"   ✓ Client {client_id}: {X_train.shape}")
    
    if len(client_data_list) == 0:
        raise RuntimeError("❌ No client data found!")
    
    # 4. Initialize optimized server
    print(f"\n🏗️ Initializing optimized federated server...")
    server = OptimizedFederatedServer(
        OptimizedCNN_GRU_Model, input_shape, num_classes, gpu_optimizer
    )
    
    # 5. Training configuration with GPU optimizations
    base_lr = 0.001
    lr_scaling = gpu_optimizer.optimization_config['learning_rate_scaling']
    
    config = {
        'algorithm': algorithm,
        'local_epochs': local_epochs,
        'learning_rate': base_lr * lr_scaling,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'mu': 0.01 if algorithm == 'fedprox' else 0.0
    }
    
    print(f"\n⚙️ Training Configuration:")
    print(f"   • Algorithm: {algorithm.upper()}")
    print(f"   • Learning Rate: {config['learning_rate']:.4f} (scaled by {lr_scaling:.1f}x)")
    print(f"   • Batch Size: {gpu_optimizer.optimization_config['batch_size']}")
    print(f"   • Mixed Precision: {gpu_optimizer.optimization_config['use_mixed_precision']}")
    print(f"   • GPU Type: {gpu_optimizer.gpu_info['type']}")
    
    # 6. Auto-select training method
    if method == 'auto':
        if gpu_optimizer.gpu_info['type'] in ['tesla_t4', 'tesla_l4'] and JOBLIB_AVAILABLE:
            method = 'joblib'
        else:
            method = 'sequential'
    
    print(f"   • Training Method: {method.upper()}")
    
    # 7. Training loop with performance tracking
    print(f"\n🚀 Starting {num_rounds} rounds of federated learning...")
    print("="*80)
    
    round_results = []
    
    for round_idx in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\n📍 ROUND {round_idx + 1}/{num_rounds}")
        print("-" * 60)
        
        round_start = time.time()
        
        # Train round with selected method
        if method == 'joblib':
            round_result = server.train_round_joblib(
                client_data_list, config, n_jobs=2
            )
        else:
            round_result = server.train_round_sequential(client_data_list, config)
        
        round_time = time.time() - round_start
        round_result['round_time'] = round_time
        
        # Store results
        round_results.append(round_result)
        server.history['train_loss'].append(round_result['train_loss'])
        server.history['samples_per_second'].append(round_result['samples_per_second'])
        server.history['training_time'].append(round_result['training_time'])
        
        # Print round summary
        print(f"\n✅ Round {round_idx + 1} Summary:")
        print(f"   • Train Loss: {round_result['train_loss']:.4f}")
        print(f"   • Samples/sec: {round_result['samples_per_second']:.1f}")
        print(f"   • Training Time: {round_result['training_time']:.2f}s")
        print(f"   • Total Round Time: {round_time:.2f}s")
        print(f"   • Method: {round_result['method']}")
        print(f"   • Clients: {round_result['num_clients']}/{len(client_data_list)}")
    
    # 8. Final performance summary
    print(f"\n" + "="*80)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("="*80)
    
    avg_samples_per_second = np.mean([r['samples_per_second'] for r in round_results])
    total_training_time = sum([r['training_time'] for r in round_results])
    final_loss = round_results[-1]['train_loss']
    
    print(f"🎯 GPU Type: {gpu_optimizer.gpu_info['type']}")
    print(f"🚀 Average Speed: {avg_samples_per_second:.1f} samples/second")
    print(f"⏱️  Total Training Time: {total_training_time:.2f}s")
    print(f"📉 Final Loss: {final_loss:.4f}")
    print(f"🔧 Mixed Precision: {gpu_optimizer.optimization_config['use_mixed_precision']}")
    print(f"📦 Batch Size: {gpu_optimizer.optimization_config['batch_size']}")
    print(f"🏃 Training Method: {method.upper()}")
    
    # 9. Performance comparison
    print(f"\n📈 PERFORMANCE INSIGHTS:")
    if gpu_optimizer.gpu_info['type'] == 'tesla_t4':
        if gpu_optimizer.optimization_config['use_mixed_precision']:
            print(f"   ✅ T4 is OPTIMIZED with mixed precision!")
            print(f"   🚀 Expected speedup: {gpu_optimizer.gpu_info['mixed_precision_benefit']:.1f}x vs FP32")
        else:
            print(f"   ⚠️ T4 NOT using mixed precision - missing {gpu_optimizer.gpu_info['mixed_precision_benefit']:.1f}x speedup!")
    
    elif gpu_optimizer.gpu_info['type'] == 'tesla_p100':
        print(f"   ℹ️ P100 has no Tensor Cores - mixed precision less effective")
        print(f"   📊 Performance limited by older architecture")
    
    # 10. Save results
    results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"kaggle_optimized_{timestamp}.json")
    
    summary = {
        'gpu_info': gpu_optimizer.gpu_info,
        'optimization_config': gpu_optimizer.optimization_config,
        'training_config': config,
        'round_results': round_results,
        'final_summary': {
            'avg_samples_per_second': avg_samples_per_second,
            'total_training_time': total_training_time,
            'final_loss': final_loss,
            'method': method
        },
        'timestamp': timestamp
    }
    
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {result_file}")
    print("="*80)
    print("🎉 KAGGLE/COLAB OPTIMIZED FEDERATED LEARNING COMPLETED!")
    print("="*80)
    
    return summary

# ============================================================================

# 🚀 QUICK START FUNCTION
# ============================================================================

def quick_start_demo():
    """
    Quick demo with dummy data for testing
    """
    print("🧪 Running quick demo with dummy data...")
    
    # Create dummy data directory
    dummy_dir = './dummy_federated_data'
    os.makedirs(dummy_dir, exist_ok=True)
    
    # Generate dummy federated data
    num_clients = 3
    input_features = 100
    num_classes = 2
    samples_per_client = 1000
    
    for client_id in range(num_clients):
        X_train = np.random.randn(samples_per_client, input_features).astype(np.float32)
        y_train = np.random.randint(0, num_classes, samples_per_client)
        X_test = np.random.randn(samples_per_client // 4, input_features).astype(np.float32)
        y_test = np.random.randint(0, num_classes, samples_per_client // 4)
        
        client_file = os.path.join(dummy_dir, f"client_{client_id}_data.npz")
        np.savez(client_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    print(f"✅ Created dummy data for {num_clients} clients")
    
    # Run optimized federated learning
    try:
        results = run_kaggle_optimized_federated_learning(
            data_dir=dummy_dir,
            num_clients=num_clients,
            num_rounds=3,
            local_epochs=2,
            algorithm='fedprox',
            method='auto'
        )
        
        print("\n🎉 Demo completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================

# 🚀 MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if running in Kaggle/Colab
    try:
        import google.colab
        print("🔍 Detected Google Colab environment")
    except ImportError:
        try:
            import kaggle_secrets
            print("🔍 Detected Kaggle environment")
        except ImportError:
            print("🔍 Running in local environment")
    
    # Run quick demo
    print("\n" + "="*80)
    print("🧪 KAGGLE/COLAB OPTIMIZATION DEMO")
    print("="*80)
    
    results = quick_start_demo()
    
    if results:
        print(f"\n✅ Demo successful! Check your GPU optimization results.")
        print(f"🚀 Average speed: {results['final_summary']['avg_samples_per_second']:.1f} samples/sec")
        print(f"🎯 GPU: {results['gpu_info']['type']}")
    else:
        print(f"\n❌ Demo failed. Check the error messages above.")
