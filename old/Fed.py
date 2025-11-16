import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import List, Dict, Tuple
import copy
from collections import OrderedDict
from tqdm.auto import tqdm  # progress bar

# Import CNN-GRU model từ model.py
from old.model import CNN_GRU_Model, build_cnn_gru_model


class FederatedClient:
    """
    Client trong Federated Learning
    Mỗi client có data riêng và train model local
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
        """Lấy toàn bộ state_dict của model"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_params(self, params: OrderedDict):
        """Set parameters cho model"""
        self.model.load_state_dict(params)
    
    def train_fedavg(
        self, 
        epochs: int, 
        learning_rate: float = 0.01,
        verbose: int = 1
    ) -> Dict:
        """
        Train model với FedAvg (optimizer: Adam)
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
                batch_iter = tqdm(
                    self.train_loader,
                    desc=f"[FedAvg] Client {self.client_id} - Epoch {epoch+1}/{epochs}",
                    leave=False
                )
            else:
                batch_iter = self.train_loader
            
            for batch_idx, (data, target) in enumerate(batch_iter):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
            
            avg_loss = epoch_loss / epoch_samples
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if verbose:
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        avg_total_loss = total_loss / total_samples
        
        return {
            'client_id': self.client_id,
            'num_samples': total_samples // epochs,
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
        Train model với FedProx (optimizer: Adam)
        Thêm proximal term: mu/2 * ||w - w_global||^2
        Chỉ tính trên learnable parameters (weight/bias), không dùng buffer.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Map global parameters theo tên, chỉ lấy learnable parameters
        global_param_dict = {
            name: global_params[name].clone().detach().to(self.device)
            for name, _ in self.model.named_parameters()
        }
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            if verbose:
                batch_iter = tqdm(
                    self.train_loader,
                    desc=f"[FedProx] Client {self.client_id} - Epoch {epoch+1}/{epochs}",
                    leave=False
                )
            else:
                batch_iter = self.train_loader
            
            for batch_idx, (data, target) in enumerate(batch_iter):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Loss chuẩn (cross entropy)
                ce_loss = criterion(output, target)
                
                # Proximal term: mu/2 * ||w - w_global||^2 trên learnable params
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    global_param = global_param_dict[name]
                    proximal_term += torch.sum((param - global_param) ** 2)
                proximal_term = (mu / 2) * proximal_term
                
                # Total loss
                loss = ce_loss + proximal_term
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += ce_loss.item() * data.size(0)  # Chỉ log CE loss
                epoch_samples += data.size(0)
            
            avg_loss = epoch_loss / epoch_samples
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if verbose:
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        avg_total_loss = total_loss / total_samples
        
        return {
            'client_id': self.client_id,
            'num_samples': total_samples // epochs,
            'loss': avg_total_loss
        }
    
    def evaluate(self) -> Dict:
        """Đánh giá model trên test set của client"""
        if self.test_loader is None:
            return {'accuracy': 0.0, 'loss': 0.0}
        
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


class FederatedServer:
    """
    Server trong Federated Learning
    Quản lý global model và thực hiện aggregation
    """
    def __init__(
        self,
        model: nn.Module,
        clients: List[FederatedClient],
        test_loader: DataLoader = None,
        client_test_loaders: List[DataLoader] = None,
        device: str = 'cpu'
    ):
        self.global_model = model
        self.clients = clients
        self.test_loader = test_loader
        self.client_test_loaders = client_test_loaders  # Hỗ trợ list test loaders
        self.device = device
        self.global_model.to(device)

        # History để track performance
        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': []
        }
    
    def get_global_params(self) -> OrderedDict:
        """Lấy global model parameters (full state_dict)"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def set_global_params(self, params: OrderedDict):
        """Set global model parameters"""
        self.global_model.load_state_dict(params)
    
    def distribute_model(self):
        """Gửi global model xuống tất cả clients"""
        global_params = self.get_global_params()
        for client in self.clients:
            client.set_model_params(global_params)
    
    def aggregate_fedavg(self, client_results: List[Dict]) -> OrderedDict:
        """
        FedAvg aggregation: weighted average theo số lượng samples
        🔥 Chỉ aggregate learnable parameters (weight/bias),
        KHÔNG aggregate BatchNorm running_mean / running_var / num_batches_tracked.
        """
        # Tính tổng số samples
        total_samples = sum(result['num_samples'] for result in client_results)
        
        # Lấy danh sách tên learnable parameters từ global model
        param_names = [name for name, _ in self.global_model.named_parameters()]
        
        # Khởi tạo dict aggregated cho learnable params
        # dùng shape từ global model (cùng shape với tất cả clients)
        global_state = self.global_model.state_dict()
        aggregated_learnable = {
            name: torch.zeros_like(global_state[name]) for name in param_names
        }
        
        # Weighted sum trên learnable params
        for result in client_results:
            client_id = result['client_id']
            num_samples = result['num_samples']
            weight = num_samples / total_samples
            
            client = self.clients[client_id]
            client_state = client.get_model_params()  # full state_dict
            
            for name in param_names:
                aggregated_learnable[name] += client_state[name] * weight
        
        # Tạo state_dict mới: giữ nguyên buffer cũ, chỉ thay learnable params
        new_state = copy.deepcopy(global_state)
        for name in param_names:
            new_state[name] = aggregated_learnable[name]
        
        return new_state
    
    def train_round_fedavg(
        self,
        num_epochs: int,
        learning_rate: float = 0.01,
        client_fraction: float = 1.0,
        verbose: int = 1
    ) -> Dict:
        """
        Thực hiện 1 round training với FedAvg
        """
        # Chọn subset clients (nếu client_fraction < 1.0)
        num_selected = max(1, int(len(self.clients) * client_fraction))
        selected_clients = np.random.choice(self.clients, num_selected, replace=False)
        
        # Distribute global model xuống clients
        self.distribute_model()
        
        # Train trên các clients được chọn
        client_results = []
        for idx, client in enumerate(selected_clients):
            if verbose:
                num_batches = len(client.train_loader)
                print(f"\n→ Training Client {client.client_id} ({idx+1}/{num_selected}) - {len(client.train_loader.dataset):,} samples, {num_batches} batches")
            result = client.train_fedavg(
                epochs=num_epochs,
                learning_rate=learning_rate,
                verbose=verbose
            )
            client_results.append(result)
            if verbose:
                print(f"  ✓ Client {client.client_id} completed - Avg Loss: {result['loss']:.4f}")
        
        # Aggregate models từ clients
        aggregated_params = self.aggregate_fedavg(client_results)
        self.set_global_params(aggregated_params)
        
        # Tính average loss
        avg_loss = np.mean([result['loss'] for result in client_results])
        
        return {
            'train_loss': avg_loss,
            'num_clients': len(selected_clients)
        }
    
    def train_round_fedprox(
        self,
        num_epochs: int,
        mu: float = 0.01,
        learning_rate: float = 0.01,
        client_fraction: float = 1.0,
        verbose: int = 0
    ) -> Dict:
        """
        Thực hiện 1 round training với FedProx
        """
        # Chọn subset clients
        num_selected = max(1, int(len(self.clients) * client_fraction))
        selected_clients = np.random.choice(self.clients, num_selected, replace=False)
        
        # Lưu global params để truyền cho clients (full state_dict)
        global_params = self.get_global_params()
        
        # Distribute global model xuống clients
        self.distribute_model()
        
        # Train trên các clients được chọn
        client_results = []
        for idx, client in enumerate(selected_clients):
            if verbose:
                num_batches = len(client.train_loader)
                print(f"\n→ Training Client {client.client_id} ({idx+1}/{num_selected}) - {len(client.train_loader.dataset):,} samples, {num_batches} batches")
            result = client.train_fedprox(
                epochs=num_epochs,
                global_params=global_params,
                mu=mu,
                learning_rate=learning_rate,
                verbose=verbose
            )
            client_results.append(result)
            if verbose:
                print(f"  ✓ Client {client.client_id} completed - Avg Loss: {result['loss']:.4f}")
        
        # Aggregate models từ clients (giống FedAvg, chỉ learnable params)
        aggregated_params = self.aggregate_fedavg(client_results)
        self.set_global_params(aggregated_params)
        
        # Tính average loss
        avg_loss = np.mean([result['loss'] for result in client_results])
        
        return {
            'train_loss': avg_loss,
            'num_clients': len(selected_clients)
        }
    
    def evaluate_global(self) -> Dict:
        """Đánh giá global model trên test set"""
        # Hỗ trợ cả test_loader (1 loader) và client_test_loaders (list loaders)
        test_loaders = []

        if self.client_test_loaders is not None:
            # Ưu tiên dùng client_test_loaders nếu có
            test_loaders = self.client_test_loaders
        elif self.test_loader is not None:
            # Fallback về test_loader đơn lẻ
            test_loaders = [self.test_loader]
        else:
            # Không có test loader nào
            return {'accuracy': 0.0, 'loss': 0.0}

        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for loader in test_loaders:
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    loss = criterion(output, target)

                    total_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += data.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def train_fedavg(
        self,
        num_rounds: int,
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        client_fraction: float = 1.0,
        eval_every: int = 1,
        verbose: int = 1
    ):
        """
        Train toàn bộ process với FedAvg
        """
        print(f"\n{'='*50}")
        print(f"Training with FedAvg")
        print(f"Rounds: {num_rounds}, Local epochs: {num_rounds}")
        print(f"Learning rate: {learning_rate}, Client fraction: {client_fraction}")
        print(f"{'='*50}\n")
        
        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"📍 ROUND {round_idx+1}/{num_rounds}")
                print(f"{'='*60}")
            
            # Train 1 round
            round_result = self.train_round_fedavg(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                client_fraction=client_fraction,
                verbose=verbose
            )
            
            if verbose:
                print(f"\n🔄 Aggregating models from {round_result['num_clients']} clients...")
            
            # Evaluate
            if (round_idx + 1) % eval_every == 0:
                if verbose:
                    print(f"📊 Evaluating global model on test set...")
                eval_result = self.evaluate_global()
                
                self.history['train_loss'].append(round_result['train_loss'])
                self.history['test_accuracy'].append(eval_result['accuracy'])
                self.history['test_loss'].append(eval_result['loss'])
                
                if verbose:
                    print(f"\n✅ Round {round_idx+1}/{num_rounds} Summary:")
                    print(f"   • Train Loss: {round_result['train_loss']:.4f}")
                    print(f"   • Test Accuracy: {eval_result['accuracy']*100:.2f}%")
                    print(f"   • Test Loss: {eval_result['loss']:.4f}")
        
        print(f"\nFedAvg Training Completed!")
        if self.history['test_accuracy']:
            print(f"Final Test Accuracy: {self.history['test_accuracy'][-1]*100:.2f}%")
        
        return self.history
    
    def train_fedprox(
        self,
        num_rounds: int,
        num_epochs: int = 5,
        mu: float = 0.01,
        learning_rate: float = 0.01,
        client_fraction: float = 1.0,
        eval_every: int = 1,
        verbose: int = 1
    ):
        """
        Train toàn bộ process với FedProx
        """
        print(f"\n{'='*50}")
        print(f"Training with FedProx")
        print(f"Rounds: {num_rounds}, Local epochs: {num_epochs}, Mu: {mu}")
        print(f"Learning rate: {learning_rate}, Client fraction: {client_fraction}")
        print(f"{'='*50}\n")
        
        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"📍 ROUND {round_idx+1}/{num_rounds}")
                print(f"{'='*60}")
            
            # Train 1 round
            round_result = self.train_round_fedprox(
                num_epochs=num_epochs,
                mu=mu,
                learning_rate=learning_rate,
                client_fraction=client_fraction,
                verbose=verbose
            )
            
            if verbose:
                print(f"\n🔄 Aggregating models from {round_result['num_clients']} clients...")
            
            # Evaluate
            if (round_idx + 1) % eval_every == 0:
                if verbose:
                    print(f"📊 Evaluating global model on test set...")
                eval_result = self.evaluate_global()
                
                self.history['train_loss'].append(round_result['train_loss'])
                self.history['test_accuracy'].append(eval_result['accuracy'])
                self.history['test_loss'].append(eval_result['loss'])
                
                if verbose:
                    print(f"\n✅ Round {round_idx+1}/{num_rounds} Summary:")
                    print(f"   • Train Loss: {round_result['train_loss']:.4f}")
                    print(f"   • Test Accuracy: {eval_result['accuracy']*100:.2f}%")
                    print(f"   • Test Loss: {eval_result['loss']:.4f}")
        
        print(f"\nFedProx Training Completed!")
        if self.history['test_accuracy']:
            print(f"Final Test Accuracy: {self.history['test_accuracy'][-1]*100:.2f}%")
        
        return self.history


def create_non_iid_split(
    dataset: Dataset, 
    num_clients: int, 
    num_classes: int = 10,
    alpha: float = 0.5
) -> List[Subset]:
    """
    Tạo Non-IID data split cho các clients sử dụng Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Số lượng clients
        num_classes: Số lượng classes
        alpha: Concentration parameter (nhỏ hơn = non-IID hơn)
    """
    # Lấy labels từ dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([label for _, label in dataset])
    
    # Tạo indices cho mỗi class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Phân chia sử dụng Dirichlet distribution
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        # Sample proportions từ Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Chia indices theo proportions
        indices = class_indices[class_idx]
        np.random.shuffle(indices)
        
        splits = np.cumsum(proportions * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, splits)
        
        for client_id, split in enumerate(split_indices):
            client_indices[client_id].extend(split.tolist())
    
    # Shuffle indices của mỗi client
    for indices in client_indices:
        np.random.shuffle(indices)
    
    # Tạo Subsets
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    return client_datasets


def print_data_distribution(client_datasets: List[Subset], num_classes: int = 10):
    """In ra phân phối data của mỗi client"""
    print("\nData Distribution across Clients:")
    print(f"{'Client':<10} {'Total':<10} {'Class Distribution'}")
    print("-" * 70)
    
    for client_id, dataset in enumerate(client_datasets):
        # Đếm số samples mỗi class
        if hasattr(dataset.dataset, 'targets'):
            labels = np.array(dataset.dataset.targets)[dataset.indices]
        elif hasattr(dataset.dataset, 'labels'):
            labels = np.array(dataset.dataset.labels)[dataset.indices]
        else:
            labels = np.array([dataset.dataset[i][1] for i in dataset.indices])
        
        class_counts = [np.sum(labels == i) for i in range(num_classes)]
        total = len(dataset)
        
        dist_str = ", ".join([f"{i}:{c}" for i, c in enumerate(class_counts) if c > 0])
        print(f"Client {client_id:<3} {total:<10} {dist_str}")
    
    print()
