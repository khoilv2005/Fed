import torch
import torch.nn as nn


class CNN_GRU_Model(nn.Module):
    """
    Mô hình CNN-GRU (CNN + GRU + MLP + Softmax) bằng PyTorch
    Tương đương với model trong TensorFlow/Keras
    """
    def __init__(self, input_shape, num_classes=2):
        """
        Args:
            input_shape: tuple (sequence_length,) - độ dài chuỗi input
            num_classes: số lượng classes để phân loại
        """
        super(CNN_GRU_Model, self).__init__()

        # Lấy sequence length từ input_shape
        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
        else:
            seq_length = input_shape

        self.input_shape = input_shape
        self.num_classes = num_classes

        # ===== CNN MODULE =====
        print("\n→ Building CNN Module...")

        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Conv Block 3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Tính output size sau các pooling layers
        cnn_output_length = seq_length // (2 * 2 * 2)  # 3 pooling layers với size=2
        self.cnn_output_size = 256 * cnn_output_length

        # ===== GRU MODULE =====
        print("→ Building GRU Module...")

        # GRU Layer 1 (return sequences)
        self.gru1 = nn.GRU(input_size=1, hidden_size=128, batch_first=True)

        # GRU Layer 2 (no return sequences)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, batch_first=True)

        self.gru_output_size = 64

        # ===== MLP MODULE =====
        print("→ Building MLP Module...")

        # Concatenated size
        concat_size = self.cnn_output_size + self.gru_output_size

        # Dense Layer 1
        self.dense1 = nn.Linear(concat_size, 256)
        self.bn_mlp1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        # Dense Layer 2
        self.dense2 = nn.Linear(256, 128)
        self.bn_mlp2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        # Output layer
        self.output = nn.Linear(128, num_classes)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        batch_size = x.size(0)

        # ===== CNN MODULE =====
        # Reshape cho CNN: (batch, seq_length, 1) -> (batch, 1, seq_length)
        x_cnn = x.permute(0, 2, 1)

        # Conv Block 1
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.bn1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.pool1(x_cnn)

        # Conv Block 2
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.bn2(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.pool2(x_cnn)

        # Conv Block 3
        x_cnn = self.conv3(x_cnn)
        x_cnn = self.bn3(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.pool3(x_cnn)

        # Flatten
        cnn_output = x_cnn.view(batch_size, -1)

        # ===== GRU MODULE =====
        x_gru = x  # (batch, seq_length, 1)

        # GRU Layer 1
        x_gru, _ = self.gru1(x_gru)

        # GRU Layer 2 (lấy output cuối cùng)
        x_gru, _ = self.gru2(x_gru)
        gru_output = x_gru[:, -1, :]  # Lấy timestep cuối cùng

        # ===== CONCATENATE =====
        concatenated = torch.cat([cnn_output, gru_output], dim=1)

        # ===== MLP MODULE =====
        # Dense Layer 1
        x = self.dense1(concatenated)
        x = self.bn_mlp1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Dense Layer 2
        x = self.dense2(x)
        x = self.bn_mlp2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Output
        output = self.output(x)
        # Note: Không cần softmax ở đây nếu dùng CrossEntropyLoss
        # CrossEntropyLoss đã bao gồm softmax

        return output


def build_cnn_gru_model(input_shape, num_classes=2):
    """
    Hàm tiện ích để xây dựng và khởi tạo model CNN-GRU

    Args:
        input_shape: tuple (sequence_length,) - độ dài chuỗi input
        num_classes: số lượng classes để phân loại

    Returns:
        model: CNN_GRU_Model instance
    """
    model = CNN_GRU_Model(input_shape, num_classes)
    print(f"\n✓ Model created successfully!")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    return model


if __name__ == "__main__":
    # Test model
    print("="*60)
    print("Testing CNN-GRU Model")
    print("="*60)

    # Tạo model
    input_shape = (100,)  # sequence length = 100
    num_classes = 2
    model = build_cnn_gru_model(input_shape, num_classes)

    # In model summary
    print("\nModel Architecture:")
    print(model)

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_shape[0])

    print(f"\nTesting forward pass...")
    print(f"Input shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"✓ Forward pass successful!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
