import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard Residual Block for PolicyNetworkV2.
    Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> Add Skip -> ReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu2(out)
        return out

class PolicyNetworkV2(nn.Module):
    """
    Policy Network v2 for Great Kingdom AI.
    AlphaZero-style architecture: 1 Conv Layer -> 4 Residual Blocks -> 1x1 Conv Policy Head -> Dense Layer.
    Total parameters are significantly reduced (~0.5M) compared to V1, reducing overfitting risk.
    """
    def __init__(self):
        super(PolicyNetworkV2, self).__init__()
        
        # Initial Conv Block
        self.conv = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # 4 Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(4)]
        )
        
        # Policy Head: Conv1x1 reduces channels to 32, followed by flatten and Linear to 82 classes
        self.conv_policy = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(32)
        self.relu_policy = nn.ReLU()
        self.fc_policy = nn.Linear(32 * 9 * 9, 82)
        
    def forward(self, x):
        # Input shape: (Batch_size, 4, 9, 9)
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # Policy Head
        x_pol = self.relu_policy(self.bn_policy(self.conv_policy(x)))
        x_pol = torch.flatten(x_pol, start_dim=1)
        logits = self.fc_policy(x_pol) # Output shape: (Batch_size, 82)
        return logits
