import torch
import torch.nn as nn
from model_v2 import ResidualBlock

class ValueNetwork(nn.Module):
    """
    Value Network for Great Kingdom AI.
    Reuses the PolicyNetworkV2 CNN trunk: 1 Conv Layer -> 4 Residual Blocks.
    Replaces the policy head with a value head outputting a scalar in range [-1, 1] using Tanh.
    """
    def __init__(self):
        super(ValueNetwork, self).__init__()
        
        # Trunk: Initial Conv Block
        self.conv = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Trunk: 4 Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(4)]
        )
        
        # Value Head: Conv1x1 reduces channels to 1 (or 2), followed by linear layers and Tanh activation
        self.conv_value = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, bias=False)
        self.bn_value = nn.BatchNorm2d(2)
        self.relu_value = nn.ReLU()
        self.fc_value1 = nn.Linear(2 * 9 * 9, 64)
        self.relu_fc = nn.ReLU()
        self.fc_value2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Input shape: (Batch_size, 4, 9, 9)
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # Value Head
        x_val = self.relu_value(self.bn_value(self.conv_value(x)))
        x_val = torch.flatten(x_val, start_dim=1)
        x_val = self.relu_fc(self.fc_value1(x_val))
        value = self.tanh(self.fc_value2(x_val)) # Output range: [-1, 1]
        return value.squeeze(-1) # Output shape: (Batch_size,)
