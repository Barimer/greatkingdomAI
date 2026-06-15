import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    Policy Network v1 for Great Kingdom AI.
    Replicates Depth-2 Minimax player behavior using a 3-layer CNN followed by dense layers.
    """
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        # Conv Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        # Flatten and Dense Layers
        # 64 channels * 9 * 9 = 5184
        self.fc1 = nn.Linear(64 * 9 * 9, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu_fc1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 82)
        
    def forward(self, x):
        # Input shape: (Batch_size, 4, 9, 9)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully Connected
        x = self.relu_fc1(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x) # Output logits of shape (Batch_size, 82)
        return x
