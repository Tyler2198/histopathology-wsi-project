import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, input_size=96):
        super(BaselineCNN, self).__init__()

        # Input: (3, 96, 96)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, 48, 48)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output after pool: (64, 24, 24)

        # Flattening
        flattened_size = 64 * (input_size // 4) * (input_size // 4)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Conv + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))   # Conv + ReLU + Pool
        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))         # Binary classification
        return x.squeeze()
