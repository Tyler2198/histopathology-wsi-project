import torch
import torch.nn as nn
from torchvision import models


class ResNetBinary(nn.Module):
    def __init__(self, pretrained=True, freeze_base=True):
        super(ResNetBinary, self).__init__()

        # Load ResNet18 base
        self.base_model = models.resnet18(pretrained=pretrained)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Replace final layer for binary classification
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Use sigmoid for binary classification
        )

    def forward(self, x):
        return self.base_model(x).squeeze()
