import torch
import torch.nn as nn


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*7*7, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.network(x)


class ConvModel_1(nn.Module):
    def __init__(self):
        super(ConvModel_1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 1, 96),
            nn.ReLU(),
            nn.Linear(96, 10),
        )

    def forward(self, x):
        return self.network(x)
