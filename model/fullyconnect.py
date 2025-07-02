import torch.nn as nn


class FullyConnect(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnect, self).__init__()
        hidden_dim = input_dim*4

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.block(x)

        return x
