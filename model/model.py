import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(x + out)

class ValueNet(nn.Module):
    def __init__(self, board_size: int, in_ch: int = 2, num_blocks: int = 4,
                 hidden: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(num_blocks)]
        )
        # global avg pool -> 1 scalar
        self.fc = nn.Linear(hidden * board_size * board_size, 1)

    def forward(self, x):
        # x: [B, in_ch, H, W]
        h = F.relu(self.conv_in(x))
        h = self.res_blocks(h)
        h = h.view(h.size(0), -1)
        return torch.tanh(self.fc(h))  # in [-1,1]
