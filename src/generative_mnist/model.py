import torch

import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_down1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv_down2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_down3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up1 = nn.Conv2d(128, 32, 3, padding=1)
        self.conv_up2 = nn.Conv2d(64, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.time = nn.Linear(1, 64)
        self.embedding = nn.Embedding(11, 64)

    def forward(self, x, t, y):
        c = self.time(t[:, None]) + self.embedding(y)

        x = F.relu(self.conv_down1(x))
        r1 = x
        x = F.relu(self.conv_down2(self.pool(x)))
        r2 = x
        x = F.relu(self.conv_down3(self.pool(x)) + c[:, :, None, None])

        x = F.relu(self.conv_up1(torch.cat((self.upsample(x), r2), dim=1)))
        x = self.conv_up2(torch.cat((self.upsample(x), r1), dim=1))

        return x
