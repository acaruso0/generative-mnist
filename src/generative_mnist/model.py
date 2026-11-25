import torch

import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(x, dim=128, scale=1.0):
    freqs = scale / torch.pow(10000, 2 * torch.arange(dim // 2).to(x.device) / dim)
    args = x[:, None] * freqs[None, :]
    
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, activation=nn.ReLU, last_bias=True):
        super().__init__()

        assert len(hidden_channels) > 0

        layers = []

        input_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, hidden_channels[-1], bias=last_bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_down1 = nn.Conv2d(1, 32, 3, padding=1)
        self.proj_down1 = nn.Linear(128, 32)
        self.conv_down2 = nn.Conv2d(32, 64, 3, padding=1)
        self.proj_down2 = nn.Linear(128, 64)
        self.conv_down3 = nn.Conv2d(64, 64, 3, padding=1)
        self.proj_down3 = nn.Linear(128, 64)
        self.conv_up1 = nn.Conv2d(128, 32, 3, padding=1)
        self.proj_up1 = nn.Linear(128, 32)
        self.conv_up2 = nn.Conv2d(64, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.mlp = MLP(128, [128, 128])
        self.conditional_embedding = nn.Embedding(11, 128)

    def forward(self, x, t, c):
        z = self.mlp(sinusoidal_embedding(t, dim=128, scale=1000.0)) + self.conditional_embedding(c)

        h = F.silu(self.conv_down1(x) + self.proj_down1(z)[:, :, None, None])
        r1 = h
        h = F.silu(self.conv_down2(self.pool(h)) + self.proj_down2(z)[:, :, None, None])
        r2 = h
        h = F.silu(self.conv_down3(self.pool(h)) + self.proj_down3(z)[:, :, None, None])

        h = F.silu(self.conv_up1(torch.cat((self.upsample(h), r2), dim=1)) + self.proj_up1(z)[:, :, None, None])
        x = self.conv_up2(torch.cat((self.upsample(h), r1), dim=1))

        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
