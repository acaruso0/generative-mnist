import torch

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .data import training_loader
from .model import UNet

model = UNet()


def sample_path_and_velocity(x0, x1, t):
    t = t[:, None, None, None]

    path = (1 - t)*x0 + t*x1
    velocity = x1 - x0

    return path, velocity


@torch.inference_mode()
def euler_method(x0, c, model, steps=100):
    x = x0.clone()

    for k in range(steps):
        t = torch.ones(x.shape[0]) * k / steps
        x += (1 / steps) * model(x, t, c)

    return x


loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

def train_one_epoch(epoch_index, tb_writer):
    epoch_loss = 0.

    for i, data in enumerate(training_loader):
        x1, labels = data
        labels[torch.rand(labels.shape, device=labels.device) < 0.1] = 10

        batch = x1.shape[0]
        d1 = x1.shape[-2]
        d2 = x1.shape[-1]

        x0 = torch.normal(0, 1, size=(batch, 1, d1, d2))
        t = torch.rand(batch)

        path, target = sample_path_and_velocity(x0, x1, t)

        optimizer.zero_grad()
        outputs = model(x0, t, labels)

        loss = loss_fn(outputs, target)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() * x1.shape[0]

    return epoch_loss / len(training_loader.dataset)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/digits_trainer_{timestamp}')

epoch_number = 0
EPOCHS = 20
for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number + 1}:')

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    print(f'LOSS train {avg_loss}')
    writer.add_scalar('Loss/train', avg_loss, epoch_number + 1)
    writer.flush()

    epoch_number += 1

labels = torch.arange(10).repeat_interleave(5)
inputs = torch.normal(0, 1, size=(labels.shape[0], 1, 28, 28))

with torch.inference_mode():
    x = euler_method(inputs, labels, model)

print(x)
print(x.shape)

import torchvision
import matplotlib.pyplot as plt

from .utils import matplotlib_imshow

img_grid = torchvision.utils.make_grid(x, nrow=5)
matplotlib_imshow(img_grid, one_channel=True)

plt.show()

# saved_model = UNet()
# saved_model.load_state_dict(torch.load(PATH))
