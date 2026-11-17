import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datetime import datetime

from .data import training_loader
from .model import UNet

def sample_path_and_velocity(x0, x1, t):
    t = t[:, None, None, None]

    path = (1 - t)*x0 + t*x1
    velocity = x1 - x0

    return path, velocity


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    model = UNet().to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    def train_one_epoch(epoch_index, tb_writer):
        epoch_loss = 0.

        for i, data in enumerate(training_loader):
            x1, c = data
            c[torch.rand(c.shape, device=c.device) < 0.1] = 10

            batch = x1.shape[0]
            d1 = x1.shape[-2]
            d2 = x1.shape[-1]

            x0 = torch.normal(0, 1, size=(batch, 1, d1, d2))
            t = torch.rand(batch)

            x1 = x1.to(device)
            x0 = x0.to(device)
            c = c.to(device)
            t = t.to(device)

            path, target = sample_path_and_velocity(x0, x1, t)

            optimizer.zero_grad()
            outputs = model(x0, t, c)

            loss = loss_fn(outputs, target)
            loss.backward()

            optimizer.step()
            ema.update_parameters(model)

            epoch_loss += loss.item() * x1.shape[0]

        return epoch_loss / len(training_loader.dataset)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/digits_trainer_{timestamp}')

    epoch_number = 0
    EPOCHS = 2
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}:')

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        print(f'LOSS train {avg_loss}')
        writer.add_scalar('Loss/train', avg_loss, epoch_number + 1)
        writer.flush()

        epoch_number += 1

    os.makedirs(f'ckpts/model_{timestamp}', exist_ok=True)
    model_path = f'ckpts/model_{timestamp}/model_{timestamp}_{epoch_number}'
    torch.save({
        "model": model.state_dict(),
        "ema": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch_number,
    }, model_path)
