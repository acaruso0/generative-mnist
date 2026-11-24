import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datetime import datetime

from .data import training_loader
from .model import UNet
from .utils import sampler_from_dict


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    arch = sys.argv[1]
    assert(arch == '--diffusion' or arch == '--flow')
    if arch == '--diffusion':
        sampler_kwargs = {
            'architecture': 'diffusion',
            'schedule': {'start': 1e-4,
                         'end': 0.02,
                         'steps': 1000},
            'device': device,
        }
    elif arch == '--flow':
        sampler_kwargs = {
            'architecture': 'flow-matching',
            'device': device,
        }
    sampler = sampler_from_dict(**sampler_kwargs)

    model = UNet().to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    def train_one_epoch():
        epoch_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data
            labels[torch.rand(labels.shape, device=labels.device) < 0.1] = 10

            inputs, labels = inputs.to(device), labels.to(device)

            path, target, time = sampler.sample(inputs)

            optimizer.zero_grad()
            outputs = model(path, time, labels)

            loss = loss_fn(outputs, target)
            loss.backward()

            optimizer.step()
            ema.update_parameters(model)

            epoch_loss += loss.item() * outputs.shape[0]

        return epoch_loss / len(training_loader.dataset)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/digits_trainer_{timestamp}')

    epoch_number = 0
    EPOCHS = 25
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}:')

        model.train(True)
        avg_loss = train_one_epoch()

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
        "sampler_kwargs": sampler_kwargs,
    }, model_path)
