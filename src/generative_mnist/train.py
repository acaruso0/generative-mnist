import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datetime import datetime

from .data import training_loader
from .model import CNN, UNet
from .utils import sampler_from_dict


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    mode = sys.argv[1][2:]
    assert(mode == 'diffusion' or mode == 'flow')
    if mode == 'diffusion':
        sampler_kwargs = {
            'architecture': mode,
            'schedule': {'start': 1e-4,
                         'end': 0.02,
                         'steps': 1000},
            'device': device,
        }
    elif mode == 'flow':
        sampler_kwargs = {
            'architecture': mode,
            'device': device,
        }
    sampler = sampler_from_dict(**sampler_kwargs)

    assert(sys.argv[2] == "--classifier" and len(sys.argv) > 3)
    classifier_path = sys.argv[3]

    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()

    model = UNet().to(device)

    loss_fn = torch.nn.MSELoss()
    vloss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    def train_one_epoch():
        avg_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels[torch.rand(labels.shape) < 0.1] = 10

            path, target, time = sampler.sample(inputs)

            optimizer.zero_grad()
            outputs = model(path, time, labels)

            loss = loss_fn(outputs, target)
            loss.backward()

            optimizer.step()
            ema.update_parameters(model)

            avg_loss += loss.item() * outputs.shape[0]

        return avg_loss / len(training_loader.dataset)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{mode}/{timestamp}')

    os.makedirs(f'ckpts/{mode}', exist_ok=True)

    epoch_number = 0
    EPOCHS = 50
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}:')

        model.train(True)
        avg_loss = train_one_epoch()

        avg_vloss = 0.0
        model.eval()

        with torch.inference_mode():
            for n in range(20):
                vlabels = torch.randint(0, 10, size=(50,)).to(device)
                vinputs = torch.normal(0, 1, size=(vlabels.shape[0], 1, 28, 28)).to(device)
                if mode == 'diffusion':
                    voutputs = sampler.predict(model, vinputs, vlabels, w=5.0)
                if mode == 'flow':
                    voutputs = sampler.predict(model, vinputs, vlabels, w=5.0, steps=250)
                voutputs = classifier(voutputs)
                vloss = vloss_fn(voutputs, vlabels)
                avg_vloss += vloss.item() * voutputs.shape[0]
            avg_vloss /= 1000
            
            # Just for funzies
            vlabels_pred = voutputs.argmax(dim=1)
            print(f'Conditioning labels: {vlabels[:20].tolist()}')
            print(f'Classifier predicts: {vlabels_pred[:20].tolist()}')

        print(f'LOSS train {avg_loss:.4f} valid {avg_vloss:.4f}')
        writer.add_scalar('Loss/train', avg_loss, epoch_number + 1)
        writer.add_scalar('Loss/valid', avg_vloss, epoch_number + 1)
        writer.flush()

        epoch_number += 1

    model_path = f'ckpts/{mode}/{timestamp}_{epoch_number}'
    torch.save({
        "model": model.state_dict(),
        "ema": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch_number,
        "sampler_kwargs": sampler_kwargs,
    }, model_path)
