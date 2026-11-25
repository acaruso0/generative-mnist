# Adapted from: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

import os
import torch
import torchvision
from torchvision.transforms import v2

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F

from .data import training_loader, validation_loader
from .model import CNN


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    model = CNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train_one_epoch(epoch_index, tb_writer):
        avg_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            avg_loss += loss.item() * outputs.shape[0]

        return avg_loss / len(training_loader.dataset)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/classifier/{timestamp}')

    os.makedirs('ckpts/classifier', exist_ok=True)

    epoch_number = 0
    EPOCHS = 15
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}:')

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        avg_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                avg_vloss += vloss.item() * voutputs.shape[0]
            avg_vloss /= len(validation_loader.dataset)

        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training' : avg_loss,
                            'Validation' : avg_vloss},
                        epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'ckpts/classifier/{timestamp}_{epoch_number}'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
