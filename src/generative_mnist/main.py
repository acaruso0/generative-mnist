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


loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

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
        outputs = model(x1, t, labels)

        loss = loss_fn(outputs, target)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/digits_trainer_{timestamp}')
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number + 1}:')

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    '''
    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    '''
    # print(f'LOSS train {avg_loss} valid {avg_vloss}')
    print(f'LOSS train {avg_loss}')

    '''
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'model_{timestamp}_{epoch_number}'
        torch.save(model.state_dict(), model_path)
    '''

    epoch_number += 1


# saved_model = UNet()
# saved_model.load_state_dict(torch.load(PATH))
