import torch
import torchvision
from torchvision.transforms import v2

import matplotlib.pyplot as plt

from .utils import matplotlib_imshow


transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

training_set = torchvision.datasets.MNIST('/home/betelgeuse/codes/generative-mnist/data', train=True, transform=transform, download=True)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True)


if __name__ == "__main__":
    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    print(f'Training set has {len(training_set)} instances')

    print(f'Shape of batched digit images: {images.shape}')
    print(f'Shape of batched labels: {labels.shape}')

    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)

    print(labels.reshape(-1, 8))  # Reshaped to match the grid shape of imshow.
    plt.show()
