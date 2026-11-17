import sys
import torch
import torchvision
import matplotlib.pyplot as plt

from .model import UNet
from .utils import matplotlib_imshow


@torch.inference_mode()
def euler_method(x0, c, model, steps=100):
    x = x0.clone()

    for k in range(steps):
        t = (torch.ones(x.shape[0]) * k / steps).to(x.device)
        x += (1 / steps) * model(x, t, c)

    return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Inference on {device}')

    path = sys.argv[1]
    print(path)
    ckpt = torch.load(path, map_location=device)

    model = UNet().to(device)
    model.load_state_dict(ckpt["ema"])

    labels = torch.arange(10).repeat_interleave(5).to(device)
    inputs = torch.normal(0, 1, size=(labels.shape[0], 1, 28, 28)).to(device)

    with torch.inference_mode():
        x = euler_method(inputs, labels, model, steps=250)

    x = x.cpu()
    img_grid = torchvision.utils.make_grid(x, nrow=5)
    matplotlib_imshow(img_grid, one_channel=True)

    plt.show()
