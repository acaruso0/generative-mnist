import matplotlib.pyplot as plt
import numpy as np

from .sampling import DiffusionSampler, FlowSampler


def sampler_from_dict(**kwargs):
    if kwargs['architecture'] == 'diffusion':
        sampler = DiffusionSampler(kwargs['schedule'])
    elif kwargs['architecture'] == 'flow-matching':
        sampler = FlowSampler()
    return sampler

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # Unnormalize.
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
