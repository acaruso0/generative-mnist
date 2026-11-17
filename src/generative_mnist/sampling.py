import torch


def sample_path_and_velocity(x0, x1, t):
    t = t[:, None, None, None]

    path = (1 - t)*x0 + t*x1
    velocity = x1 - x0

    return path, velocity


@torch.inference_mode()
def euler_method_with_guidance(model, x0, c, w=1.0, steps=100):
    x = x0.clone()
    null_labels = 10 * torch.ones_like(c).to(x.device)

    for k in range(steps):
        t = (torch.ones(x.shape[0]) * k / steps).to(x.device)
        v_c = model(x, t, c)
        v_u = model(x, t, null_labels)
        x += (1 / steps) * (v_u + w * (v_c - v_u))

    return x
