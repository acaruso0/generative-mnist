import torch


class Sampler:
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class FlowSampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, x1):
        t = torch.rand(x1.shape[0]).to(x1.device)[:, None, None, None]
        x0 = torch.randn_like(x1)

        path = (1 - t)*x0 + t*x1
        velocity = x1 - x0

        return path, velocity, t.squeeze()

    @torch.inference_mode()
    def predict(self, model, x0, c, w=1.0, steps=100):
        # Euler method with guidance
        x = x0.clone()
        null_labels = 10 * torch.ones_like(c).to(x.device)

        for k in range(steps):
            t = (torch.ones(x.shape[0]) * k / steps).to(x.device)
            v_c = model(x, t, c)
            v_u = model(x, t, null_labels)
            x += (1 / steps) * (v_u + w * (v_c - v_u))

        return x


class DiffusionSampler(Sampler):
    def __init__(self, schedule):
        super().__init__()
        self.T = schedule['steps']
        self.betas = torch.linspace(**schedule)
        self._alphas = torch.cumprod(1 - self.betas, dim=0)

    def sample(self, x0):
        t = torch.randint(0, self.T, (x0.shape[0],)).to(x0.device)
        _at = self._alphas.to(t.device)[t][:, None, None, None]
        noise = torch.randn_like(x0)

        path = torch.sqrt(_at) * x0 + torch.sqrt(1 - _at) * noise

        return path, noise, t

    @torch.inference_mode()
    def predict(self, model, x0, c, w=1.0, steps=100):
        raise NotImplementedError
