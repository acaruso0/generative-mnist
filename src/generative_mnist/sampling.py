import torch


class Sampler:
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class FlowSampler(Sampler):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def sample(self, x1):
        t = torch.rand(x1.shape[0]).to(self.device)[:, None, None, None]
        x0 = torch.randn_like(x1)

        path = (1 - t)*x0 + t*x1
        velocity = x1 - x0

        return path, velocity, t.squeeze()

    @torch.inference_mode()
    def predict(self, model, x0, c, w=1.0, steps=100):
        # Euler method with classifier-free guidance.
        x = x0.clone()
        c_b = torch.cat([c, torch.full_like(c, 10)])

        for k in range(steps):
            x_b = torch.cat([x, x])
            t = (torch.ones(x_b.shape[0]) * k / steps).to(self.device)

            v_c, v_u = model(x_b, t, c_b).chunk(2)
            x += (1 / steps) * (v_u + w * (v_c - v_u))

        return x


class DiffusionSampler(Sampler):
    def __init__(self, schedule, device):
        super().__init__()
        self.T = schedule['steps']
        self.betas = torch.linspace(**schedule, device=device)
        self._alphas = torch.cumprod(1 - self.betas, dim=0)
        self.device = device

    def sample(self, x0):
        t = torch.randint(0, self.T, (x0.shape[0],)).to(self.device)
        _at = self._alphas[t][:, None, None, None]
        noise = torch.randn_like(x0)

        path = torch.sqrt(_at) * x0 + torch.sqrt(1 - _at) * noise

        return path, noise, t

    @torch.inference_mode()
    def predict(self, model, x1, c, w=1.0):
        # DDPM with classifier-free guidance.
        x = x1.clone()
        c_b = torch.cat([c, torch.full_like(c, 10)])

        for t in range(self.T - 1, 0, -1): # Because it's trained from 0 to T. Extra special 0-step.
            bt = self.betas[t]
            _at = self._alphas[t]
            z = torch.randn_like(x)

            x_b = torch.cat([x, x])
            t = torch.ones(x_b.shape[0], device=self.device) * t
            e_c, e_u = model(x_b, t, c_b).chunk(2)
            e = e_u + w * (e_c - e_u)
            x = torch.pow(1 - bt, -0.5) * (x - bt * torch.pow(1 - _at, -0.5) * e) + torch.pow(bt, 0.5) * z
        b0 = self.betas[0]
        _a0 = self._alphas[0]

        x_b = torch.cat([x, x])
        t = torch.zeros(x_b.shape[0]).to(self.device)
        e_c, e_u = model(x_b, t, c_b).chunk(2)
        e = e_u + w * (e_c - e_u)
        x = torch.pow(1 - b0, -0.5) * (x - b0 * torch.pow(1 - _a0, -0.5) * e)

        return x
