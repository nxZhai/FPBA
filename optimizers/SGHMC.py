import math

import torch
from torch.optim import Optimizer


class SGHMC(Optimizer):
    def __init__(self, params, config=None):
        config = dict(config or {})
        defaults = {"lr": 0.5, "alpha": 0, "gamma": 0.01, "L": 15, "T": 1e-5}
        config = {key: config.get(key, value) for key, value in defaults.items()}
        super().__init__(params, config)
        self.config = config

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        steps = int(self.config["L"])
        step_size = float(self.config["lr"]) / math.sqrt(steps + 1)
        gamma = float(self.config["gamma"])
        temperature = float(self.config["T"])

        for _ in range(steps):
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(parameter)
                    momentum = state["momentum"]
                    noise = torch.randn_like(parameter)
                    momentum.mul_(1 - gamma).add_(parameter.grad, alpha=-step_size)
                    momentum.add_(
                        noise, alpha=math.sqrt(2 * step_size * gamma * temperature)
                    )
                    parameter.add_(momentum)
        return loss


class SGAdaHMC(Optimizer):
    def __init__(self, params, config=None):
        config = dict(config or {})
        defaults = {
            "lr": 0.001,
            "alpha": 0,
            "gamma": 0.01,
            "L": 30,
            "T": 1e-5,
            "tao": 2,
            "C": 1,
        }
        config = {key: config.get(key, value) for key, value in defaults.items()}
        super().__init__(params, config)
        self.config = config

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        steps = int(self.config["L"])
        epsilon = float(self.config["lr"]) / math.sqrt(steps + 1)
        temperature = float(self.config["T"])
        tao = float(self.config["tao"])
        c = float(self.config["C"])

        for _ in range(steps):
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(parameter)
                        state["v"] = torch.full_like(parameter, 1e-5)
                    momentum = state["momentum"]
                    v = state["v"]
                    grad = parameter.grad
                    v.mul_(1 - 1 / tao).addcmul_(grad, grad, value=1 / tao)
                    v.clamp_min_(1e-5)
                    inv_sqrt_v = v.rsqrt()
                    noise = torch.randn_like(parameter)
                    momentum.add_(
                        -(epsilon**2) * inv_sqrt_v * grad
                        - epsilon * inv_sqrt_v * c * momentum
                        + temperature
                        * (2 * epsilon**3 * inv_sqrt_v * c * inv_sqrt_v - epsilon**4)
                        * noise
                    )
                    parameter.add_(momentum)
        return loss
