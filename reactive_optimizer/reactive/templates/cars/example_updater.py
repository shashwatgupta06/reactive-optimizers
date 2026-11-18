"""Example of a self-contained updater. It shows the interface only; it is not a recommendation.

It re-implements SGD with Nesterov momentum, with parameters grouped by role, its own schedule
written as a function of the absolute update index, and momentum carried over from the
previous updater when that one exposes buffers under the same names.
"""
import math

import torch


def role(name, p):
    if p.ndim <= 1:
        return "norm_bias"      # BatchNorm gains and biases, the classifier bias
    if name.startswith("fc."):
        return "head"
    if name.startswith("conv1."):
        return "stem"
    return "conv"


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.warmup = 570
        self.params = dict(named_params)
        self.settings = {  # per role: peak rate, weight decay
            "conv": dict(lr=0.05, wd=5e-4),
            "stem": dict(lr=0.05, wd=5e-4),
            "head": dict(lr=0.05, wd=5e-4),
            "norm_bias": dict(lr=0.05, wd=0.0),
        }
        self.momentum = 0.9
        self.roles = {n: role(n, p) for n, p in named_params}
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}
        old = getattr(previous, "buf", None)  # carry momentum over when the previous updater has it
        if isinstance(old, dict):
            for n in self.buf:
                if n in old and old[n].shape == self.buf[n].shape:
                    self.buf[n].copy_(old[n])
        # torch.optim-style groups, so the training script can log learning rates
        self.param_groups = [
            {"name": r, "params": [p for n, p in named_params if self.roles[n] == r], "lr": s["lr"]}
            for r, s in self.settings.items()
        ]

    def factor(self, update):
        if update <= self.warmup:
            return update / self.warmup
        t = (update - self.warmup) / max(1, self.total - self.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        f = self.factor(update)
        for group in self.param_groups:
            group["lr"] = self.settings[group["name"]]["lr"] * f
        for n, p in self.params.items():
            g = grads[n]
            if g is None:
                continue
            s = self.settings[self.roles[n]]
            d = g.add(p, alpha=s["wd"]) if s["wd"] else g
            buf = self.buf[n].mul_(self.momentum).add_(d)
            p.add_(d.add(buf, alpha=self.momentum), alpha=-s["lr"] * f)  # Nesterov

    def state_dict(self):
        return {"buf": self.buf}

    def load_state_dict(self, state):
        for n, b in state["buf"].items():
            self.buf[n].copy_(b)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
