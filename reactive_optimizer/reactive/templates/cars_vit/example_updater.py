"""Example of a self-contained updater for this ViT-B/16 run. It shows the interface only; it is not a
recommendation, and its settings are not the starting optimizer's.

It runs AdamW with parameters grouped by role and by depth (a per-layer learning-rate multiplier),
owns its own schedule written as a function of the absolute update index, carries Adam moments over
from the previous updater when that one exposes them under the same names, and exposes param_groups.
"""
import math
import re

import torch


def role(name, p):
    if name.startswith("heads."):
        return "head"
    if p.ndim <= 1 or name.endswith(("pos_embedding", "class_token")):
        return "no_decay"        # LayerNorm gains, biases, position embedding, class token
    return "weight"


def depth(name):
    """0 for the patch embedding, 1-12 for encoder layers, 13 for the final norm and head."""
    m = re.search(r"encoder_layer_(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if name.startswith(("conv_proj", "class_token", "encoder.pos_embedding")):
        return 0
    return 13


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.warmup = 3 * spec["updates_per_epoch"]
        self.params = dict(named_params)
        self.peak_lr = 1e-4
        self.wd = {"weight": 0.05, "no_decay": 0.0, "head": 0.05}
        self.betas, self.eps = (0.9, 0.999), 1e-8
        self.mult = {n: 0.75 ** (13 - depth(n)) for n in self.params}   # deeper layers learn faster
        self.roles = {n: role(n, p) for n, p in named_params}
        self.m = {n: torch.zeros_like(p) for n, p in named_params}
        self.v = {n: torch.zeros_like(p) for n, p in named_params}
        self.t = 0
        old_m, old_v = getattr(previous, "m", None), getattr(previous, "v", None)
        if isinstance(old_m, dict) and isinstance(old_v, dict):
            for n in self.m:
                if n in old_m and old_m[n].shape == self.m[n].shape:
                    self.m[n].copy_(old_m[n])
                    self.v[n].copy_(old_v[n])
            self.t = int(getattr(previous, "t", 0))
        self.param_groups = [{"name": "all", "params": list(self.params.values()), "lr": self.peak_lr}]

    def factor(self, update):
        if update <= self.warmup:
            return update / self.warmup
        t = (update - self.warmup) / max(1, self.total - self.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        lr = self.peak_lr * self.factor(update)
        self.param_groups[0]["lr"] = lr
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for n, p in self.params.items():
            g = grads[n]
            if g is None:
                continue
            step_lr = lr * self.mult[n]
            wd = self.wd[self.roles[n]]
            if wd:
                p.mul_(1 - step_lr * wd)
            self.m[n].mul_(b1).add_(g, alpha=1 - b1)
            self.v[n].mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (self.v[n] / c2).sqrt_().add_(self.eps)
            p.addcdiv_(self.m[n], denom, value=-step_lr / c1)

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load_state_dict(self, state):
        for n in self.m:
            self.m[n].copy_(state["m"][n])
            self.v[n].copy_(state["v"][n])
        self.t = int(state["t"])


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
