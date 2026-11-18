"""Example of a self-contained updater. It shows the interface only; it is not a recommendation,
and its settings are not the starting optimizer's.

It runs AdamW with parameters grouped by role (classifier head, weights, no-decay tensors), owns its
own schedule written as a function of the absolute update index, carries Adam moments over from the
previous updater when that one exposes them under the same names, and exposes param_groups.
"""
import math

import torch

HEAD_PREFIXES = ("classifier.", "head.", "fc.", "heads.")
NO_DECAY_NAMES = ("pos_embed", "cls_token", "reg_token", "pos_embedding", "class_token")


def role(name, p):
    if name.startswith(HEAD_PREFIXES):
        return "head"
    if p.ndim <= 1 or name.endswith(NO_DECAY_NAMES):
        return "no_decay"        # norm gains, biases, LayerScale, position embedding, class token
    return "weight"


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.warmup = 5 * spec["updates_per_epoch"]
        self.params = dict(named_params)
        self.roles = {n: role(n, p) for n, p in named_params}
        self.lr = {"weight": 1e-4, "no_decay": 1e-4, "head": 1e-3}
        self.wd = {"weight": 0.05, "no_decay": 0.0, "head": 0.05}
        self.betas, self.eps = (0.9, 0.999), 1e-8
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
        self.param_groups = [
            {"name": r, "params": [p for n, p in named_params if self.roles[n] == r], "lr": self.lr[r]}
            for r in self.lr
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
            group["lr"] = self.lr[group["name"]] * f
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for n, p in self.params.items():
            g = grads[n]
            if g is None:
                continue
            r = self.roles[n]
            step_lr = self.lr[r] * f
            if self.wd[r]:
                p.mul_(1 - step_lr * self.wd[r])
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
