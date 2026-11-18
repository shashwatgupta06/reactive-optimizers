"""Own SGD-Nesterov (foreach) with per-role groups. Version 1 reproduces the starting recipe exactly,
but owns its state so that later decisions can change per-role rates / decay without surprises.
"""
import math

import torch

WARMUP = 570

# per role: peak lr, weight decay
SETTINGS = {
    "conv": dict(lr=0.05, wd=1e-3),   # v2: 5e-4 -> 1e-3 (val stalled while train loss fell)
    "stem": dict(lr=0.05, wd=1e-3),
    "head": dict(lr=0.05, wd=1e-3),
    "bn": dict(lr=0.05, wd=0.0),
}
MOMENTUM = 0.9


def role(name, p):
    if p.ndim <= 1:
        return "bn"
    if name.startswith("fc."):
        return "head"
    if name.startswith("conv1."):
        return "stem"
    return "conv"


def lr_factor(update, total):
    if update <= WARMUP:
        return update / WARMUP
    t = (update - WARMUP) / max(1, total - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * t))


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.names = [n for n, _ in named_params]
        self.params = dict(named_params)
        self.roles = {n: role(n, p) for n, p in named_params}
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}
        self._carry(previous, spec)
        self.groups = {r: [n for n in self.names if self.roles[n] == r] for r in SETTINGS}
        self.param_groups = [
            {"name": r, "params": [self.params[n] for n in ns], "lr": SETTINGS[r]["lr"]}
            for r, ns in self.groups.items()
        ]

    @torch.no_grad()
    def _carry(self, previous, spec):
        old = getattr(previous, "buf", None)
        if isinstance(old, dict):
            for n in self.buf:
                if n in old and old[n].shape == self.buf[n].shape:
                    self.buf[n].copy_(old[n])
            return
        opt = spec.get("user_optimizer") if spec else None
        if opt is not None:
            for n, p in self.params.items():
                st = opt.state.get(p, {})
                mb = st.get("momentum_buffer")
                if mb is not None:
                    self.buf[n].copy_(mb)

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        f = lr_factor(update, self.total)
        for g in self.param_groups:
            r = g["name"]
            lr = SETTINGS[r]["lr"] * f
            g["lr"] = lr
            ns = [n for n in self.groups[r] if grads.get(n) is not None]
            if not ns:
                continue
            ps = [self.params[n] for n in ns]
            gs = [grads[n] for n in ns]
            bs = [self.buf[n] for n in ns]
            wd = SETTINGS[r]["wd"]
            if wd:
                d = torch._foreach_add(gs, ps, alpha=wd)
            else:
                d = gs
            torch._foreach_mul_(bs, MOMENTUM)
            torch._foreach_add_(bs, d)
            upd = torch._foreach_add(d, bs, alpha=MOMENTUM)  # Nesterov
            torch._foreach_add_(ps, upd, alpha=-lr)

    def state_dict(self):
        return {"buf": {n: b for n, b in self.buf.items()}}

    def load_state_dict(self, state):
        for n, b in state["buf"].items():
            self.buf[n].copy_(b)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
