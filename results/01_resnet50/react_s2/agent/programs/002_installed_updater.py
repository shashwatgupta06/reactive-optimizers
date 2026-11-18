"""Self-contained SGD-Nesterov with per-role settings, foreach implementation.

v1: same schedule (warmup 570, cosine to 0) and peak lr 0.05; weight decay on conv/fc weights
raised 5e-4 -> 1e-3 (small dataset from scratch: more regularisation, faster effective lr for
BN-scaled conv weights).
"""
import math

import torch

WARMUP = 570
MOMENTUM = 0.9
# v2 (installed at u1250): peak lr 0.05 -> 0.1, ramped linearly over updates 1250..1550,
# wd 1e-3 -> 7.5e-4 (lr*wd 5e-5 -> 7.5e-5 at the new peak).
RAMP_START, RAMP_END, LR_MULT = 1250, 1550, 2.0
SETTINGS = {  # role: base lr (multiplied by peak_mult), weight decay
    "conv": dict(lr=0.05, wd=7.5e-4),
    "head": dict(lr=0.05, wd=7.5e-4),
    "norm_bias": dict(lr=0.05, wd=0.0),
}


def peak_mult(update):
    if update <= RAMP_START:
        return 1.0
    if update >= RAMP_END:
        return LR_MULT
    return 1.0 + (LR_MULT - 1.0) * (update - RAMP_START) / (RAMP_END - RAMP_START)


def role(name, p):
    if p.ndim <= 1:
        return "norm_bias"
    if name.startswith("fc."):
        return "head"
    return "conv"


def lr_factor(update, total):
    if update <= WARMUP:
        return update / WARMUP
    t = (update - WARMUP) / max(1, total - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.names = [n for n, _ in named_params]
        self.params = dict(named_params)
        self.roles = {n: role(n, p) for n, p in named_params}
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}
        # carry momentum: from our own kind of updater, or from the user's torch SGD
        old = getattr(previous, "buf", None)
        if isinstance(old, dict):
            for n in self.names:
                if n in old and old[n].shape == self.buf[n].shape:
                    self.buf[n].copy_(old[n])
        else:
            opt = getattr(previous, "opt", None) or spec.get("user_optimizer")
            if opt is not None:
                for n, p in named_params:
                    st = opt.state.get(p, {})
                    mb = st.get("momentum_buffer")
                    if mb is not None and mb.shape == p.shape:
                        self.buf[n].copy_(mb)
        self.groups = {}
        for r in SETTINGS:
            ns = [n for n in self.names if self.roles[n] == r]
            self.groups[r] = dict(
                names=ns,
                params=[self.params[n] for n in ns],
                bufs=[self.buf[n] for n in ns],
            )
        self.param_groups = [
            {"name": r, "params": g["params"], "lr": SETTINGS[r]["lr"]} for r, g in self.groups.items()
        ]

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        f = lr_factor(update, self.total) * peak_mult(update)
        for pg in self.param_groups:
            r = pg["name"]
            s = SETTINGS[r]
            g = self.groups[r]
            lr = s["lr"] * f
            pg["lr"] = lr
            gs = [grads[n] for n in g["names"]]
            if s["wd"]:
                d = torch._foreach_add(gs, g["params"], alpha=s["wd"])
            else:
                d = gs
            torch._foreach_mul_(g["bufs"], MOMENTUM)
            torch._foreach_add_(g["bufs"], d)
            # nesterov: p -= lr * (d + m * buf)
            upd = torch._foreach_add(d, g["bufs"], alpha=MOMENTUM)
            torch._foreach_add_(g["params"], upd, alpha=-lr)

    def state_dict(self):
        return {"buf": {n: b for n, b in self.buf.items()}}

    def load_state_dict(self, state):
        for n, b in state["buf"].items():
            self.buf[n].copy_(b)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
