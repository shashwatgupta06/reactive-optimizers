"""Own SGD-Nesterov updater (foreach), per-role settings, schedule as function of absolute update.

Version 1: numerically the same recipe as the starting optimizer (lr 0.05, wd 5e-4 on ndim>1,
warmup 570, cosine to 0), but owned here so later decisions can change roles/schedules.
Momentum buffers are carried over from the previous updater (torch SGD state or our own `buf`).
"""
import math

import torch

WARMUP = 570


def role(name, p):
    if p.ndim <= 1:
        return "norm_bias"
    if name.startswith("fc."):
        return "head"
    if name.startswith("conv1."):
        return "stem"
    return "conv"


SETTINGS = {  # per role: peak lr, weight decay
    "stem": dict(lr=0.05, wd=5e-4),
    "conv": dict(lr=0.05, wd=5e-4),
    "head": dict(lr=0.05, wd=5e-4),
    "norm_bias": dict(lr=0.05, wd=0.0),
}
MOMENTUM = 0.9


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.names = [n for n, _ in named_params]
        self.params = dict(named_params)
        self.roles = {n: role(n, p) for n, p in named_params}
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}
        self._carry(previous)
        self.groups = {}
        for r in SETTINGS:
            ns = [n for n in self.names if self.roles[n] == r]
            self.groups[r] = dict(names=ns, params=[self.params[n] for n in ns],
                                  bufs=[self.buf[n] for n in ns])
        self.param_groups = [{"name": r, "params": g["params"], "lr": SETTINGS[r]["lr"]}
                             for r, g in self.groups.items()]
        dev = next(iter(self.params.values())).device
        self.loss_ema = torch.zeros((), device=dev)

    @torch.no_grad()
    def _carry(self, previous):
        if previous is None:
            return
        old = getattr(previous, "buf", None)
        if isinstance(old, dict):
            for n, b in self.buf.items():
                if n in old and old[n].shape == b.shape:
                    b.copy_(old[n])
            return
        opt = getattr(previous, "opt", None)
        if opt is not None:
            for n, p in self.params.items():
                st = opt.state.get(p, {})
                mb = st.get("momentum_buffer")
                if mb is not None:
                    self.buf[n].copy_(mb)

    def factor(self, update):
        if update <= WARMUP:
            return update / WARMUP
        t = (update - WARMUP) / max(1, self.total - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        f = self.factor(update)
        for pg in self.param_groups:
            pg["lr"] = SETTINGS[pg["name"]]["lr"] * f
        for r, g in self.groups.items():
            s = SETTINGS[r]
            ps, bs = g["params"], g["bufs"]
            gs = [grads[n] for n in g["names"]]
            if s["wd"]:
                d = torch._foreach_add(gs, ps, alpha=s["wd"])
            else:
                d = torch._foreach_mul(gs, 1.0)  # copy; grads are not modified
            torch._foreach_mul_(bs, MOMENTUM)
            torch._foreach_add_(bs, d)
            torch._foreach_add_(d, bs, alpha=MOMENTUM)
            torch._foreach_add_(ps, d, alpha=-s["lr"] * f)
        if ctx is not None and getattr(ctx, "loss", None) is not None:
            self.loss_ema.mul_(0.98).add_(ctx.loss.detach().float(), alpha=0.02)

    def state_dict(self):
        return {"buf": {n: b.clone() for n, b in self.buf.items()}, "loss_ema": self.loss_ema.clone()}

    def load_state_dict(self, state):
        for n, b in state["buf"].items():
            self.buf[n].copy_(b)
        if "loss_ema" in state:
            self.loss_ema.copy_(state["loss_ema"])


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
