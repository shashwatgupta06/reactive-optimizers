"""SGD + Nesterov, per-role settings, stronger regularisation (lr*wd), global-norm clip as safety.

Schedule: linear warmup to WARMUP (absolute update index), then cosine to 0 at total_updates.
"""
import math

import torch

PEAK_LR = {"conv": 0.08, "stem": 0.08, "head": 0.08, "norm_bias": 0.08}
WD = {"conv": 8e-4, "stem": 8e-4, "head": 8e-4, "norm_bias": 0.0}
MOMENTUM = 0.9
WARMUP = 1140
CLIP = 5.0


def role(name, p):
    if p.ndim <= 1:
        return "norm_bias"
    if name.startswith("fc."):
        return "head"
    if name.startswith("conv1."):
        return "stem"
    return "conv"


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.names = [n for n, _ in named_params]
        self.params = dict(named_params)
        self.roles = {n: role(n, p) for n, p in named_params}
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}
        # carry momentum: from a custom updater (buf) or from torch SGD state
        old = getattr(previous, "buf", None)
        if isinstance(old, dict):
            for n in self.buf:
                if n in old and old[n].shape == self.buf[n].shape:
                    self.buf[n].copy_(old[n])
        else:
            opt = getattr(previous, "opt", None)
            if opt is not None:
                for n, p in named_params:
                    st = opt.state.get(p, {})
                    mb = st.get("momentum_buffer") if isinstance(st, dict) else None
                    if mb is not None:
                        self.buf[n].copy_(mb)
        dev = next(iter(self.params.values())).device
        self.stats = {
            "gnorm_sum": torch.zeros((), device=dev),
            "clip_count": torch.zeros((), device=dev),
            "n": torch.zeros((), device=dev),
        }
        self.groups = {}
        for r in PEAK_LR:
            ns = [n for n in self.names if self.roles[n] == r]
            self.groups[r] = ns
        self.param_groups = [
            {"name": r, "params": [self.params[n] for n in ns], "lr": PEAK_LR[r]}
            for r, ns in self.groups.items()
        ]

    def factor(self, update):
        if update <= WARMUP:
            return update / WARMUP
        t = (update - WARMUP) / max(1, self.total - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        f = self.factor(update)
        all_g = [grads[n] for n in self.names if grads[n] is not None]
        norms = torch._foreach_norm(all_g)
        gnorm = torch.linalg.vector_norm(torch.stack(norms))
        scale = torch.clamp(CLIP / (gnorm + 1e-6), max=1.0)
        self.stats["gnorm_sum"].add_(gnorm)
        self.stats["clip_count"].add_((gnorm > CLIP).float())
        self.stats["n"].add_(1.0)
        torch._foreach_mul_(all_g, scale)

        for gi, (r, ns) in enumerate(self.groups.items()):
            lr = PEAK_LR[r] * f
            self.param_groups[gi]["lr"] = lr
            ns = [n for n in ns if grads[n] is not None]
            if not ns:
                continue
            ps = [self.params[n] for n in ns]
            gs = [grads[n] for n in ns]
            bs = [self.buf[n] for n in ns]
            if WD[r]:
                d = torch._foreach_add(gs, ps, alpha=WD[r])
            else:
                d = gs
            torch._foreach_mul_(bs, MOMENTUM)
            torch._foreach_add_(bs, d)
            # Nesterov: p -= lr * (d + m * buf)
            upd = torch._foreach_add(d, bs, alpha=MOMENTUM)
            torch._foreach_add_(ps, upd, alpha=-lr)

    def state_dict(self):
        return {"buf": self.buf, "stats": self.stats}

    def load_state_dict(self, state):
        for n, b in state["buf"].items():
            self.buf[n].copy_(b)
        for k, v in state.get("stats", {}).items():
            self.stats[k].copy_(v)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
