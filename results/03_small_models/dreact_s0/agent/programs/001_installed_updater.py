"""AdamW with layer-wise LR decay, higher head LR, global grad clipping, foreach implementation.

Schedule: linear warmup to WARMUP updates, then cosine to 0 at total_updates (absolute index).
Moments are carried over from the user AdamW (exp_avg/exp_avg_sq) or from a previous Updater.
"""
import math
import re

import torch

CFG = dict(
    top_lr=2e-4,        # lr of last block / final norm
    layer_decay=0.85,   # per-block multiplicative decay going down
    head_lr=2e-3,
    wd=0.05,
    head_wd=0.05,
    warmup=300,
    betas=(0.9, 0.999),
    eps=1e-8,
    clip=5.0,           # only catches spikes
    min_factor=0.0,
)
N_BLOCKS = 12


def layer_id(name):
    if name.startswith(("cls_token", "pos_embed", "patch_embed")):
        return 0
    m = re.match(r"blocks\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    return N_BLOCKS + 1  # norm.*, head.*


def no_decay(name, p):
    return p.ndim <= 1 or name in ("cls_token", "pos_embed")


class Updater:
    def __init__(self, named_params, previous, spec):
        self.cfg = dict(CFG)
        self.total = spec["total_updates"]
        self.names = [n for n, _ in named_params]
        self.params = dict(named_params)
        # build groups
        groups = {}
        for n, p in named_params:
            if n.startswith("head."):
                key = ("head", "nd" if no_decay(n, p) else "d")
                lr = self.cfg["head_lr"]
                wd = 0.0 if no_decay(n, p) else self.cfg["head_wd"]
            else:
                lid = layer_id(n)
                key = (f"L{lid}", "nd" if no_decay(n, p) else "d")
                lr = self.cfg["top_lr"] * self.cfg["layer_decay"] ** (N_BLOCKS + 1 - lid)
                wd = 0.0 if no_decay(n, p) else self.cfg["wd"]
            g = groups.setdefault(key, {"name": f"{key[0]}_{key[1]}", "names": [], "params": [],
                                        "base_lr": lr, "wd": wd, "lr": lr})
            g["names"].append(n)
            g["params"].append(p)
        self.param_groups = list(groups.values())

        self.m = {n: torch.zeros_like(p) for n, p in named_params}
        self.v = {n: torch.zeros_like(p) for n, p in named_params}
        self.t = 0
        # carry over state
        old_m, old_v = getattr(previous, "m", None), getattr(previous, "v", None)
        if isinstance(old_m, dict) and isinstance(old_v, dict):
            for n in self.m:
                if n in old_m and old_m[n].shape == self.m[n].shape:
                    self.m[n].copy_(old_m[n])
                    self.v[n].copy_(old_v[n])
            self.t = int(getattr(previous, "t", 0))
        else:
            opt = getattr(previous, "opt", None) or spec.get("user_optimizer")
            if opt is not None:
                steps = []
                for n, p in named_params:
                    st = opt.state.get(p, {})
                    if "exp_avg" in st:
                        self.m[n].copy_(st["exp_avg"])
                        self.v[n].copy_(st["exp_avg_sq"])
                        s = st.get("step", 0)
                        steps.append(int(s.item()) if torch.is_tensor(s) else int(s))
                if steps:
                    self.t = max(steps)
        self.clip_coef = torch.ones((), device=next(iter(self.params.values())).device)
        self.last_gnorm = torch.zeros_like(self.clip_coef)

    def factor(self, update):
        w = self.cfg["warmup"]
        if update <= w:
            return update / w
        t = min(1.0, (update - w) / max(1, self.total - w))
        mf = self.cfg["min_factor"]
        return mf + (1 - mf) * 0.5 * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        f = self.factor(update)
        self.t += 1
        b1, b2 = self.cfg["betas"]
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        eps = self.cfg["eps"]

        all_g = [grads[n] for n in self.names if grads.get(n) is not None]
        norms = torch._foreach_norm(all_g)
        gnorm = torch.linalg.vector_norm(torch.stack(norms))
        self.last_gnorm.copy_(gnorm)
        coef = torch.clamp(self.cfg["clip"] / (gnorm + 1e-6), max=1.0)
        self.clip_coef.copy_(coef)
        torch._foreach_mul_(all_g, coef)

        for g in self.param_groups:
            lr = g["base_lr"] * f
            g["lr"] = lr
            names = [n for n in g["names"] if grads.get(n) is not None]
            if not names:
                continue
            ps = [self.params[n] for n in names]
            gs = [grads[n] for n in names]
            ms = [self.m[n] for n in names]
            vs = [self.v[n] for n in names]
            if g["wd"] and lr > 0:
                torch._foreach_mul_(ps, 1 - lr * g["wd"])
            torch._foreach_lerp_(ms, gs, 1 - b1)
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, gs, gs, value=1 - b2)
            denom = torch._foreach_sqrt(vs)
            torch._foreach_div_(denom, math.sqrt(c2))
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(ps, ms, denom, value=-lr / c1)

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load_state_dict(self, state):
        for n in self.m:
            self.m[n].copy_(state["m"][n])
            self.v[n].copy_(state["v"][n])
        self.t = int(state["t"])


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
