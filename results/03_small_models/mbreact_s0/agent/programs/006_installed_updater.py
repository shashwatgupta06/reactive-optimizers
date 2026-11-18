"""AdamW (foreach) with per-role groups and own schedule.

Roles: head_w (classifier.weight), head_b (classifier.bias), weight (conv kernels), no_decay (BN, biases).
Schedule: per-group peak lr * factor(update). Backbone: linear warmup to WARMUP, cosine to 0 at total.
Head: fast warmup (HEAD_WARMUP) then same cosine.
"""
import math

import torch

_S = 0.252  # decision 11: x0.6 (val plateau + val loss rising); decision 14: x0.6 again (plateau); decision 17: x0.7
LR = {"head_w": 2.5e-3 * _S, "head_b": 2.5e-3 * _S,
      "weight": 4e-4 * _S, "no_decay": 4e-4 * _S,
      "weight_early": 2e-4 * _S, "no_decay_early": 2e-4 * _S}
WD = {"head_w": 0.05, "head_b": 0.0, "weight": 0.05, "no_decay": 0.0,
      "weight_early": 0.05, "no_decay_early": 0.0}
BETAS = (0.9, 0.999)
EPS = 1e-8
WARMUP = 570
HEAD_WARMUP = 114
AVG_START = 8000     # start uniform weight averaging
SWAP_AT = 11100      # copy the average into params; ~300 near-zero-lr updates refresh BN stats; val at 11172 shows effect
REVERT_SWAP = False  # set True in a later build to restore the pre-swap weights
EARLY = ("conv_stem", "bn1.", "blocks.0.", "blocks.1.", "blocks.2.")


def role(name, p):
    if name.startswith("classifier."):
        return "head_w" if p.ndim > 1 else "head_b"
    r = "weight" if p.ndim > 1 else "no_decay"
    if name.startswith(EARLY):
        r += "_early"
    return r


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.names = [n for n, _ in named_params]
        self.params = dict(named_params)
        self.roles = {n: role(n, p) for n, p in named_params}
        self.m = {n: torch.zeros_like(p) for n, p in named_params}
        self.v = {n: torch.zeros_like(p) for n, p in named_params}
        self.t = 0
        self._carry(previous, spec)
        # tail weight averaging (uniform, SWA-style); swapped into params at SWAP_AT
        self.avg, self.pre = None, None
        self.n_avg, self.swapped, self.reverted = 0, False, False
        if previous is not None and getattr(previous, "avg", None) is not None:
            self.avg = {n: t.clone() for n, t in previous.avg.items()}
            self.n_avg = int(previous.n_avg)
        if previous is not None and getattr(previous, "pre", None) is not None:
            self.pre = {n: t.clone() for n, t in previous.pre.items()}
        if previous is not None:
            self.swapped = bool(getattr(previous, "swapped", False))
            self.reverted = bool(getattr(previous, "reverted", False))
        self.group_names = {r: [n for n in self.names if self.roles[n] == r] for r in LR}
        self.param_groups = [
            {"name": r, "params": [self.params[n] for n in ns], "lr": LR[r]}
            for r, ns in self.group_names.items()
        ]

    def _carry(self, previous, spec):
        if previous is None:
            return
        old_m, old_v = getattr(previous, "m", None), getattr(previous, "v", None)
        if isinstance(old_m, dict) and isinstance(old_v, dict):
            for n in self.names:
                if n in old_m:
                    self.m[n].copy_(old_m[n])
                    self.v[n].copy_(old_v[n])
            self.t = int(getattr(previous, "t", 0))
            return
        opt = getattr(previous, "opt", None)
        if opt is None:
            return
        steps = []
        for n in self.names:
            st = opt.state.get(self.params[n], None)
            if st and "exp_avg" in st:
                self.m[n].copy_(st["exp_avg"])
                self.v[n].copy_(st["exp_avg_sq"])
                steps.append(int(float(st["step"])))
        if steps:
            self.t = max(steps)

    def factor(self, r, u):
        w = HEAD_WARMUP if r.startswith("head") else WARMUP
        if u <= w:
            return u / w
        if u <= WARMUP:
            return 1.0
        t = (u - WARMUP) / max(1, self.total - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        self.t += 1
        b1, b2 = BETAS
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for g_ in self.param_groups:
            r = g_["name"]
            ns = [n for n in self.group_names[r] if grads.get(n) is not None]
            if not ns:
                continue
            lr = LR[r] * self.factor(r, update)
            g_["lr"] = lr
            ps = [self.params[n] for n in ns]
            gs = [grads[n] for n in ns]
            ms = [self.m[n] for n in ns]
            vs = [self.v[n] for n in ns]
            if WD[r]:
                torch._foreach_mul_(ps, 1 - lr * WD[r])
            torch._foreach_mul_(ms, b1)
            torch._foreach_add_(ms, gs, alpha=1 - b1)
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, gs, gs, value=1 - b2)
            den = torch._foreach_sqrt(vs)
            torch._foreach_div_(den, math.sqrt(c2))
            torch._foreach_add_(den, EPS)
            torch._foreach_addcdiv_(ps, ms, den, value=-lr / c1)
        self._average(update)

    @torch.no_grad()
    def _average(self, update):
        if update < AVG_START:
            return
        names = self.names
        ps = [self.params[n] for n in names]
        if AVG_START <= update < SWAP_AT:
            if self.avg is None:
                self.avg = {n: self.params[n].detach().clone() for n in names}
                self.n_avg = 1
            else:
                self.n_avg += 1
                a = [self.avg[n] for n in names]
                w = 1.0 / self.n_avg
                torch._foreach_mul_(a, 1.0 - w)
                torch._foreach_add_(a, ps, alpha=w)
        if update >= SWAP_AT and not self.swapped and self.avg is not None and not REVERT_SWAP:
            self.pre = {n: self.params[n].detach().clone() for n in names}
            for n in names:
                self.params[n].copy_(self.avg[n])
            self.swapped = True
        if REVERT_SWAP and self.swapped and not self.reverted and self.pre is not None:
            for n in names:
                self.params[n].copy_(self.pre[n])
            self.reverted = True

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": self.t,
                "avg": self.avg if self.avg is not None else {}, "n_avg": self.n_avg,
                "pre": self.pre if self.pre is not None else {},
                "swapped": self.swapped, "reverted": self.reverted}

    def load_state_dict(self, state):
        for n in self.m:
            self.m[n].copy_(state["m"][n])
            self.v[n].copy_(state["v"][n])
        self.t = int(state["t"])
        avg = state.get("avg") or {}
        self.avg = {n: t.clone() for n, t in avg.items()} if avg else None
        self.n_avg = int(state.get("n_avg", 0))
        pre = state.get("pre") or {}
        self.pre = {n: t.clone() for n, t in pre.items()} if pre else None
        self.swapped = bool(state.get("swapped", False))
        self.reverted = bool(state.get("reverted", False))


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
