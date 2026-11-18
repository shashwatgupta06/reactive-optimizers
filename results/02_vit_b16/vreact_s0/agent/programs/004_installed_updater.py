"""AdamW with layer-wise LR decay, a faster head, own absolute-index schedule, plus weight EMA.

Groups: one per (depth, decay?) plus the head. Adam moments are carried over from the previous
updater by parameter identity; EMA is carried over too if present.

EMA: from update ema_start, ema <- lerp(ema, params, 1-ema_decay) after each step.
Probe: at updates in `probes` (ends of even epochs, right before validation) the EMA weights are
swapped into the params; at the next step the raw weights are restored before the optimizer step.
Final: if ema_final, the EMA is copied into the params at the last update.
"""
import math
import re

import torch

CFG = {
    "backbone_lr": 1e-4,      # peak lr for the top encoder layer / final LN
    "layer_decay": 0.85,      # lr multiplier per depth step down
    "head_lr": 1e-3,          # peak lr of heads.head
    "wd": 0.05,
    "head_wd": 0.05,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "warmup": 570,
    "head_warmup": 200,
    "clip": None,
    "ema_start": 3460,
    "ema_decay": 0.995,       # horizon ~200 updates; shortened at u9868 to cut lag in the low-lr tail
    # ends of epochs 40, 44, ..., 96 (validation follows); raw val at 42, 46, ..., 98, 100
    "probes": [114 * e for e in range(40, 97, 4)],
    "ema_final": True,
}


def depth(name):
    m = re.search(r"encoder_layer_(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if name.startswith(("conv_proj", "class_token", "encoder.pos_embedding")):
        return 0
    return 13


def no_decay(name, p):
    return p.ndim <= 1 or name.endswith(("pos_embedding", "class_token"))


class Updater:
    def __init__(self, named_params, previous, spec):
        self.total = spec["total_updates"]
        self.cfg = dict(CFG)
        c = self.cfg
        groups = {}
        for n, p in named_params:
            if n.startswith("heads."):
                key = ("head", no_decay(n, p))
                lr = c["head_lr"]
                wd = 0.0 if no_decay(n, p) else c["head_wd"]
            else:
                d = depth(n)
                key = (d, no_decay(n, p))
                lr = c["backbone_lr"] * c["layer_decay"] ** (13 - d)
                wd = 0.0 if no_decay(n, p) else c["wd"]
            if key not in groups:
                groups[key] = {"params": [], "names": [], "lr": lr, "base_lr": lr, "weight_decay": wd,
                               "is_head": key[0] == "head", "name": str(key)}
            groups[key]["params"].append(p)
            groups[key]["names"].append(n)
        self.opt = torch.optim.AdamW(list(groups.values()), lr=c["backbone_lr"], betas=c["betas"],
                                     eps=c["eps"], foreach=True)
        self.params = [p for _, p in named_params]
        old_opt = getattr(previous, "opt", None)
        if old_opt is not None:
            for p in self.params:
                st = old_opt.state.get(p)
                if st:
                    self.opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
        self.param_groups = self.opt.param_groups

        # EMA state
        self.ema = None
        self.ema_count = 0
        self.raw = None
        self.swapped = False
        self.probe_log = []   # updates at which EMA was swapped in
        old_ema = getattr(previous, "ema", None)
        if old_ema is not None:
            self.ema = [t.clone() for t in old_ema]
            self.ema_count = int(getattr(previous, "ema_count", 0))
            self.probe_log = list(getattr(previous, "probe_log", []))
            if getattr(previous, "swapped", False):
                # previous left EMA weights in params; carry its raw copy so we restore it next step
                self.raw = [t.clone() for t in previous.raw]
                self.swapped = True

    def factor(self, update, warmup):
        if update <= warmup:
            return update / warmup
        t = (update - warmup) / max(1, self.total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        c = self.cfg
        if self.swapped:
            torch._foreach_copy_(self.params, self.raw)
            self.swapped = False
        fb = self.factor(update, c["warmup"])
        fh = self.factor(update, c["head_warmup"])
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * (fh if g["is_head"] else fb)
        if c["clip"]:
            torch.nn.utils.clip_grad_norm_(self.params, c["clip"], foreach=True)
        self.opt.step()

        if c["ema_start"] is not None and update >= c["ema_start"]:
            if self.ema is None:
                self.ema = [p.detach().clone() for p in self.params]
            else:
                torch._foreach_lerp_(self.ema, self.params, 1.0 - c["ema_decay"])
            self.ema_count += 1

        if self.ema is not None:
            if update == self.total and c["ema_final"]:
                torch._foreach_copy_(self.params, self.ema)
            elif update in c["probes"] and update < self.total:
                if self.raw is None:
                    self.raw = [p.detach().clone() for p in self.params]
                else:
                    torch._foreach_copy_(self.raw, self.params)
                torch._foreach_copy_(self.params, self.ema)
                self.swapped = True
                self.probe_log.append(update)

    def state_dict(self):
        return {"opt": self.opt.state_dict(),
                "ema": self.ema, "ema_count": self.ema_count,
                "raw": self.raw if self.swapped else None, "swapped": self.swapped,
                "probe_log": list(self.probe_log)}

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])
        if state.get("ema") is not None:
            if self.ema is None:
                self.ema = [t.clone() for t in state["ema"]]
            else:
                torch._foreach_copy_(self.ema, state["ema"])
        else:
            self.ema = None
        self.ema_count = int(state.get("ema_count", 0))
        self.probe_log = list(state.get("probe_log", []))
        self.swapped = bool(state.get("swapped", False)) and state.get("raw") is not None
        if self.swapped:
            self.raw = [t.clone() for t in state["raw"]]


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
