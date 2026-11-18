"""AdamW with layer-wise LR decay, a faster head, and own absolute-index schedule.

Groups: one per (depth, decay?) plus the head. Adam moments are carried over from the previous
updater (user optimizer or a previous version of this one) by parameter identity.
"""
import math
import re

import torch

CFG = {
    "backbone_lr": 1e-4,      # peak lr for the top encoder layer / final LN
    "layer_decay": 0.85,      # lr multiplier per depth step down
    "head_lr": 1e-3,          # peak lr of heads.head (zero-init, needs to move fast)
    "wd": 0.05,
    "head_wd": 0.05,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "warmup": 570,            # backbone warmup (absolute updates)
    "head_warmup": 200,
    "clip": None,             # global grad-norm clip, None = off
}


def depth(name):
    m = re.search(r"encoder_layer_(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if name.startswith(("conv_proj", "class_token", "encoder.pos_embedding")):
        return 0
    return 13  # encoder.ln, head


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
        # carry Adam state
        old_opt = getattr(previous, "opt", None)
        if old_opt is not None:
            for p in self.params:
                st = old_opt.state.get(p)
                if st:
                    self.opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
        self.param_groups = self.opt.param_groups

    def factor(self, update, warmup):
        if update <= warmup:
            return update / warmup
        t = (update - warmup) / max(1, self.total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        c = self.cfg
        fb = self.factor(update, c["warmup"])
        fh = self.factor(update, c["head_warmup"])
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * (fh if g["is_head"] else fb)
        if c["clip"]:
            torch.nn.utils.clip_grad_norm_(self.params, c["clip"], foreach=True)
        self.opt.step()

    def state_dict(self):
        return {"opt": self.opt.state_dict()}

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
