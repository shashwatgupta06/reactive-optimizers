"""Loss stats, per-role grad/param norms, relative weight change since last call, validation."""
import math

import torch

EVERY = 57  # half an epoch


def role(name, p):
    if name.startswith("classifier."):
        return "head"
    if name.startswith(("conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2")):
        return "early" if p.ndim > 1 else "early_nd"
    return "late" if p.ndim > 1 else "late_nd"


@torch.no_grad()
def analyze(obs, memory):
    losses = obs.losses
    n = max(1, len(losses))
    mean = sum(losses) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / n)

    acc = {}
    prev = memory.get("prev")
    newprev = {}
    for name, p in obs.params.items():
        r = role(name, p)
        a = acc.setdefault(r, [0.0, 0.0, 0.0])
        g = obs.grads.get(name)
        if g is not None:
            a[0] += float(g.float().pow(2).sum())
        a[1] += float(p.float().pow(2).sum())
        if prev is not None and name in prev:
            a[2] += float((p.float() - prev[name]).pow(2).sum())
        newprev[name] = p.detach().float().clone()
    memory["prev"] = newprev

    groups = {
        r: {"g": round(math.sqrt(v[0]), 4), "p": round(math.sqrt(v[1]), 2),
            "dW/W": round(math.sqrt(v[2]) / max(1e-12, math.sqrt(v[1])), 5)}
        for r, v in acc.items()
    }
    lrs = {}
    for g in getattr(obs.updater, "param_groups", []) or []:
        lrs[g.get("name", str(len(lrs)))] = float(g["lr"])

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "groups": groups,
        "lrs": {k: float(f"{v:.3g}") for k, v in lrs.items()},
        "val": [(v["epoch"], round(v["val_loss"], 4), round(v["val_acc"], 2)) for v in obs.validation],
        "ms": obs.wall_ms_per_update,
    }
    out = {"report": report}
    hist = memory.setdefault("val_hist", [])
    for v in obs.validation:
        hist.append((v["epoch"], v["val_acc"], v["val_loss"]))
        # wake if val is poor after the lr cut (expect >= 38% by epoch 6), or if val drops sharply from its best
        best = max(h[1] for h in hist)
        if (v["epoch"] >= 6 and v["val_acc"] < 38.0) or (v["val_acc"] < best - 4.0):
            out["wake"] = f"val_acc {v['val_acc']:.2f} at epoch {v['epoch']} (best {best:.2f})"
    return out
