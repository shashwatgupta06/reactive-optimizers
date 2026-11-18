"""Loss stats, per-role grad/param norms, validation passthrough."""
import math

import torch

EVERY = 50  # updates between calls


def _role(name, p):
    if p.ndim <= 1:
        return "nb"
    if name.startswith("fc."):
        return "head"
    return "conv"


def analyze(obs, memory):
    losses = obs.losses
    n = max(1, len(losses))
    mean = sum(losses) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / n)

    sq = {}
    with torch.no_grad():
        for name, g in obs.grads.items():
            if g is None:
                continue
            p = obs.params[name]
            r = _role(name, p)
            a = sq.setdefault(r, [0.0, 0.0])
            a[0] += float(g.float().pow(2).sum())
            a[1] += float(p.float().pow(2).sum())

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "gnorm": {r: round(math.sqrt(v[0]), 4) for r, v in sq.items()},
        "pnorm": {r: round(math.sqrt(v[1]), 2) for r, v in sq.items()},
        "lr": [round(float(g.get("lr", 0.0)), 5) for g in getattr(obs.updater, "param_groups", [])],
        "ms": round(float(obs.wall_ms_per_update or 0.0), 1),
    }
    if obs.validation:
        report["val"] = [
            {"u": v["update"], "acc": round(v["val_acc"], 2), "loss": round(v["val_loss"], 4)}
            for v in obs.validation
        ]
    return {"report": report}
