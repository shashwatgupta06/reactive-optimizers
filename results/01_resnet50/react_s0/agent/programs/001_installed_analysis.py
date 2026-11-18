"""Loss stats, per-stage grad/param norms, momentum-update/param ratio, validation."""
import math

import torch

EVERY = 50


def stage(name):
    if name.startswith("fc."):
        return "fc"
    if name.startswith("layer"):
        return name.split(".")[0]
    return "stem"


@torch.no_grad()
def analyze(obs, memory):
    losses = obs.losses
    n = max(1, len(losses))
    mean = sum(losses) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / n)

    names = [k for k, g in obs.grads.items() if g is not None]
    gn = torch.stack([obs.grads[k].float().norm() for k in names])
    pn = torch.stack([obs.params[k].float().norm() for k in names])
    buf = getattr(obs.updater, "buf", None)
    bn_ = torch.stack([buf[k].float().norm() for k in names]) if isinstance(buf, dict) else None
    gn, pn = gn.tolist(), pn.tolist()
    bn_ = bn_.tolist() if bn_ is not None else None

    per = {}
    for i, k in enumerate(names):
        kind = "w" if obs.params[k].ndim > 1 else "b"
        key = f"{stage(k)}.{kind}"
        d = per.setdefault(key, [0.0, 0.0, 0.0])
        d[0] += gn[i] ** 2
        d[1] += pn[i] ** 2
        if bn_ is not None:
            d[2] += bn_[i] ** 2
    stages = {k: {"g": round(math.sqrt(v[0]), 4), "p": round(math.sqrt(v[1]), 2),
                  "m/p": round(math.sqrt(v[2]) / max(1e-12, math.sqrt(v[1])), 5)} for k, v in per.items()}

    lrs = {}
    for g in getattr(obs.updater, "param_groups", []) or []:
        lrs[g.get("name", str(len(lrs)))] = round(float(g["lr"]), 6)

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "grad_norm": round(math.sqrt(sum(x * x for x in gn)), 4),
        "param_norm": round(math.sqrt(sum(x * x for x in pn)), 3),
        "lrs": lrs,
        "stages": stages,
        "ms": round(obs.wall_ms_per_update, 1) if obs.wall_ms_per_update else None,
    }
    if obs.validation:
        report["val"] = [{"ep": v["epoch"], "loss": round(v["val_loss"], 4), "acc": round(v["val_acc"], 2)}
                         for v in obs.validation]
    out = {"report": report}
    if not all(math.isfinite(x) for x in losses):
        out["wake"] = "non-finite loss"
    return out
