"""Analysis: loss stats, per-stage grad/param norms, relative weight change per interval, validation."""
import math

import torch

EVERY = 50


def stage(name):
    if name.startswith("fc."):
        return "fc"
    if name.startswith("conv1.") or name.startswith("bn1."):
        return "stem"
    return name.split(".")[0]  # layer1..layer4


@torch.no_grad()
def analyze(obs, memory):
    losses = obs.losses
    n = max(1, len(losses))
    mean = sum(losses) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / n)

    names = [k for k, g in obs.grads.items() if g is not None]
    gn = torch.stack([obs.grads[k].float().norm() for k in names]).cpu().tolist()
    pn = torch.stack([obs.params[k].float().norm() for k in names]).cpu().tolist()

    snap = memory.get("snap")
    dn = None
    if snap is not None:
        dn = torch.stack([(obs.params[k].float() - snap[k]).norm() for k in names]).cpu().tolist()
    memory["snap"] = {k: obs.params[k].detach().float().clone() for k in names}

    agg = {}
    for i, k in enumerate(names):
        kind = "w" if obs.params[k].ndim > 1 else "nb"
        key = f"{stage(k)}.{kind}"
        a = agg.setdefault(key, [0.0, 0.0, 0.0])
        a[0] += gn[i] ** 2
        a[1] += pn[i] ** 2
        if dn is not None:
            a[2] += dn[i] ** 2
    per = {}
    for key, (g2, p2, d2) in agg.items():
        e = {"g": round(math.sqrt(g2), 4), "p": round(math.sqrt(p2), 2)}
        if dn is not None:
            e["drel"] = round(math.sqrt(d2) / max(1e-12, math.sqrt(p2)), 5)
        per[key] = e

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "loss_min": round(min(losses), 4) if losses else None,
        "grad_norm": round(math.sqrt(sum(x * x for x in gn)), 4),
        "param_norm": round(math.sqrt(sum(x * x for x in pn)), 3),
        "per_stage": per,
        "lrs": [round(float(g["lr"]), 6) for g in getattr(obs.updater, "param_groups", [])],
        "ms": round(obs.wall_ms_per_update, 1) if obs.wall_ms_per_update else None,
    }
    if obs.validation:
        report["val"] = [{"ep": v["epoch"], "loss": round(v["val_loss"], 4), "acc": round(v["val_acc"], 2)}
                         for v in obs.validation]
    return {"report": report}
