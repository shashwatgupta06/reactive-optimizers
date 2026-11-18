"""Loss stats, grad norms by section, distance moved from init per section, val tracking."""
import math
import re

import torch

EVERY = 50


def section(name):
    if name.startswith("head."):
        return "head"
    m = re.match(r"blocks\.(\d+)\.", name)
    if m:
        i = int(m.group(1))
        return "b0-3" if i < 4 else ("b4-7" if i < 8 else "b8-11")
    if name.startswith("norm."):
        return "b8-11"
    return "embed"


@torch.no_grad()
def analyze(obs, memory):
    losses = obs.losses
    mean = sum(losses) / max(1, len(losses))
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / max(1, len(losses)))

    if "init" not in memory:
        memory["init"] = {n: p.detach().clone() for n, p in obs.params.items()}
    gsq, dsq, psq = {}, {}, {}
    for name, p in obs.params.items():
        s = section(name)
        g = obs.grads.get(name)
        if g is not None:
            gsq[s] = gsq.get(s, 0.0) + float(g.float().pow(2).sum())
        dsq[s] = dsq.get(s, 0.0) + float((p - memory["init"][name]).float().pow(2).sum())
        psq[s] = psq.get(s, 0.0) + float(memory["init"][name].float().pow(2).sum())

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "gnorm": {k: round(math.sqrt(v), 3) for k, v in gsq.items()},
        "rel_move": {k: round(math.sqrt(dsq[k]) / max(1e-12, math.sqrt(psq[k])), 4) for k in dsq},
    }
    up = obs.updater
    if hasattr(up, "last_gnorm"):
        report["gnorm_preclip"] = round(float(up.last_gnorm), 3)
        report["clip_coef"] = round(float(up.clip_coef), 3)
    if hasattr(up, "factor"):
        try:
            report["lr_factor"] = round(up.factor(obs.update), 4)
        except Exception:
            pass
    if obs.validation:
        report["val"] = [(v["update"], round(v["val_loss"], 4), round(v["val_acc"], 2)) for v in obs.validation]
    return {"report": report}
