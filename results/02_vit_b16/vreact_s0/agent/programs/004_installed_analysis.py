"""Loss, validation, per-depth grad norms, parameter drift from init, current lrs."""
import math
import re

import torch

EVERY = 50


def _depth(name):
    m = re.search(r"encoder_layer_(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if name.startswith(("conv_proj", "class_token", "encoder.pos_embedding")):
        return 0
    if name.startswith("heads."):
        return 14
    return 13


@torch.no_grad()
def analyze(obs, memory):
    losses = obs.losses
    mean = sum(losses) / max(1, len(losses))
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / max(1, len(losses)))

    if "init" not in memory:
        memory["init"] = {n: p.detach().float().clone() for n, p in obs.params.items()
                          if not n.startswith("heads.")}
    gsq, drift, pnorm = {}, {}, {}
    for n, g in obs.grads.items():
        d = _depth(n)
        if g is not None:
            gsq[d] = gsq.get(d, 0.0) + float(g.float().pow(2).sum())
        p = obs.params[n].float()
        pnorm[d] = pnorm.get(d, 0.0) + float(p.pow(2).sum())
        if n in memory["init"]:
            drift[d] = drift.get(d, 0.0) + float((p - memory["init"][n]).pow(2).sum())
    total_g = math.sqrt(sum(gsq.values()))
    rel_drift = {d: round(math.sqrt(drift[d] / max(pnorm[d], 1e-12)), 4) for d in sorted(drift)}

    lrs = {}
    for g in getattr(obs.updater, "param_groups", []):
        lrs[g.get("name", str(len(lrs)))] = g["lr"]
    lr_summary = {"max": max(lrs.values()) if lrs else None, "min": min(lrs.values()) if lrs else None}

    probes = set(getattr(obs.updater, "probe_log", []) or [])
    vals = [{"u": v["update"], "ep": v["epoch"], "loss": round(v["val_loss"], 4), "acc": round(v["val_acc"], 2),
             "ema_probe": v["update"] in probes}
            for v in obs.validation]
    memory.setdefault("best_val", 0.0)
    for v in obs.validation:
        if v["update"] not in probes:
            memory["best_val"] = max(memory["best_val"], v["val_acc"])
    ema = getattr(obs.updater, "ema", None)
    ema_info = None
    if ema is not None:
        params = list(obs.params.values())
        dsq, psq = 0.0, 0.0
        for e, p in zip(ema, params):
            dsq += float((e.float() - p.float()).pow(2).sum())
            psq += float(p.float().pow(2).sum())
        ema_info = {"count": getattr(obs.updater, "ema_count", None),
                    "rel_dist": round(math.sqrt(dsq / max(psq, 1e-12)), 5),
                    "swapped_now": bool(getattr(obs.updater, "swapped", False))}

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "grad_norm": round(total_g, 4),
        "grad_by_depth": {d: round(math.sqrt(v), 4) for d, v in sorted(gsq.items())},
        "rel_drift_by_depth": rel_drift,
        "head_norm": round(math.sqrt(pnorm.get(14, 0.0)), 3),
        "lr": lr_summary,
        "val": vals,
        "best_val": round(memory["best_val"], 2),
        "ema": ema_info,
        "ms": obs.wall_ms_per_update,
    }
    out = {"report": report}
    nonfinite = any(not math.isfinite(x) for x in losses)
    if obs.validation and obs.validation[-1]["update"] not in probes and \
            obs.validation[-1]["val_acc"] < memory["best_val"] - 4.0:
        out["wake"] = f"val dropped to {obs.validation[-1]['val_acc']:.2f} (best {memory['best_val']:.2f})"
    elif nonfinite:
        out["wake"] = "non-finite training loss"
    return out
