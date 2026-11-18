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
    # residual-branch strength: norm of bn3 gains per stage (zero-init at start)
    with torch.no_grad():
        st = {}
        for name, p in obs.params.items():
            if name.endswith("bn3.weight"):
                s = name.split(".")[0]
                a = st.setdefault(s, [0.0, 0])
                a[0] += float(p.float().abs().mean())
                a[1] += 1
        report["bn3_absmean"] = {s: round(v[0] / v[1], 4) for s, v in st.items()}
        # momentum-buffer step relative to weight norm, conv role
        buf = getattr(obs.updater, "buf", None)
        if isinstance(buf, dict):
            bsq = sum(float(b.float().pow(2).sum()) for n, b in buf.items()
                      if _role(n, obs.params[n]) == "conv")
            report["conv_buf_norm"] = round(math.sqrt(bsq), 3)
    if obs.validation:
        report["val"] = [
            {"u": v["update"], "acc": round(v["val_acc"], 2), "loss": round(v["val_loss"], 4)}
            for v in obs.validation
        ]
    out = {"report": report}
    # one-shot wakes to act on fresh validation results promptly
    for u_wake in (1250,):
        key = f"woke_{u_wake}"
        if obs.update >= u_wake and not memory.get(key):
            memory[key] = True
            out["wake"] = f"review val@epoch10 (scheduled wake at {u_wake})"
    return out
