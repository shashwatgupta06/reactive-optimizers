"""Loss, gradient and weight statistics, clip activity, validation."""
import math

EVERY = 50


def _stage(name):
    if name.startswith("layer"):
        return name.split(".")[0]
    if name.startswith("fc"):
        return "fc"
    return "stem"


def analyze(obs, memory):
    losses = obs.losses
    mean = sum(losses) / max(1, len(losses))
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / max(1, len(losses)))

    grad_sq, param_sq = 0.0, 0.0
    stage_w = {}
    for name, g in obs.grads.items():
        if g is None:
            continue
        grad_sq += float(g.float().pow(2).sum())
        p = obs.params[name]
        ps = float(p.float().pow(2).sum())
        param_sq += ps
        if p.ndim > 1:
            s = _stage(name)
            stage_w[s] = stage_w.get(s, 0.0) + ps

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "grad_norm": round(math.sqrt(grad_sq), 4),
        "param_norm": round(math.sqrt(param_sq), 3),
        "w_norm": {k: round(math.sqrt(v), 2) for k, v in stage_w.items()},
    }
    up = obs.updater
    st = getattr(up, "stats", None)
    if isinstance(st, dict) and "n" in st:
        cur = {k: float(v) for k, v in st.items()}
        prev = memory.get("stats_prev", {k: 0.0 for k in cur})
        dn = cur["n"] - prev.get("n", 0.0)
        if dn > 0:
            report["mean_pre_clip_gnorm"] = round((cur["gnorm_sum"] - prev.get("gnorm_sum", 0.0)) / dn, 3)
            report["clip_frac"] = round((cur["clip_count"] - prev.get("clip_count", 0.0)) / dn, 3)
        memory["stats_prev"] = cur
    pg = getattr(up, "param_groups", None)
    if pg:
        report["lr"] = round(float(pg[0]["lr"]), 5)
    if obs.validation:
        report["val"] = [(v["epoch"], round(v["val_loss"], 4), round(v["val_acc"], 2)) for v in obs.validation]
    report["ms"] = obs.wall_ms_per_update
    return {"report": report}
