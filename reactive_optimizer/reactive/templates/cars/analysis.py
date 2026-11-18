"""Starting analysis program: a few statistics of the loss and the gradients."""
import math

EVERY = 50  # updates between calls


def analyze(obs, memory):
    losses = obs.losses
    mean = sum(losses) / max(1, len(losses))
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / max(1, len(losses)))

    grad_sq, param_sq = 0.0, 0.0
    for name, g in obs.grads.items():
        if g is None:
            continue
        grad_sq += float(g.float().pow(2).sum())
        param_sq += float(obs.params[name].float().pow(2).sum())

    report = {
        "loss_mean": round(mean, 4),
        "loss_std": round(std, 4),
        "grad_norm": round(math.sqrt(grad_sq), 4),
        "param_norm": round(math.sqrt(param_sq), 3),
    }
    return {"report": report}
