EVERY = 10


def analyze(obs, memory):
    memory["calls"] = memory.get("calls", 0) + 1
    out = {"report": {"calls": memory["calls"], "last_loss": obs.losses[-1]}}
    if memory["calls"] == 12:
        out["wake"] = "test wake"
    return out
