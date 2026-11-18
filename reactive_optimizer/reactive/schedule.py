"""When the agent is asked for a decision."""


def decision_updates(total_updates: int, decisions: int) -> list[int]:
    """Update indices (1-based) before which a decision is requested.

    A fifth of the decisions fall in the first 5% of the run, where the recipe matters most and
    early evidence is cheap; the rest are spread evenly up to 95% of the run.
    """
    if decisions <= 0:
        return []
    early = max(1, decisions // 5) if decisions >= 5 else 0
    late = decisions - early
    a, b = 0.05 * total_updates, 0.95 * total_updates
    points = [a * (i + 1) / early for i in range(early)]
    points += [a + (b - a) * (i + 1) / late for i in range(late)]
    out = sorted({min(total_updates, max(2, round(p))) for p in points})
    return out
