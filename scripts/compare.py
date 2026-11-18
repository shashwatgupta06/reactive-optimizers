"""Summarise finished runs: mean and spread of final test accuracy per mode.

  python scripts/compare.py            # every run under runs/
  python scripts/compare.py base_ react_   # only runs whose name starts with one of these
"""
import glob
import json
import os
import statistics
import sys


def main() -> None:
    prefixes = tuple(sys.argv[1:])
    by_mode = {}
    for path in sorted(glob.glob("runs/*/result.json")):
        name = os.path.basename(os.path.dirname(path))
        if prefixes and not name.startswith(prefixes):
            continue
        r = json.load(open(path))
        by_mode.setdefault(r["mode"], []).append((name, r))

    for mode, runs in by_mode.items():
        print(f"\n{mode}")
        for name, r in runs:
            extra = ""
            if "decisions" in r and r["decisions"]:
                extra = (f"  decisions {r['decisions']}, swaps {r['swaps']}, rollbacks {r['rollbacks']}, "
                         f"agent ${r['agent_cost_usd']:.2f}, waited {r['agent_wait_seconds'] / 60:.1f} min")
            print(f"  {name:<24} seed {r['seed']}  test {r['final_test_acc']:.2f}%  val {r['final_val_acc']:.2f}%  "
                  f"train time {r['train_seconds'] / 60:.1f} min{extra}")
        accs = [r["final_test_acc"] for _, r in runs]
        spread = f" +/- {statistics.stdev(accs):.2f} (std, n={len(accs)})" if len(accs) > 1 else " (one run)"
        print(f"  mean test accuracy: {statistics.mean(accs):.2f}%{spread}")

    if "baseline" in by_mode and "reactive" in by_mode:
        b = {r["seed"]: r["final_test_acc"] for _, r in by_mode["baseline"]}
        a = {r["seed"]: r["final_test_acc"] for _, r in by_mode["reactive"]}
        shared = sorted(set(a) & set(b))
        if shared:
            diffs = [a[s] - b[s] for s in shared]
            print("\nreactive minus baseline, same seed: " + ", ".join(f"seed {s}: {d:+.2f}" for s, d in zip(shared, diffs))
                  + f"; mean {statistics.mean(diffs):+.2f} points")


if __name__ == "__main__":
    main()
