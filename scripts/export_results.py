"""Export the finished runs in runs/ into the small, versioned results/ folder.

  python scripts/export_results.py

For every run: metrics.csv (one row per epoch), result.json (final numbers), config.json (arguments).
For reactive runs also agent/: every installed program version, the agent's notes, its instructions, a CSV
of decisions and decisions.md (the agent's own explanation at each decision). Model weights are not exported.
results/summary.csv has one row per run.
"""
import csv
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reactive_optimizer.reactive.backends import AGENT_ENV_PREFIXES, AGENT_EXECUTABLE  # noqa: E402

EXPERIMENTS = {
    "01_resnet50": {
        "model": "ResNet-50 (from scratch)",
        "runs": ["base_s0", "base_s1", "base_s2", "react_s0", "react_s1", "react_s2"],
    },
    "02_vit_b16": {
        "model": "ViT-B/16 (ImageNet-1k, fine-tuned)",
        "runs": ["vbase_s0", "vreact_s0"],
    },
    "03_small_models": {
        "model": None,  # per run below
        "runs": ["r34base_s0", "r34react_s0", "mbbase_s0", "mbreact_s0", "dbase_s0", "dreact_s0"],
    },
}
SMALL_MODELS = {"r34": "ResNet-34 (from scratch)", "mb": "MobileNetV3-Small (ImageNet-1k, fine-tuned)",
                "d": "DINOv2 ViT-S/14 (fine-tuned)"}
SETUPS = {  # reactive runs that differ from the standard setup (no cost cap, "aim for the best" instructions)
    "react_s0": "agent, $15 cap, original instructions",
    "react_s1": "agent, $15 cap, original instructions",
}
METRIC_FIELDS = ["epoch", "update", "train_loss", "lr", "epoch_seconds",
                 "val_loss", "val_acc", "test_loss", "test_acc"]
VENDOR = re.compile("|".join([AGENT_EXECUTABLE] + [p.split("_")[0] for p in AGENT_ENV_PREFIXES]), re.I)
CONFIG_DROP = {"data", "runs", "agent_model"}  # local paths; the agent model is stated once in docs/method.md
SRC, DST = "runs", "results"


def model_name(exp: str, run: str) -> str:
    if EXPERIMENTS[exp]["model"]:
        return EXPERIMENTS[exp]["model"]
    return SMALL_MODELS[run.split("base")[0].split("react")[0]]


def clean_text(text: str) -> str:
    """Drop absolute paths into this repository; refuse to export text that names the agent's vendor."""
    text = text.replace(ROOT + os.sep, "")
    if VENDOR.search(text):
        raise ValueError("export contains the agent vendor's name; clean the source or extend clean_text")
    return text


def copy_text(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src) as f, open(dst, "w") as g:
        g.write(clean_text(f.read()))


def export_metrics(src: str, dst: str) -> None:
    with open(src) as f, open(dst, "w", newline="") as g:
        w = csv.DictWriter(g, METRIC_FIELDS)
        w.writeheader()
        for line in f:
            row = json.loads(line)
            if "epoch" not in row:
                continue
            row["lr"] = row["lrs"][0] if row.get("lrs") else ""
            w.writerow({k: round(row[k], 6) if isinstance(row.get(k), float) else row.get(k, "")
                        for k in METRIC_FIELDS})


def export_config(src: str, dst: str) -> None:
    cfg = {k: v for k, v in json.load(open(src)).items() if k not in CONFIG_DROP}
    if cfg.get("backend") == AGENT_EXECUTABLE:  # runs were made before the backend was renamed
        cfg["backend"] = "agent"
    json.dump(cfg, open(dst, "w"), indent=1)


def export_agent(run_dir: str, dst: str) -> None:
    ws = os.path.join(run_dir, "workspace")
    out = os.path.join(dst, "agent")
    for name in sorted(os.listdir(os.path.join(ws, "programs"))):
        copy_text(os.path.join(ws, "programs", name), os.path.join(out, "programs", name))
    for name in ("NOTES.md", "INSTRUCTIONS.md"):
        if os.path.exists(os.path.join(ws, name)):
            copy_text(os.path.join(ws, name), os.path.join(out, name))

    decisions = [json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl"))]
    decisions = [d for d in decisions if d.get("event") == "decision"]
    with open(os.path.join(out, "decisions.csv"), "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["decision", "update", "reason", "changed", "installed", "cost_usd", "seconds", "error"])
        for d in decisions:
            w.writerow([d["decision"], d["update"], d.get("reason", ""), " ".join(d.get("changed", [])),
                        d.get("installed", False), round(d.get("cost_usd", 0), 4), round(d.get("seconds", 0), 1),
                        clean_text(d.get("error", ""))[:200]])

    parts = [f"# Agent decisions: `{os.path.basename(run_dir)}`\n",
             "The agent's reply at each decision point, as written. `changed` lists the programs it edited.\n"]
    for d in decisions:
        answer = os.path.join(ws, "inbox", f"decision_{d['decision']:03d}_answer.md")
        body = clean_text(open(answer).read().strip()) if os.path.exists(answer) else "_(no answer recorded)_"
        changed = ", ".join(d.get("changed", [])) or "nothing"
        parts.append(f"\n## Decision {d['decision']} (update {d['update']}, {d.get('reason', '')}; "
                     f"changed: {changed})\n\n{body}\n")
    with open(os.path.join(out, "decisions.md"), "w") as g:
        g.write("".join(parts))


def main() -> None:
    if os.path.exists(DST):
        shutil.rmtree(DST)
    summary = []
    for exp, spec in EXPERIMENTS.items():
        for run in spec["runs"]:
            src = os.path.join(SRC, run)
            if not os.path.exists(os.path.join(src, "result.json")):
                print(f"skip {run}: not finished")
                continue
            dst = os.path.join(DST, exp, run)
            os.makedirs(dst)
            export_metrics(os.path.join(src, "log.jsonl"), os.path.join(dst, "metrics.csv"))
            export_config(os.path.join(src, "config.json"), os.path.join(dst, "config.json"))
            result = json.load(open(os.path.join(src, "result.json")))
            json.dump(result, open(os.path.join(dst, "result.json"), "w"), indent=1)
            reactive = result["mode"] == "reactive"
            if reactive:
                export_agent(src, dst)
            summary.append({
                "experiment": exp, "model": model_name(exp, run), "run": run, "seed": result["seed"],
                "setup": SETUPS.get(run, "agent, no cap" if reactive else "fixed recipe"),
                "epochs": result["epochs"],
                "test_acc": round(result["final_test_acc"], 2), "val_acc": round(result["final_val_acc"], 2),
                "decisions": result.get("decisions", ""), "program_versions": result.get("swaps", ""),
                "agent_cost_usd": round(result["agent_cost_usd"], 2) if reactive else "",
                "train_min": round(result["train_seconds"] / 60, 1),
                "wall_min": round(result["wall_seconds"] / 60, 1),
            })
            print(f"exported {exp}/{run}")
    with open(os.path.join(DST, "summary.csv"), "w", newline="") as g:
        w = csv.DictWriter(g, list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
