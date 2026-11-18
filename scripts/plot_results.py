"""Plot test accuracy over training, baseline against reactive, for each model in results/.

  python scripts/plot_results.py        # writes docs/figures/*.png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAIRS = [  # (title, experiment, baseline run, reactive run, figure file)
    ("ResNet-50, from scratch", "01_resnet50", "base_s2", "react_s2", "resnet50"),
    ("ResNet-34, from scratch", "03_small_models", "r34base_s0", "r34react_s0", "resnet34"),
    ("ViT-B/16, fine-tuned", "02_vit_b16", "vbase_s0", "vreact_s0", "vit_b16"),
    ("DINOv2 ViT-S/14, fine-tuned", "03_small_models", "dbase_s0", "dreact_s0", "dinov2_vits14"),
    ("MobileNetV3-Small, fine-tuned", "03_small_models", "mbbase_s0", "mbreact_s0", "mobilenetv3_small"),
]
BASE, REACT = "#7a7a7a", "#1f6fd1"
OUT = "docs/figures"


def curve(exp: str, run: str):
    path = os.path.join("results", exp, run, "metrics.csv")
    if not os.path.exists(path):
        return None
    xs, ys = [], []
    for row in csv.DictReader(open(path)):
        if row["test_acc"]:
            xs.append(int(row["epoch"]))
            ys.append(float(row["test_acc"]))
    return xs, ys


def decisions(exp: str, run: str):
    """Epochs at which the agent installed a new optimizer program."""
    path = os.path.join("results", exp, run, "agent", "decisions.csv")
    metrics = os.path.join("results", exp, run, "metrics.csv")
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(metrics)))
    per_epoch = int(rows[0]["update"]) / int(rows[0]["epoch"])
    return [int(d["update"]) / per_epoch for d in csv.DictReader(open(path))
            if d["installed"] == "True" and "updater.py" in d["changed"]]


def draw(ax, title, exp, base, react):
    b, r = curve(exp, base), curve(exp, react)
    if not b or not r:
        return False
    ax.plot(*b, color=BASE, lw=1.6, label=f"fixed recipe  {b[1][-1]:.2f}%")
    ax.plot(*r, color=REACT, lw=1.6, label=f"reactive  {r[1][-1]:.2f}%")
    for e in decisions(exp, react):
        ax.axvline(e, color=REACT, alpha=0.25, lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return True


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    done = []
    for title, exp, base, react, name in PAIRS:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        if draw(ax, title, exp, base, react):
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=150)
            done.append((title, exp, base, react))
        plt.close(fig)

    if done:  # one overview figure for the README
        cols = min(3, len(done))
        rows = (len(done) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
        for ax, args in zip(axes.flat, done):
            draw(ax, *args)
        for ax in list(axes.flat)[len(done):]:
            ax.axis("off")
        fig.suptitle("Test accuracy during training on Stanford Cars (vertical lines: agent changed the optimizer)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "overview.png"), dpi=150)
        plt.close(fig)
    print(f"wrote {len(done)} figures to {OUT}/")


if __name__ == "__main__":
    main()
