"""Train an image classifier on Stanford Cars: baseline, dry run, or with the reactive optimizer.

  python -m reactive_optimizer.train --mode baseline --name base_s0 --seed 0
  python -m reactive_optimizer.train --mode keep     --name keep_s0 --seed 0   # wrapped, no agent; equals baseline
  python -m reactive_optimizer.train --mode reactive --name react_s0 --seed 0  # an agent rewrites the optimizer

The three modes share everything except who owns the optimizer step.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import torchvision

from .data import CarsData
from .reactive.backends import AGENT_MODEL, make_backend
from .reactive.wrapper import ReactiveOptimizer

HERE = os.path.dirname(os.path.abspath(__file__))


def make_lr_factor(warmup: int, total: int):
    """Linear warmup, then cosine decay to zero. `update` counts from 1."""
    def lr_factor(update: int) -> float:
        if update <= warmup:
            return update / warmup
        return 0.5 * (1 + math.cos(math.pi * (update - warmup) / max(1, total - warmup)))
    return lr_factor


class PlainStepper:
    """The baseline: the same optimizer and schedule, with nothing around them."""

    def __init__(self, optimizer, lr_factor):
        self.opt, self.lr_factor, self.update = optimizer, lr_factor, 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, loss):
        self.update += 1
        f = self.lr_factor(self.update)
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group["lr"] = base * f
        self.opt.step()

    def report_validation(self, epoch, **metrics):
        pass

    def current_lrs(self):
        return [g["lr"] for g in self.opt.param_groups]

    def summary(self):
        return {}

    def close(self):
        pass


ARCH_NAMES = {"resnet50": "ResNet-50", "resnet34": "ResNet-34", "vit_b_16": "ViT-B/16",
              "mobilenetv3_small": "MobileNetV3-Small", "dinov2_vits14": "DINOv2 ViT-S/14"}
BATCHNORM_ARCHS = ("resnet50", "resnet34", "mobilenetv3_small")


def build_model(arch, pretrained, num_classes):
    """The network and a one-line description for the agent."""
    if arch in ("resnet50", "resnet34"):
        if pretrained:
            raise SystemExit("the ResNet experiments train from scratch")
        model = getattr(torchvision.models, arch)(weights=None, num_classes=num_classes, zero_init_residual=True)
        n = sum(p.numel() for p in model.parameters())
        return model, (f"torchvision.models.{arch}(weights=None, num_classes=196, zero_init_residual=True), "
                       f"channels_last, about {n / 1e6:.1f}M parameters in {len(list(model.parameters()))} tensors.")
    if arch in ("mobilenetv3_small", "dinov2_vits14"):
        import timm
        if not pretrained:
            raise SystemExit(f"{arch} is only used fine-tuned from its pretrained weights")
        if arch == "mobilenetv3_small":
            name, kw = "mobilenetv3_small_100.lamb_in1k", {}
            desc = ("timm mobilenetv3_small_100.lamb_in1k (ImageNet-1k weights), BatchNorm, hard-swish, "
                    "squeeze-and-excitation; the 1024->{c} classifier (classifier.*) is new, timm's default init")
        else:
            name, kw = "vit_small_patch14_dinov2.lvd142m", {"img_size": 224}
            desc = ("timm vit_small_patch14_dinov2.lvd142m (DINOv2 self-supervised weights on LVD-142M), patch 14, "
                    "224x224 input (16x16 patches, position embedding resized from 518x518), 12 blocks of width 384 "
                    "with LayerScale (ls1/ls2.gamma); the 384->{c} head (head.*) is new and zero-initialised")
        model = timm.create_model(name, pretrained=True, num_classes=num_classes, **kw)
        if arch == "dinov2_vits14":
            torch.nn.init.zeros_(model.head.weight)
            torch.nn.init.zeros_(model.head.bias)
        n = sum(p.numel() for p in model.parameters())
        return model, desc.format(c=num_classes) + f"; about {n / 1e6:.2f}M parameters."
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.vit_b_16(weights=weights)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    torch.nn.init.zeros_(model.heads.head.weight)
    torch.nn.init.zeros_(model.heads.head.bias)
    n = sum(p.numel() for p in model.parameters())
    return model, (f"torchvision.models.vit_b_16 ({'ImageNet-1k weights IMAGENET1K_V1' if pretrained else 'random init'}"
                   f"), patch 16, 224x224 input, 12 encoder layers of width 768, head replaced by a zero-initialised "
                   f"768->{num_classes} linear layer (heads.head); about {n / 1e6:.1f}M parameters.")


@torch.no_grad()
def evaluate(model, data, split):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in data.eval_batches(split):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    model.train()
    return loss_sum / n, 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "keep", "reactive"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=list(ARCH_NAMES), default="resnet50")
    ap.add_argument("--pretrained", action="store_true", help="start from torchvision's ImageNet weights")
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5, help="epochs between evaluations")
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--deterministic", action="store_true", help="bit-reproducible convolutions (slower)")
    # reactive mode
    ap.add_argument("--decisions", type=int, default=40)
    ap.add_argument("--backend", default="agent", help="agent | scripted:<folder>")
    ap.add_argument("--agent-model", default=AGENT_MODEL)
    ap.add_argument("--agent-effort", default="medium")
    ap.add_argument("--max-cost-usd", type=float, default=None)
    a = ap.parse_args()

    run_dir = os.path.join(a.runs, a.name)
    if os.path.exists(os.path.join(run_dir, "log.jsonl")):
        raise SystemExit(f"{run_dir} already holds a run; pick another --name")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = not a.deterministic
    torch.backends.cudnn.deterministic = a.deterministic
    device = torch.device("cuda")

    data = CarsData(a.data, device, a.seed, a.batch_size)
    model, model_summary = build_model(a.arch, a.pretrained, data.num_classes)
    model = model.to(device).to(memory_format=torch.channels_last)

    # Weight decay on weight matrices and conv kernels only: not on norm gains, biases, ViT position embedding
    # or class token.
    no_decay_names = ("pos_embedding", "class_token", "pos_embed", "cls_token", "reg_token")
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and not n.endswith(no_decay_names)]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or n.endswith(no_decay_names)]
    groups = [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if a.optimizer == "sgd":
        optimizer = torch.optim.SGD(groups, lr=a.lr, momentum=a.momentum, nesterov=True)
        opt_text = f"SGD with Nesterov momentum {a.momentum}"
    else:
        optimizer = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
        opt_text = "AdamW (betas 0.9, 0.999, eps 1e-8, decoupled weight decay)"
    norm_text = "BatchNorm gains and biases" if a.arch in BATCHNORM_ARCHS else \
        "LayerNorm gains, biases, the position embedding and the class token"
    total = a.epochs * data.updates_per_epoch
    warmup = a.warmup_epochs * data.updates_per_epoch
    lr_factor = make_lr_factor(warmup, total)

    if a.mode == "baseline":
        stepper = PlainStepper(optimizer, lr_factor)
    else:
        options = dict(model=a.agent_model, effort=a.agent_effort) if a.backend == "agent" else {}
        backend = make_backend("keep" if a.mode == "keep" else a.backend, **options)
        starting = (f"{opt_text}, peak learning rate {a.lr} for every group, weight "
                    f"decay {a.weight_decay} on weight matrices and conv kernels and 0 on {norm_text}; linear "
                    f"warmup over the first {warmup} updates, then cosine decay to 0 at update {total}. It is a "
                    f"standard recipe that has not been tuned for this task.")
        facts = (f"Facts of the run: {len(data.train_x)} training images, {data.num_classes} classes (about "
                 f"{len(data.train_x) // data.num_classes} images per class), batch size {a.batch_size}, "
                 f"{data.updates_per_epoch} updates per epoch, {a.epochs} epochs, {total} updates in total. Images "
                 f"are 224x224 random resized crops with horizontal flips; the loss is cross-entropy with label "
                 f"smoothing {a.label_smoothing}, computed under bfloat16 autocast; parameters and gradients are "
                 f"float32. " + ("The network starts from torchvision's ImageNet-1k weights with a new, zero-initialised "
                 f"{data.num_classes}-class head. " if a.pretrained else "The training set is small for a network "
                 f"this size trained from scratch. ") + "The same images repeat every epoch.")
        evaluation = (f"The run is judged by top-1 accuracy on a held-out test set, measured once on the model's "
                      f"parameters as they are after the last update ({total})" + (", with BatchNorm running statistics as "
                      f"the training forward passes left them" if a.arch in BATCHNORM_ARCHS else "") + f". You never see test results. Every {a.eval_every} "
                      f"epochs the training script evaluates a separate validation set of {len(data.val_x)} images "
                      f"(taken out of the training data, centre crops, no label smoothing) on the current "
                      f"parameters, and you see those results. The validation set is small, so its accuracy moves "
                      f"in steps of about {100 / len(data.val_x):.2f} points and is noisy.")
        stepper = ReactiveOptimizer(
            optimizer, model, total_updates=total, templates={"resnet50": "cars", "resnet34": "cars", "vit_b_16": "cars_vit"}.get(a.arch, "cars_generic"),
            spec_extra={"user_lr_factor": lr_factor, "base_lrs": [g["lr"] for g in optimizer.param_groups]},
            spec_note=f"`user_optimizer` (the training script's `torch.optim.{type(optimizer).__name__}`; you may "
                      "keep using it or ignore it), `user_lr_factor` (the starting schedule: update index -> multiplier), `base_lrs`.",
            validation_fields="`val_loss` and `val_acc` (top-1 accuracy, %)",
            updates_per_epoch=data.updates_per_epoch, run_dir=run_dir, backend=backend,
            decisions=a.decisions, max_cost_usd=a.max_cost_usd,
            goal=(f"Train a {ARCH_NAMES[a.arch]} "
                  f"{('fine-tuned from DINOv2 self-supervised weights' if a.arch == 'dinov2_vits14' else 'fine-tuned from ImageNet weights') if a.pretrained else 'from random initialisation'} to "
                  f"classify the 196 car models of the Stanford Cars dataset; judged by top-1 test accuracy after "
                  f"the last update."),
            facts=facts, evaluation=evaluation, starting=starting,
            model_summary=model_summary,
            reference_files=[os.path.join(HERE, "train.py"), os.path.join(HERE, "data.py")])

    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    print(f"{a.mode} run '{a.name}': {total} updates, {data.updates_per_epoch} per epoch", flush=True)
    t_start, train_seconds, record = time.time(), 0.0, {}
    for epoch in range(1, a.epochs + 1):
        t0, wait0 = time.time(), stepper.summary().get("agent_wait_seconds", 0.0)
        loss_sum = torch.zeros((), device=device)
        for x, y in data.train_epoch():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y, label_smoothing=a.label_smoothing)
            loss.backward()
            stepper.step(loss)
            loss_sum += loss.detach()
        epoch_seconds = time.time() - t0 - (stepper.summary().get("agent_wait_seconds", 0.0) - wait0)
        train_seconds += epoch_seconds
        record = {"epoch": epoch, "update": epoch * data.updates_per_epoch,
                  "train_loss": loss_sum.item() / data.updates_per_epoch,
                  "lrs": stepper.current_lrs(), "epoch_seconds": round(epoch_seconds, 2)}
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            val_loss, val_acc = evaluate(model, data, "val")
            test_loss, test_acc = evaluate(model, data, "test")  # recorded only, never shown to the agent
            stepper.report_validation(epoch, val_loss=val_loss, val_acc=val_acc)
            record.update(val_loss=val_loss, val_acc=val_acc, test_loss=test_loss, test_acc=test_acc)
            print(f"epoch {epoch:3d}  train {record['train_loss']:.4f}  val {val_loss:.4f} / {val_acc:.2f}%  "
                  f"test {test_acc:.2f}%  {epoch_seconds:.1f} s", flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    result = {"mode": a.mode, "seed": a.seed, "epochs": a.epochs, "updates": total,
              "final_val_acc": record["val_acc"], "final_val_loss": record["val_loss"],
              "final_test_acc": record["test_acc"], "final_test_loss": record["test_loss"],
              "train_seconds": round(train_seconds, 1), "wall_seconds": round(time.time() - t_start, 1),
              **stepper.summary()}
    stepper.close()
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    torch.save(model.state_dict(), os.path.join(run_dir, "final.pt"))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
