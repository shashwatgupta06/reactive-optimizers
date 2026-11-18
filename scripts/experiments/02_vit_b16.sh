#!/usr/bin/env bash
# Experiment 2: ViT-B/16 (ImageNet-1k weights) fine-tuned on Stanford Cars, 100 epochs. About 1 hour per run.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
VIT="--arch vit_b_16 --pretrained --optimizer adamw --lr 1e-4 --weight-decay 0.05 --warmup-epochs 5 --epochs 100 --eval-every 2"

$PY -m reactive_optimizer.train --mode baseline --name vbase_s0 --seed 0 $VIT
$PY -m reactive_optimizer.train --mode reactive --name vreact_s0 --seed 0 $VIT
$PY scripts/compare.py vbase_ vreact_
