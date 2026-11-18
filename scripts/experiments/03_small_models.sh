#!/usr/bin/env bash
# Experiment 3: ResNet-34 from scratch (200 epochs), MobileNetV3-Small and DINOv2 ViT-S/14 fine-tuned (100 epochs).
# Runs take about 22, 3 and 24 minutes each without the agent.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
R34="--arch resnet34"
MB="--arch mobilenetv3_small --pretrained --optimizer adamw --lr 1e-3 --weight-decay 0.05 --warmup-epochs 5 --epochs 100 --eval-every 2"
DINO="--arch dinov2_vits14 --pretrained --optimizer adamw --lr 1e-4 --weight-decay 0.05 --warmup-epochs 5 --epochs 100 --eval-every 2"

for model in r34 mb d; do
  case $model in r34) args=$R34 ;; mb) args=$MB ;; d) args=$DINO ;; esac
  $PY -m reactive_optimizer.train --mode baseline --name ${model}base_s0 --seed 0 $args
  $PY -m reactive_optimizer.train --mode reactive --name ${model}react_s0 --seed 0 $args
done
$PY scripts/compare.py r34 mb d
