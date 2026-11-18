#!/usr/bin/env bash
# Experiment 1: ResNet-50 from scratch on Stanford Cars, 200 epochs. About 40 minutes per run on an RTX 5060 Ti.
# react_s0 and react_s1 were run with --max-cost-usd 15 and an earlier version of INSTRUCTIONS.md without the
# "aim for the highest final score" section; react_s2 used the current instructions and no cap.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}

for seed in 0 1 2; do
  $PY -m reactive_optimizer.train --mode baseline --name base_s$seed --seed $seed
done
$PY -m reactive_optimizer.train --mode reactive --name react_s0 --seed 0 --max-cost-usd 15
$PY -m reactive_optimizer.train --mode reactive --name react_s1 --seed 1 --max-cost-usd 15
$PY -m reactive_optimizer.train --mode reactive --name react_s2 --seed 2
$PY scripts/compare.py base_ react_
