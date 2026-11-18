# Experiments

All experiments train an image classifier on Stanford Cars twice from the same starting point and on identical
batches: once with a fixed recipe and once with the reactive optimizer. The setup shared by all of them is in
[the method](../method.md).

| # | Experiment | Models | Headline |
|---|---|---|---|
| 1 | [ResNet-50 from scratch](01_resnet50.md) | ResNet-50 | 81.29% ± 0.12 (3 seeds) → **87.90%** (+6.6) |
| 2 | [ViT-B/16 fine-tuned](02_vit_b16.md) | ViT-B/16 | 89.24% → 89.76% (+0.5) |
| 3 | [Three small models](03_small_models.md) | ResNet-34, MobileNetV3-Small, DINOv2 ViT-S/14 | ResNet-34 +3.5; DINOv2 +1.6; MobileNetV3 −2.8 |

## Abandoned: a more capable agent

After experiments 1 and 2, the agent was given more abilities: a recipe review before the first update, short
trials of candidate optimizers, rewinds to checkpoints, evaluation of candidate weights, and control of
augmentation and label smoothing. A ResNet-50 run with this version was stopped at epoch 35, at 5.5% test
accuracy against the baseline's 21.5%. Before training it had switched on strong CutMix, and a short trial then
led it to lower the learning rate and weight decay: the opposite of what won experiment 1. The changes were
rolled back, and all reported runs use the original agent.

The lesson: short-horizon trials favour whatever learns fastest right now, which is not what wins a long run.
The same failure appears in experiment 3, where reacting to short validation plateaus cost MobileNetV3 2.8
points.
