You are the optimizer of one PyTorch training run that is in progress right now. At a number of
decision points during this single run you read what the run reports and may rewrite the code
that turns gradients into weight updates. Nothing carries over from other runs.

# Goal
Train a ResNet-50 from random initialisation to classify the 196 car models of the Stanford Cars dataset; judged by top-1 test accuracy after the last update.

Aim for the highest final test accuracy this run can reach within its fixed number of updates, not
a small improvement over the starting recipe. The starting recipe is an untuned default and is
probably far from the best for this task; treat it as something to beat, not something to protect.
- Prefer changes that could plausibly move final accuracy by a point or more over fine adjustments
  that could move it by a tenth.
- Make the important changes early, while there is still time to see their effect and to revert or
  correct them. A change in the last few epochs has little time to pay off.
- Keeping the programs unchanged is the right decision only when your evidence says the current
  recipe is close to the best you can reach. "Nothing is going wrong" is not by itself a reason to
  keep it.
- Stay grounded in this run's evidence: an ambitious change still needs a reason you can state and
  a prediction you can check.

Facts of the run: 7356 training images, 196 classes (about 37 images per class), batch size 64, 114 updates per epoch, 200 epochs, 22800 updates in total. Images are 224x224 random resized crops with horizontal flips; the loss is cross-entropy with label smoothing 0.1, computed under bfloat16 autocast; parameters and gradients are float32. The training set is small for a network this size trained from scratch, and the same images repeat every epoch.

The model, its initialisation, the data, their order and augmentation, the loss, the batch size
and the number of updates are fixed. The optimization is entirely yours.

# What you can change
Two programs in this folder, plus any helper `.py` files you add next to them:

- `updater.py` runs at every update. It owns the whole optimizer: which rule updates which
  tensor, every hyperparameter per group or per tensor (learning rate and its schedule,
  momentum, weight decay, betas, epsilon), preconditioning, clipping, update-size control,
  weight averaging, and feedback rules that read gradient or loss statistics.
- `analysis.py` measures whatever you want to know about the run and reports it to you.

You may treat any block, layer type or single tensor differently, and change that at any
decision. Keeping the programs as they are is a valid decision.

# How the run works
- An **update** is one forward pass, one backward pass and one call of your updater.
  A **decision** is a turn like this one; there are far fewer decisions than updates.
- Every `EVERY` updates (set in `analysis.py`, default 50), after the backward pass and before
  the update, `analyze(obs, memory)` runs. What it returns under `"report"` is stored for your
  next decision. Returning `"wake": "<reason>"` asks for an extra decision now (extra decisions
  are limited and need at least 100 updates since the previous decision).
- Training is paused while you decide. When your turn ends, the run compares your files with the
  running ones. A changed `updater.py` (or helper) is built with `previous=<running updater>` and
  takes over at the next update. A changed `analysis.py` is reloaded.
- **Trial and rollback.** A program that fails to import or build is not installed. If a newly
  installed updater raises, or the training loss becomes non-finite, during its first 20
  updates, the model weights go back to what they were when it was installed and the previous
  updater takes over again with its state restored. If that happens later, the weights go back
  to a snapshot at most 100 updates old and the previous updater takes over. The update
  counter never goes back. You see the error at your next decision. You cannot restart the run.
- The analysis only reports; it cannot change the updater. Feedback that must act at every
  update belongs in the updater, which receives the loss of each update.

# Interfaces
`updater.py` defines `build(named_params, previous=None, spec=None)` and returns an object with:
- `step(grads, update, ctx)`: change the parameters in place. `grads` maps parameter name to its
  gradient tensor (the same tensor as `param.grad`). `update` counts from 1. `ctx.loss` is this
  update's training loss as a 0-d tensor on the GPU; `ctx.update` and `ctx.total_updates` are ints.
- `state_dict()` and `load_state_dict(state)`: everything needed to restore the updater after a
  rollback, as plain tensors, numbers, strings, lists and dicts.
- optionally `param_groups`: a list of dicts with `"params"` and `"lr"`, used for logging.

`named_params` is the list of `(name, parameter)` pairs in the table at the end of this file.
`spec` is a dict: `total_updates`, `update` (the next update), `updates_per_epoch`,
`user_optimizer` (the training script's `torch.optim.SGD`; you may keep using it or ignore it),
`user_lr_factor` (the starting schedule: update index -> multiplier), `base_lrs`.
`previous` is the running updater object; read its state to carry momentum or averages over.
Copy what you take (`.clone()`, `copy.deepcopy`); never modify the previous updater's tensors or
the user optimizer's state inside `build`, because a rejected build must leave them intact.

`analysis.py` defines `analyze(obs, memory)` and may set `EVERY`. `memory` is a dict that
persists between calls. `obs` has:
- `obs.update`, `obs.total_updates`, `obs.updates_per_epoch`
- `obs.losses`: training loss of each update since the previous call (floats)
- `obs.params`, `obs.grads`: name -> tensor, read only
- `obs.updater`: your running updater object, read only
- `obs.validation`: validation results since the previous call, each
  `{"update", "epoch", "val_loss", "val_acc"}` (often empty)
- `obs.wall_ms_per_update`: recent wall time per update
`analyze` returns `None` or a dict with `"report"` (small JSON-serialisable value) and/or `"wake"`.

# Rules
- Edit only files in this folder. Do not try to reach the training script's other objects, the
  data or the evaluation from your programs.
- Do not change the dtype or device of parameters or gradients.
- Write every schedule as a function of the absolute update index and `total_updates`, never as
  an offset from "now" kept in a counter.
- Your updater's time is training time. Avoid Python loops over tensors where a batched or
  `torch._foreach_*` form exists, and do not read values from the GPU inside `step`
  (`.item()`, `float(tensor)`, `if tensor:`); keep statistics on the device and read them in `analyze`.
- Each decision shows the time per update next to the starting program's. A slower rule can be
  worth it; an accidentally slow one is not.

# Evaluation
The run is judged by top-1 accuracy on a held-out test set, measured once on the model's parameters as they are after the last update (22800), with BatchNorm running statistics as the training forward passes left them. You never see test results. Every 5 epochs the training script evaluates a separate validation set of 788 images (taken out of the training data, centre crops, no label smoothing) on the current parameters, and you see those results. The validation set is small, so its accuracy moves in steps of about 0.13 points and is noisy.

# Files
- `INSTRUCTIONS.md`: this text. Re-read it whenever you are unsure of the protocol.
- `updater.py`, `analysis.py`: your programs.
- `inbox/decision_NNN.md`: every decision message. `inbox/reports.jsonl`: all analysis reports.
  `inbox/train_loss.csv`: update, loss for every update so far. `inbox/validation.csv`.
- `programs/`: every version of your programs that was installed or rejected.
- `reference/`: read-only material: `example_updater.py` (interface example, not a
  recommendation), `starting_optimizer.txt`, and the training script and data code.
- `NOTES.md`: yours. Keep beliefs, predictions and plans here; it persists for the whole run.

# Each decision
Before changing anything, check what you predicted last time against what happened. Let each
change follow from evidence in this run and from how your earlier changes played out.
Complexity is not the objective; a better final result is. End your turn with a short rationale:
1. **Observed**: what the evidence since the last decision shows, in numbers.
2. **Changed or kept**: what you edited and why, or why you kept the programs.
3. **Prediction**: the effect you expect, by roughly which update, and what would prove you wrong.

# The run
Starting recipe: SGD with Nesterov momentum 0.9, peak learning rate 0.05 for every group, weight decay 0.0005 on conv and linear weights and 0 on BatchNorm gains and biases; linear warmup over the first 570 updates, then cosine decay to 0 at update 22800. It is a standard recipe that has not been tuned for this task.

Model: torchvision.models.resnet50(weights=None, num_classes=196, zero_init_residual=True), channels_last, about 23.9M parameters in 161 tensors.

Parameters (name, shape, starting group), in `named_params` order:
conv1.weight  (64, 3, 7, 7)  group 0
bn1.weight  (64,)  group 1
bn1.bias  (64,)  group 1
layer1.0.conv1.weight  (64, 64, 1, 1)  group 0
layer1.0.bn1.weight  (64,)  group 1
layer1.0.bn1.bias  (64,)  group 1
layer1.0.conv2.weight  (64, 64, 3, 3)  group 0
layer1.0.bn2.weight  (64,)  group 1
layer1.0.bn2.bias  (64,)  group 1
layer1.0.conv3.weight  (256, 64, 1, 1)  group 0
layer1.0.bn3.weight  (256,)  group 1
layer1.0.bn3.bias  (256,)  group 1
layer1.0.downsample.0.weight  (256, 64, 1, 1)  group 0
layer1.0.downsample.1.weight  (256,)  group 1
layer1.0.downsample.1.bias  (256,)  group 1
layer1.1.conv1.weight  (64, 256, 1, 1)  group 0
layer1.1.bn1.weight  (64,)  group 1
layer1.1.bn1.bias  (64,)  group 1
layer1.1.conv2.weight  (64, 64, 3, 3)  group 0
layer1.1.bn2.weight  (64,)  group 1
layer1.1.bn2.bias  (64,)  group 1
layer1.1.conv3.weight  (256, 64, 1, 1)  group 0
layer1.1.bn3.weight  (256,)  group 1
layer1.1.bn3.bias  (256,)  group 1
layer1.2.conv1.weight  (64, 256, 1, 1)  group 0
layer1.2.bn1.weight  (64,)  group 1
layer1.2.bn1.bias  (64,)  group 1
layer1.2.conv2.weight  (64, 64, 3, 3)  group 0
layer1.2.bn2.weight  (64,)  group 1
layer1.2.bn2.bias  (64,)  group 1
layer1.2.conv3.weight  (256, 64, 1, 1)  group 0
layer1.2.bn3.weight  (256,)  group 1
layer1.2.bn3.bias  (256,)  group 1
layer2.0.conv1.weight  (128, 256, 1, 1)  group 0
layer2.0.bn1.weight  (128,)  group 1
layer2.0.bn1.bias  (128,)  group 1
layer2.0.conv2.weight  (128, 128, 3, 3)  group 0
layer2.0.bn2.weight  (128,)  group 1
layer2.0.bn2.bias  (128,)  group 1
layer2.0.conv3.weight  (512, 128, 1, 1)  group 0
layer2.0.bn3.weight  (512,)  group 1
layer2.0.bn3.bias  (512,)  group 1
layer2.0.downsample.0.weight  (512, 256, 1, 1)  group 0
layer2.0.downsample.1.weight  (512,)  group 1
layer2.0.downsample.1.bias  (512,)  group 1
layer2.1.conv1.weight  (128, 512, 1, 1)  group 0
layer2.1.bn1.weight  (128,)  group 1
layer2.1.bn1.bias  (128,)  group 1
layer2.1.conv2.weight  (128, 128, 3, 3)  group 0
layer2.1.bn2.weight  (128,)  group 1
layer2.1.bn2.bias  (128,)  group 1
layer2.1.conv3.weight  (512, 128, 1, 1)  group 0
layer2.1.bn3.weight  (512,)  group 1
layer2.1.bn3.bias  (512,)  group 1
layer2.2.conv1.weight  (128, 512, 1, 1)  group 0
layer2.2.bn1.weight  (128,)  group 1
layer2.2.bn1.bias  (128,)  group 1
layer2.2.conv2.weight  (128, 128, 3, 3)  group 0
layer2.2.bn2.weight  (128,)  group 1
layer2.2.bn2.bias  (128,)  group 1
layer2.2.conv3.weight  (512, 128, 1, 1)  group 0
layer2.2.bn3.weight  (512,)  group 1
layer2.2.bn3.bias  (512,)  group 1
layer2.3.conv1.weight  (128, 512, 1, 1)  group 0
layer2.3.bn1.weight  (128,)  group 1
layer2.3.bn1.bias  (128,)  group 1
layer2.3.conv2.weight  (128, 128, 3, 3)  group 0
layer2.3.bn2.weight  (128,)  group 1
layer2.3.bn2.bias  (128,)  group 1
layer2.3.conv3.weight  (512, 128, 1, 1)  group 0
layer2.3.bn3.weight  (512,)  group 1
layer2.3.bn3.bias  (512,)  group 1
layer3.0.conv1.weight  (256, 512, 1, 1)  group 0
layer3.0.bn1.weight  (256,)  group 1
layer3.0.bn1.bias  (256,)  group 1
layer3.0.conv2.weight  (256, 256, 3, 3)  group 0
layer3.0.bn2.weight  (256,)  group 1
layer3.0.bn2.bias  (256,)  group 1
layer3.0.conv3.weight  (1024, 256, 1, 1)  group 0
layer3.0.bn3.weight  (1024,)  group 1
layer3.0.bn3.bias  (1024,)  group 1
layer3.0.downsample.0.weight  (1024, 512, 1, 1)  group 0
layer3.0.downsample.1.weight  (1024,)  group 1
layer3.0.downsample.1.bias  (1024,)  group 1
layer3.1.conv1.weight  (256, 1024, 1, 1)  group 0
layer3.1.bn1.weight  (256,)  group 1
layer3.1.bn1.bias  (256,)  group 1
layer3.1.conv2.weight  (256, 256, 3, 3)  group 0
layer3.1.bn2.weight  (256,)  group 1
layer3.1.bn2.bias  (256,)  group 1
layer3.1.conv3.weight  (1024, 256, 1, 1)  group 0
layer3.1.bn3.weight  (1024,)  group 1
layer3.1.bn3.bias  (1024,)  group 1
layer3.2.conv1.weight  (256, 1024, 1, 1)  group 0
layer3.2.bn1.weight  (256,)  group 1
layer3.2.bn1.bias  (256,)  group 1
layer3.2.conv2.weight  (256, 256, 3, 3)  group 0
layer3.2.bn2.weight  (256,)  group 1
layer3.2.bn2.bias  (256,)  group 1
layer3.2.conv3.weight  (1024, 256, 1, 1)  group 0
layer3.2.bn3.weight  (1024,)  group 1
layer3.2.bn3.bias  (1024,)  group 1
layer3.3.conv1.weight  (256, 1024, 1, 1)  group 0
layer3.3.bn1.weight  (256,)  group 1
layer3.3.bn1.bias  (256,)  group 1
layer3.3.conv2.weight  (256, 256, 3, 3)  group 0
layer3.3.bn2.weight  (256,)  group 1
layer3.3.bn2.bias  (256,)  group 1
layer3.3.conv3.weight  (1024, 256, 1, 1)  group 0
layer3.3.bn3.weight  (1024,)  group 1
layer3.3.bn3.bias  (1024,)  group 1
layer3.4.conv1.weight  (256, 1024, 1, 1)  group 0
layer3.4.bn1.weight  (256,)  group 1
layer3.4.bn1.bias  (256,)  group 1
layer3.4.conv2.weight  (256, 256, 3, 3)  group 0
layer3.4.bn2.weight  (256,)  group 1
layer3.4.bn2.bias  (256,)  group 1
layer3.4.conv3.weight  (1024, 256, 1, 1)  group 0
layer3.4.bn3.weight  (1024,)  group 1
layer3.4.bn3.bias  (1024,)  group 1
layer3.5.conv1.weight  (256, 1024, 1, 1)  group 0
layer3.5.bn1.weight  (256,)  group 1
layer3.5.bn1.bias  (256,)  group 1
layer3.5.conv2.weight  (256, 256, 3, 3)  group 0
layer3.5.bn2.weight  (256,)  group 1
layer3.5.bn2.bias  (256,)  group 1
layer3.5.conv3.weight  (1024, 256, 1, 1)  group 0
layer3.5.bn3.weight  (1024,)  group 1
layer3.5.bn3.bias  (1024,)  group 1
layer4.0.conv1.weight  (512, 1024, 1, 1)  group 0
layer4.0.bn1.weight  (512,)  group 1
layer4.0.bn1.bias  (512,)  group 1
layer4.0.conv2.weight  (512, 512, 3, 3)  group 0
layer4.0.bn2.weight  (512,)  group 1
layer4.0.bn2.bias  (512,)  group 1
layer4.0.conv3.weight  (2048, 512, 1, 1)  group 0
layer4.0.bn3.weight  (2048,)  group 1
layer4.0.bn3.bias  (2048,)  group 1
layer4.0.downsample.0.weight  (2048, 1024, 1, 1)  group 0
layer4.0.downsample.1.weight  (2048,)  group 1
layer4.0.downsample.1.bias  (2048,)  group 1
layer4.1.conv1.weight  (512, 2048, 1, 1)  group 0
layer4.1.bn1.weight  (512,)  group 1
layer4.1.bn1.bias  (512,)  group 1
layer4.1.conv2.weight  (512, 512, 3, 3)  group 0
layer4.1.bn2.weight  (512,)  group 1
layer4.1.bn2.bias  (512,)  group 1
layer4.1.conv3.weight  (2048, 512, 1, 1)  group 0
layer4.1.bn3.weight  (2048,)  group 1
layer4.1.bn3.bias  (2048,)  group 1
layer4.2.conv1.weight  (512, 2048, 1, 1)  group 0
layer4.2.bn1.weight  (512,)  group 1
layer4.2.bn1.bias  (512,)  group 1
layer4.2.conv2.weight  (512, 512, 3, 3)  group 0
layer4.2.bn2.weight  (512,)  group 1
layer4.2.bn2.bias  (512,)  group 1
layer4.2.conv3.weight  (2048, 512, 1, 1)  group 0
layer4.2.bn3.weight  (2048,)  group 1
layer4.2.bn3.bias  (2048,)  group 1
fc.weight  (196, 2048)  group 0
fc.bias  (196,)  group 1
