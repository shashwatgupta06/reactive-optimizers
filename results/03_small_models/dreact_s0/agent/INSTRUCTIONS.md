You are the optimizer of one PyTorch training run that is in progress right now. At a number of
decision points during this single run you read what the run reports and may rewrite the code
that turns gradients into weight updates. Nothing carries over from other runs.

# Goal
Train a DINOv2 ViT-S/14 fine-tuned from DINOv2 self-supervised weights to classify the 196 car models of the Stanford Cars dataset; judged by top-1 test accuracy after the last update.

Aim for the highest final score on the evaluation above that this run can reach within its fixed number of updates, not
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

Facts of the run: 7356 training images, 196 classes (about 37 images per class), batch size 64, 114 updates per epoch, 100 epochs, 11400 updates in total. Images are 224x224 random resized crops with horizontal flips; the loss is cross-entropy with label smoothing 0.1, computed under bfloat16 autocast; parameters and gradients are float32. The network starts from torchvision's ImageNet-1k weights with a new, zero-initialised 196-class head. The same images repeat every epoch.

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
`spec` is a dict: `total_updates`, `update` (the next update), `updates_per_epoch`, and
`user_optimizer` (the training script's `torch.optim.AdamW`; you may keep using it or ignore it), `user_lr_factor` (the starting schedule: update index -> multiplier), `base_lrs`.
`previous` is the running updater object; read its state to carry momentum or averages over.
Copy what you take (`.clone()`, `copy.deepcopy`); never modify the previous updater's tensors or
the user optimizer's state inside `build`, because a rejected build must leave them intact.

`analysis.py` defines `analyze(obs, memory)` and may set `EVERY`. `memory` is a dict that
persists between calls. `obs` has:
- `obs.update`, `obs.total_updates`, `obs.updates_per_epoch`
- `obs.losses`: training loss of each update since the previous call (floats)
- `obs.params`, `obs.grads`: name -> tensor, read only
- `obs.updater`: your running updater object, read only
- `obs.validation`: validation results since the previous call, each a dict with `update`,
  `epoch` and `val_loss` and `val_acc` (top-1 accuracy, %) (often empty)
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
The run is judged by top-1 accuracy on a held-out test set, measured once on the model's parameters as they are after the last update (11400). You never see test results. Every 2 epochs the training script evaluates a separate validation set of 788 images (taken out of the training data, centre crops, no label smoothing) on the current parameters, and you see those results. The validation set is small, so its accuracy moves in steps of about 0.13 points and is noisy.

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
Starting recipe: AdamW (betas 0.9, 0.999, eps 1e-8, decoupled weight decay), peak learning rate 0.0001 for every group, weight decay 0.05 on weight matrices and conv kernels and 0 on LayerNorm gains, biases, the position embedding and the class token; linear warmup over the first 570 updates, then cosine decay to 0 at update 11400. It is a standard recipe that has not been tuned for this task.

Model: timm vit_small_patch14_dinov2.lvd142m (DINOv2 self-supervised weights on LVD-142M), patch 14, 224x224 input (16x16 patches, position embedding resized from 518x518), 12 blocks of width 384 with LayerScale (ls1/ls2.gamma); the 384->196 head (head.*) is new and zero-initialised; about 21.70M parameters.

Parameters (name, shape, starting group), in `named_params` order:
cls_token  (1, 1, 384)  group 1
pos_embed  (1, 257, 384)  group 1
patch_embed.proj.weight  (384, 3, 14, 14)  group 0
patch_embed.proj.bias  (384,)  group 1
blocks.0.norm1.weight  (384,)  group 1
blocks.0.norm1.bias  (384,)  group 1
blocks.0.attn.qkv.weight  (1152, 384)  group 0
blocks.0.attn.qkv.bias  (1152,)  group 1
blocks.0.attn.proj.weight  (384, 384)  group 0
blocks.0.attn.proj.bias  (384,)  group 1
blocks.0.ls1.gamma  (384,)  group 1
blocks.0.norm2.weight  (384,)  group 1
blocks.0.norm2.bias  (384,)  group 1
blocks.0.mlp.fc1.weight  (1536, 384)  group 0
blocks.0.mlp.fc1.bias  (1536,)  group 1
blocks.0.mlp.fc2.weight  (384, 1536)  group 0
blocks.0.mlp.fc2.bias  (384,)  group 1
blocks.0.ls2.gamma  (384,)  group 1
blocks.1.norm1.weight  (384,)  group 1
blocks.1.norm1.bias  (384,)  group 1
blocks.1.attn.qkv.weight  (1152, 384)  group 0
blocks.1.attn.qkv.bias  (1152,)  group 1
blocks.1.attn.proj.weight  (384, 384)  group 0
blocks.1.attn.proj.bias  (384,)  group 1
blocks.1.ls1.gamma  (384,)  group 1
blocks.1.norm2.weight  (384,)  group 1
blocks.1.norm2.bias  (384,)  group 1
blocks.1.mlp.fc1.weight  (1536, 384)  group 0
blocks.1.mlp.fc1.bias  (1536,)  group 1
blocks.1.mlp.fc2.weight  (384, 1536)  group 0
blocks.1.mlp.fc2.bias  (384,)  group 1
blocks.1.ls2.gamma  (384,)  group 1
blocks.2.norm1.weight  (384,)  group 1
blocks.2.norm1.bias  (384,)  group 1
blocks.2.attn.qkv.weight  (1152, 384)  group 0
blocks.2.attn.qkv.bias  (1152,)  group 1
blocks.2.attn.proj.weight  (384, 384)  group 0
blocks.2.attn.proj.bias  (384,)  group 1
blocks.2.ls1.gamma  (384,)  group 1
blocks.2.norm2.weight  (384,)  group 1
blocks.2.norm2.bias  (384,)  group 1
blocks.2.mlp.fc1.weight  (1536, 384)  group 0
blocks.2.mlp.fc1.bias  (1536,)  group 1
blocks.2.mlp.fc2.weight  (384, 1536)  group 0
blocks.2.mlp.fc2.bias  (384,)  group 1
blocks.2.ls2.gamma  (384,)  group 1
blocks.3.norm1.weight  (384,)  group 1
blocks.3.norm1.bias  (384,)  group 1
blocks.3.attn.qkv.weight  (1152, 384)  group 0
blocks.3.attn.qkv.bias  (1152,)  group 1
blocks.3.attn.proj.weight  (384, 384)  group 0
blocks.3.attn.proj.bias  (384,)  group 1
blocks.3.ls1.gamma  (384,)  group 1
blocks.3.norm2.weight  (384,)  group 1
blocks.3.norm2.bias  (384,)  group 1
blocks.3.mlp.fc1.weight  (1536, 384)  group 0
blocks.3.mlp.fc1.bias  (1536,)  group 1
blocks.3.mlp.fc2.weight  (384, 1536)  group 0
blocks.3.mlp.fc2.bias  (384,)  group 1
blocks.3.ls2.gamma  (384,)  group 1
blocks.4.norm1.weight  (384,)  group 1
blocks.4.norm1.bias  (384,)  group 1
blocks.4.attn.qkv.weight  (1152, 384)  group 0
blocks.4.attn.qkv.bias  (1152,)  group 1
blocks.4.attn.proj.weight  (384, 384)  group 0
blocks.4.attn.proj.bias  (384,)  group 1
blocks.4.ls1.gamma  (384,)  group 1
blocks.4.norm2.weight  (384,)  group 1
blocks.4.norm2.bias  (384,)  group 1
blocks.4.mlp.fc1.weight  (1536, 384)  group 0
blocks.4.mlp.fc1.bias  (1536,)  group 1
blocks.4.mlp.fc2.weight  (384, 1536)  group 0
blocks.4.mlp.fc2.bias  (384,)  group 1
blocks.4.ls2.gamma  (384,)  group 1
blocks.5.norm1.weight  (384,)  group 1
blocks.5.norm1.bias  (384,)  group 1
blocks.5.attn.qkv.weight  (1152, 384)  group 0
blocks.5.attn.qkv.bias  (1152,)  group 1
blocks.5.attn.proj.weight  (384, 384)  group 0
blocks.5.attn.proj.bias  (384,)  group 1
blocks.5.ls1.gamma  (384,)  group 1
blocks.5.norm2.weight  (384,)  group 1
blocks.5.norm2.bias  (384,)  group 1
blocks.5.mlp.fc1.weight  (1536, 384)  group 0
blocks.5.mlp.fc1.bias  (1536,)  group 1
blocks.5.mlp.fc2.weight  (384, 1536)  group 0
blocks.5.mlp.fc2.bias  (384,)  group 1
blocks.5.ls2.gamma  (384,)  group 1
blocks.6.norm1.weight  (384,)  group 1
blocks.6.norm1.bias  (384,)  group 1
blocks.6.attn.qkv.weight  (1152, 384)  group 0
blocks.6.attn.qkv.bias  (1152,)  group 1
blocks.6.attn.proj.weight  (384, 384)  group 0
blocks.6.attn.proj.bias  (384,)  group 1
blocks.6.ls1.gamma  (384,)  group 1
blocks.6.norm2.weight  (384,)  group 1
blocks.6.norm2.bias  (384,)  group 1
blocks.6.mlp.fc1.weight  (1536, 384)  group 0
blocks.6.mlp.fc1.bias  (1536,)  group 1
blocks.6.mlp.fc2.weight  (384, 1536)  group 0
blocks.6.mlp.fc2.bias  (384,)  group 1
blocks.6.ls2.gamma  (384,)  group 1
blocks.7.norm1.weight  (384,)  group 1
blocks.7.norm1.bias  (384,)  group 1
blocks.7.attn.qkv.weight  (1152, 384)  group 0
blocks.7.attn.qkv.bias  (1152,)  group 1
blocks.7.attn.proj.weight  (384, 384)  group 0
blocks.7.attn.proj.bias  (384,)  group 1
blocks.7.ls1.gamma  (384,)  group 1
blocks.7.norm2.weight  (384,)  group 1
blocks.7.norm2.bias  (384,)  group 1
blocks.7.mlp.fc1.weight  (1536, 384)  group 0
blocks.7.mlp.fc1.bias  (1536,)  group 1
blocks.7.mlp.fc2.weight  (384, 1536)  group 0
blocks.7.mlp.fc2.bias  (384,)  group 1
blocks.7.ls2.gamma  (384,)  group 1
blocks.8.norm1.weight  (384,)  group 1
blocks.8.norm1.bias  (384,)  group 1
blocks.8.attn.qkv.weight  (1152, 384)  group 0
blocks.8.attn.qkv.bias  (1152,)  group 1
blocks.8.attn.proj.weight  (384, 384)  group 0
blocks.8.attn.proj.bias  (384,)  group 1
blocks.8.ls1.gamma  (384,)  group 1
blocks.8.norm2.weight  (384,)  group 1
blocks.8.norm2.bias  (384,)  group 1
blocks.8.mlp.fc1.weight  (1536, 384)  group 0
blocks.8.mlp.fc1.bias  (1536,)  group 1
blocks.8.mlp.fc2.weight  (384, 1536)  group 0
blocks.8.mlp.fc2.bias  (384,)  group 1
blocks.8.ls2.gamma  (384,)  group 1
blocks.9.norm1.weight  (384,)  group 1
blocks.9.norm1.bias  (384,)  group 1
blocks.9.attn.qkv.weight  (1152, 384)  group 0
blocks.9.attn.qkv.bias  (1152,)  group 1
blocks.9.attn.proj.weight  (384, 384)  group 0
blocks.9.attn.proj.bias  (384,)  group 1
blocks.9.ls1.gamma  (384,)  group 1
blocks.9.norm2.weight  (384,)  group 1
blocks.9.norm2.bias  (384,)  group 1
blocks.9.mlp.fc1.weight  (1536, 384)  group 0
blocks.9.mlp.fc1.bias  (1536,)  group 1
blocks.9.mlp.fc2.weight  (384, 1536)  group 0
blocks.9.mlp.fc2.bias  (384,)  group 1
blocks.9.ls2.gamma  (384,)  group 1
blocks.10.norm1.weight  (384,)  group 1
blocks.10.norm1.bias  (384,)  group 1
blocks.10.attn.qkv.weight  (1152, 384)  group 0
blocks.10.attn.qkv.bias  (1152,)  group 1
blocks.10.attn.proj.weight  (384, 384)  group 0
blocks.10.attn.proj.bias  (384,)  group 1
blocks.10.ls1.gamma  (384,)  group 1
blocks.10.norm2.weight  (384,)  group 1
blocks.10.norm2.bias  (384,)  group 1
blocks.10.mlp.fc1.weight  (1536, 384)  group 0
blocks.10.mlp.fc1.bias  (1536,)  group 1
blocks.10.mlp.fc2.weight  (384, 1536)  group 0
blocks.10.mlp.fc2.bias  (384,)  group 1
blocks.10.ls2.gamma  (384,)  group 1
blocks.11.norm1.weight  (384,)  group 1
blocks.11.norm1.bias  (384,)  group 1
blocks.11.attn.qkv.weight  (1152, 384)  group 0
blocks.11.attn.qkv.bias  (1152,)  group 1
blocks.11.attn.proj.weight  (384, 384)  group 0
blocks.11.attn.proj.bias  (384,)  group 1
blocks.11.ls1.gamma  (384,)  group 1
blocks.11.norm2.weight  (384,)  group 1
blocks.11.norm2.bias  (384,)  group 1
blocks.11.mlp.fc1.weight  (1536, 384)  group 0
blocks.11.mlp.fc1.bias  (1536,)  group 1
blocks.11.mlp.fc2.weight  (384, 1536)  group 0
blocks.11.mlp.fc2.bias  (384,)  group 1
blocks.11.ls2.gamma  (384,)  group 1
norm.weight  (384,)  group 1
norm.bias  (384,)  group 1
head.weight  (196, 384)  group 0
head.bias  (196,)  group 1
