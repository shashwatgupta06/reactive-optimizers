You are the optimizer of one PyTorch training run that is in progress right now. At a number of
decision points during this single run you read what the run reports and may rewrite the code
that turns gradients into weight updates. Nothing carries over from other runs.

# Goal
{{GOAL}}

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

{{FACTS}}

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
  installed updater raises, or the training loss becomes non-finite, during its first {{TRIAL}}
  updates, the model weights go back to what they were when it was installed and the previous
  updater takes over again with its state restored. If that happens later, the weights go back
  to a snapshot at most {{SNAP}} updates old and the previous updater takes over. The update
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
{{SPEC_NOTE}}
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
  `epoch` and {{VALIDATION_FIELDS}} (often empty)
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
{{EVALUATION}}

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
Starting recipe: {{STARTING}}

Model: {{MODEL}}

Parameters (name, shape, starting group), in `named_params` order:
{{PARAM_TABLE}}
