# Method

This is an independent implementation of the reactive-optimizer idea from *Reactive Optimizers*
([Bai et al., 2026](https://arxiv.org/abs/2610.33826)): during a single training run, an LLM coding agent is
asked at fixed points to read what the run reports and may rewrite the program that turns gradients into weight
updates. It is not the authors' `opal-optimizer` package. Differences are listed at the end.

## The two arms

Every comparison is a pair of runs with the same seed:

- **Fixed recipe (baseline).** A standard optimizer and schedule set before training, not tuned for the task.
- **Reactive.** The same run with the optimizer handed to the agent. The model, its initialisation, the data,
  their order, augmentation, the loss, the batch size and the number of updates are identical, so with the same
  seed both arms see the same batches in the same order. Only the optimizer can differ.

A third mode, `keep`, wraps the optimizer in the same machinery but never asks an agent. For the torchvision
models it reproduces the baseline exactly with deterministic kernels, which checks that the wrapper itself
changes nothing. (The timm DINOv2 model is not bit-reproducible run to run, so there the check is approximate.)

## The programs the agent controls

The run keeps two Python programs in a workspace folder that the agent can edit:

- **`updater.py`** is built once (`build(named_params, previous, spec)`) and called at every update
  (`step(grads, update, ctx)`). It owns the whole optimizer: how parameters are grouped, the update rule,
  learning rates and their schedule, momentum, weight decay, clipping, and any weight averaging. It starts as a
  thin wrapper around the baseline's own optimizer and schedule, so an unchanged run trains exactly like the
  baseline. `previous` gives a new version access to the running one, so momentum and averages can carry over.
- **`analysis.py`** runs every `EVERY` updates (default 50) with read-only access to parameters, gradients,
  recent losses, validation results and the running updater. It returns a small report for the agent, and can
  ask for an extra decision ("wake").

Starting programs and an example updater for each model family are in
[`reactive_optimizer/reactive/templates/`](../reactive_optimizer/reactive/templates/). The example only shows
the interface and is not the starting optimizer.

## Decisions

- **Schedule.** 40 scheduled decisions per run. A fifth of them (8) fall in the first 5% of training, where the
  recipe matters most and evidence is cheap; the other 32 are spread evenly up to 95% of training
  ([`schedule.py`](../reactive_optimizer/reactive/schedule.py)).
- **Extra decisions.** The analysis can request up to 20 more, at least 100 updates apart. A failed updater also
  triggers one.
- **What the agent sees.** Training pauses and the agent receives a message with the training loss since the
  last decision, every validation result (accuracy and loss on 788 held-out training images), its analysis
  reports, time per update compared with the starting program, and any build errors or rollbacks. Full
  histories are kept as files in its workspace. It never sees test accuracy.
- **Memory.** One conversation spans the whole run, and the agent keeps its own `NOTES.md`. Its instructions ask
  it to compare each prediction with what happened, and to end every decision with what it observed, what it
  changed or kept, and a falsifiable prediction.
- **Instructions.** [`INSTRUCTIONS.md`](../reactive_optimizer/reactive/templates/INSTRUCTIONS.md) describes the
  task, the interfaces and the rules. Unless noted otherwise, runs include a section asking the agent to aim for
  the highest final score rather than a small improvement over the starting recipe.

## Installing changes safely

When the agent's turn ends, the wrapper compares its files with the running ones:

- A changed `updater.py` is imported and built. If that fails, the program is rejected and the old one keeps
  running; the agent sees the traceback at its next decision.
- A new updater is on **trial** for its first 20 updates. If it raises, or the training loss becomes
  non-finite, the weights return to their state at installation and the previous updater takes over with its
  state restored.
- After the trial, a failure restores the weights from a snapshot at most 100 updates old.
- Every installed or rejected version is saved in `programs/`, so each change can be diffed.

These checks protect the run, not the machine: the agent's programs run inside the training process. The agent
can only use file tools inside its workspace (no shell, no network), which is the actual containment.

## The agent

All reactive runs used **Claude Opus 5.5** at medium reasoning effort, called headless through its vendor's
command-line coding agent with only the `Read`, `Edit`, `Write`, `Glob` and `Grep` tools
([`backends.py`](../reactive_optimizer/reactive/backends.py)). Reported costs are the CLI's estimate at API list
prices. Because one conversation spans the run, each decision costs more than the last: late decisions cost
about $3 each.

Two other backends need no agent: `keep` (never changes anything) and `scripted:<folder>`, which replays canned
edits at chosen decisions. [`tests/scripted/`](../tests/scripted/) uses it to exercise a valid swap, a syntax
error, an updater that raises, one that produces a non-finite loss, and an analysis that requests a decision.

## Data and evaluation

- **Dataset.** Stanford Cars, 196 classes, from the Hugging Face mirror `tanganke/stanford_cars`. The official
  8,144 training images are split, stratified by class, into 7,356 for training and 788 for validation
  (split seed 0). The 8,041 test images are the score.
- **Preprocessing.** Images are stored at 256×256. Training uses random resized crops (scale 0.25–1) and
  horizontal flips at 224×224; evaluation uses a centre crop.
- **Metric.** Top-1 test accuracy of the weights after the last update, measured once at the end. Test
  accuracy is also logged during training for the plots, but never shown to the agent.

## Hardware

One RTX 5060 Ti (16 GB) and 14 GB of system RAM, bf16 autocast. Data is held in memory as uint8 arrays.

## Differences from the paper

| | Paper (OPAL package) | This implementation |
|---|---|---|
| Code | `opal-optimizer` | Written from the paper's description |
| Agent reasoning effort | high | medium |
| Decisions | configurable, blocking or not | 40 scheduled (+20 requested), always blocking |
| Evaluation integrity | flags weight swaps timed around evaluations | not detected (see [experiment 2](experiments/02_vit_b16.md)) |
| Scale | single node, multi-GPU (DDP) | one process, one GPU, no checkpoint or resume |
| Tasks | language models, diffusion, vision | image classification on Stanford Cars |
