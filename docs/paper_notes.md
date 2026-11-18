# Paper notes — Reactive Optimizers (OPAL)

Sources read: arXiv 2610.33826 **v2** (4 Nov 2026), the blog's interactive sections, and the official repo
(README, `opal/prompts/neutral.md`, `examples/sit_b2/README.md`) at v0.3.10.

## Idea
OPAL = **OP**timizer with an **A**gent in the **L**oop. A pretrained LLM coding agent (a frontier model at
`high` reasoning effort) is called at "decision points" during **one** training run. It can rewrite two Python programs:

- `updater.py`: the optimizer program, run at every step (update rules per parameter group, learning rates and
  schedules, momentum/β₂, preconditioning, clipping, spike guards, precision).
- `analysis.py`: computes whatever statistics the agent wants (per-group gradient norm, update/weight ratio,
  cosine between consecutive gradients, ...).

It works zero-shot: no meta-training and no memory carried over between runs. The paper gives no update
equations. The method is the agent plus this code interface.

The paper's claim: when no recipe has been tuned for the setting and you can afford one run but not a sweep,
spending LLM tokens on a reactive optimizer beats adopting a fixed recipe.

## Mechanism (from the repo and the agent prompt)
- **Start state:** `updater.py` starts as a call to the script's own optimizer and scheduler. Until the agent
  edits something, the run is bit-for-bit the plain run.
- **Loop:** `analysis.py` runs on rank 0 every `every` updates (default 50), after backward and before the update.
  It returns a small JSON report (the last 400 are kept) and may return `{"wake": reason}` to request a decision now.
  The analysis only reports; any per-step feedback must live in the updater (which can read `ctx.loss` on device).
- **Decisions:** at most 50 by default, and few enough that waiting stays under 4x the run's own time. A fifth of
  them fall in the first 5% of the run.
- **Memory within the run:** one conversation for the whole run. Each decision must end with Observed / Changed or
  kept / Prediction / Rules, and the agent checks its last prediction before changing anything. It may keep `NOTES.md`.
- **What the agent sees:** the goal sentence, parameter names and shapes, training losses, its own reports, read-only
  copies of the training script and model code, and validation values only if `validation=N` is set. Test metrics
  logged with `opt.log` are not shown.
- **Fixed:** model, initialisation, data and order, loss, batch size, number of updates, the training loop, parameter
  and gradient dtype. This is a rule in the prompt, **not a sandbox**: the agent's code runs inside the training process.
- **Swaps and failures:** a new updater is built with `previous=<old updater>` so state can be carried over. It runs
  on trial for 20 updates; if it raises, the run goes back to the previous program. A failure on the first update is
  undone exactly; a later one is not. The agent cannot roll back training, only fix its programs at the next decision.
- **Schedules** must be written as functions of the absolute update index, so a resumed run behaves the same.
- **Optional skip:** with `can_skip = True` the updater can return True from `step` to undo that update
  (about 10% more time per update).

Modes and cost:
- Blocking mode: training waits for each decision. Overhead is 1.30x to 4.39x wall-clock.
- Non-blocking mode (`blocking=False`): training continues and the edit is swapped in later (median 65 updates later
  on Goldfish). Overhead is 1.01x to 1.14x, which is the cost of the agent's own code (Muon, float32 state, analysis).
- A decision takes about 40 s of agent time, so OPAL suits long runs (hours) better than short ones. On the small
  one-GPU example scripts, 50 decisions took 27 to 31 min against 1 to 3 min without the agent.
- Token cost: $6.42 for about 100 decisions on LightningDiT-XL/1, next to about 74 H100-hours (about $510);
  $11.28 for 94 decisions on SiT-B/2.

## Reported results (single runs, mostly)
| Task | Metric | Baseline | OPAL |
|---|---|---|---|
| Goldfish 125M GPT, Amharic | FLORES bits ↓ | 83.48 (official), 80.89 (best sweep) | 81.11 (from official) / 80.73 (from best sweep) |
| Goldfish 125M GPT, Swahili (repo) | FLORES bits ↓ | 79.37 | 77.25 |
| LightningDiT-XL/1 ImageNet-256 | FID-50K ↓, with / without guidance | 2.18 / 5.15 | 1.83 / 4.33 |
| LightningDiT-B/1, 160 epochs (repo) | FID-50K w/ guidance ↓ | 3.22 | 2.31 |
| RAEv2-B ImageNet-256 | FID-50K ↓ | 3.39 (already a Muon variant) | 3.19 |
| SiT-B/2 400K (repo, not in the paper) | FID-50K ↓ | 35.7 AdamW, 30.4 untuned Muon | 23.6 |
| ResNet-18 CIFAR-10, 10% noisy labels | test acc ↑ / test loss ↓ | 77.5% / 0.832 (validation-selected EMA) | 78.6% / 0.685 |
| modded-nanogpt #36 | val loss ↓ | 3.2781 | 3.2749 (another run: 3.2878, worse) |
| 34 toy 2D landscapes, 300 steps | mean log10(f − f*) ↓ | −2.2 (best of 12 common), −7.4 (best of 382,080 swept) | −11.6 |

Also in Figure 1: modded-nanogpt Track 3 moved to Chinese, Thai and Kyrgyz data and to ¼ batch, against five
leaderboard recipes (#2 AdamW, #21 Shampoo, #36 tuned Muon, #30 and #46 elaborate Muon). OPAL starts from #36 and
matches or beats the best baseline. Only the Chinese and Thai bars are means of three runs.

The SiT-B/2 numbers come from "the earlier training system OPAL grew from"; that system and its prompt are not released.

## What the agents actually did
- **Optimizer swaps.** In most runs the first edit moved attention and MLP matrices from AdamW to Muon and kept AdamW
  for the rest. Later moves were finer (LightningDiT: adaLN weights split into six 1152×1152 chunks and moved to Muon).
- **Learning-rate schedules.** Separate multi-stage schedules per parameter group, including mid-run increases.
  - *Deduced from clues* (SiT): negative cosine between consecutive gradients of the timestep embedder, so its rate
    went to 0.3x and then 0.15x; rates were raised for the Muon matrices where the cosine stayed positive.
  - *A-B-A test* (RAEv2-B): 2x for updates 1,850–2,500 between stretches at 1.5x, then 2x kept. Later tests of 2.5x
    and 2.75x were rejected. Short dips to 0.3x were used to measure step-size noise in the loss.
  - *Re-planned cooldown*: RAEv2-B started cooldown at 45% of the run instead of 75%; SiT moved it from 320K to 280K.
- **Momentum.** Tried Muon momentum 0.98 (RAEv2-B) and 0.975 (Goldfish); both were reverted after the loss rose.
- **Precision.** On modded-nanogpt, token-embedding entries are about 15 in size where adjacent bfloat16 values are
  0.0625 apart, so it kept that matrix's Adam moments in float32 and wrote weights back with compensated rounding.
- **Loss spikes.** A SiT agent traced 17 of 18 gradient spikes to the timestep embedder and added per-tensor clipping
  against each tensor's own recent norm.
- **Data corruption** (CIFAR-10, four stretches of random images and labels, agent not told): damping, then clipping,
  then a gate that holds back batches whose loss is far above recent clean ones. Accuracy change per stretch:
  −50.0, −66.3, −12.1, +0.7. Clipped baselines and Stable-SPAM still lost 28 to 76 points in each.
- **Overfitting** (ResNet-18, 120 epochs, no augmentation, 10% wrong labels, agent sees validation but not test):
  progressively higher learning rates plus a weight average of epochs 96–117 chosen on validation. It predicts the
  wrong training label on 3.7% of the 4,500 mislabeled images, against 100% (AdamW final), 19.4% (early stopping)
  and 14.2% (validation-selected EMA). The network normalises its logits (centered LogitNorm) "to prevent
  loss-hacking behavior".
- **Self-corrections.** Many edits are tagged "fix or revert", including bugs in its own analysis (an off-by-one in
  epoch length) and three analysis edits on LightningDiT that never took effect because the answer arrived too late.

## Critical reading
Limitations the authors state: gains are small on short runs with already-tuned recipes, results vary from run to
run, there is wall-clock overhead, most results come from a single run, and sharded training and optimizer
compilation are not supported yet.

My own reading (not claims from the paper):
- **Reactivity is not isolated from recipe knowledge.** Much of the gain looks like "switch to Muon and fix the
  learning rate". On SiT-B/2 an untuned Muon baseline covers 35.7 → 30.4 of the 35.7 → 23.6 gap. On Goldfish a
  four-value learning-rate sweep (80.89) beat OPAL started from the official recipe (81.11).
- **Missing control:** the same agent writing one recipe at step 0 and never editing again. Figure 3's caption
  itself says the curves show whole-run outcomes and do not isolate the effect of each edit.
- **Wall-clock is not equalised.** SiT-B/2: 11.2 h for OPAL (1.05 h waiting) against 7.1 h for AdamW and 13.0 h for Muon.
- **Favourable comparisons.** The toy-landscape agent had moves static optimizers lack (reset momentum, return to
  best point, random jump) and was restricted to heavy-ball because unrestricted OPAL just wrote a second-order
  method. The overfitting accuracy gain over the EMA baseline is 1.1 points; the loss gap is larger.
- **Loss-hacking is a real risk.** The overfitting setup needed LogitNorm to stop it, and the package ships an
  `eval_integrity` check for programs that move parameters around an evaluation.

Experiments this suggests: the step-0-only control, repeated seeds for variance, equal wall-clock comparison, and a
cheaper model as the agent.

## Official package usage
```python
opt = opal.wrap(opt, model, scheduler=sched, total_steps=steps, goal="<what + how judged>",
                backend="keep")        # dry run, bit-identical to plain; use max_cost_usd=... for the agent
opt.step(loss=loss)
opt.log(test_acc=...)                  # recorded, not shown to the agent
opt.close()
```
Options: `decisions`, `every` (50), `blocking` (True), `max_cost_usd` (no cap by default), `validation=N` with
`opt.report_validation(...)`, `prompt` (`"neutral"`, `"aggressive"`, or a template path),
`backend_options` (`model`, `effort`, `timeout`). Experimental Codex backend.

CLI: `opal doctor [--live]`, `opal estimate`, `opal show|watch|report|compare opal_runs/<run>`.
Small examples in the repo: `mnist_mlp.py`, `cifar10_resnet.py`, `gpt_small.py`, `diffusion_mnist.py`.
Requirements: Python 3.10+, PyTorch 2.5+, the agent CLI logged in on the rank-0 machine (or the mailbox broker).
Supported: CPU, one GPU, single-node DDP. Not supported: multi-node, FSDP, DeepSpeed.
