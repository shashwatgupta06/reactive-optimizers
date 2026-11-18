# NOTES

## Decision 1 (update 142)
- Loss 5.31 -> 5.26 (ln196 = 5.278); still in warmup (570). Nothing to act on yet.
- Installed v1 updater: own foreach SGD-Nesterov, same recipe as baseline (lr .05, m .9, wd 5e-4 on ndim>1,
  warmup 570, cosine to 0). Momentum carried from user_optimizer state. Roles: conv/stem/head/bn, so later
  changes are one-line edits in SETTINGS.
- Analysis v1: per-stage grad/param norms, momentum/param ratio, lrs, val.
- Prediction: identical loss trajectory to baseline; ms/update ~105. Loss should drop below ~5.0 by ~update 600.

## Decision 2 (update 285)
- v1 installed fine, 105.1 ms. Loss 5.27 -> 5.24 (slow; Cars from scratch is known to plateau early).
- Stem m/p ~0.4-0.5 (tiny stem norm 2.6, growing), fc m/p ~0.5; conv blocks ~0.01. Watch stem for instability
  once lr reaches 0.05 at 570.
- Kept everything. Prediction: loss < 5.15 by update 570, < 5.0 by ~update 900. If still >5.2 at 700,
  consider whether lr too low/high (check grad norm trends).

## Decision 3 (update 428)
- Loss window means 5.25 -> 5.18-5.21 (std rising 0.07->0.10 = some batches learnt). Grad norm 2.33 -> 1.9.
  Conv norms shrinking slowly from wd (layer3 135->131). Stem m/p 0.33->0.25, stable. Kept.
- Prediction unchanged: <5.15 by 570, <5.0 by ~900. First val at update 570 (epoch 5): expect ~1-3% acc.

## Decision 4 (update 570, end warmup)
- Loss ~5.10 (last windows 5.08-5.15): prediction <5.15 met. Grad norm 1.55 -> 1.23. No instability at
  lr 0.048. Conv norms keep shrinking ~1%/50 upd (layer3 129.9->127.0) — effective lr rising via wd.
- Kept. Prediction: <5.0 by ~900; epoch-5 val ~1-3%, epoch-10 val ~4-8%.

## Decision 5 (update 712)
- Val ep5: 2.54% / loss 5.105 (pred 1-3% met). Train loss ~4.95-5.08, last windows 4.95/4.98.
  Grad norm ~1.16 flat, stable at lr 0.05. Kept.
- Prediction: train < 4.9 by ~1000; val ep10 (update 1140) 4-8%.

## Decision 6 (update 855)
- Train loss ~4.92 (windows 4.90-4.96), on track for <4.9 by 1000. Grad norm ~1.1-1.2. Kept.
- BN gain/bias norms essentially unchanged since init (layer3.b 64.00) — BN params move very little at
  lr .05 with grad ~.04. Possible later lever: higher lr for BN/bias. Not yet; no evidence it limits.

## Decision 7 (update 998)
- Train loss ~4.93 (windows 4.85-4.98); progress slowed (pred <4.9 by 1000 borderline miss). Grad norm 1.15-1.3.
  Kept; waiting for ep10 val at 1140 before touching anything.

## Decision 8 (update 1140)
- Train loss ~4.90 (windows 4.81-4.95), slow. Grad norm 1.18-1.49. ep10 val arrives right after; next
  scheduled decision only at 1781, so analysis v2 wakes at ~1250 if ep10 val < 3.5% or train loss > 4.85.
- Updater kept. Prediction: ep10 val 4-8%; train loss ~4.8 by 1300.

## Decision 9 (update 1781)
- Val: ep10 3.55% (5.002), ep15 4.95% (4.963). Pred ep10 4-8% missed low; wake didn't fire (just above
  thresholds: 3.55 vs 3.5, loss 4.847 vs 4.85). Train loss 4.85 -> 4.66 over 640 upd, accelerating.
- Conv norms shrinking steadily (layer3 108 -> 94.2; total 200 -> 183): effective lr on BN-scaled convs rising.
  Grad norm 1.1-1.5, no instability. No overfit sign (train LS loss > val-ish).
- Kept. Slow start is typical for from-scratch Cars; learning is picking up. Prediction: ep20 val 6-9%,
  ep25 ~9-13%; train loss < 4.5 by 2400.

## Decision 10 (update 2422) -> v2 updater: wd 1e-3 on all ndim>1
- Val ep20 4.95% / 4.948 (ep15 4.95% / 4.963): flat. Train loss 4.66 -> 4.37 (pred <4.5 met).
  Val-loss drop per 5 epochs: -0.10, -0.04, -0.015 while train drop -0.2,-0.2,-0.3 => generalisation
  lagging, likely memorising 37 img/class. Pred ep20 6-9% missed.
- Change: wd 5e-4 -> 1e-3 (conv, stem, head). Doubles weight-norm shrink rate -> higher effective lr on
  BN convs (more noise) plus stronger regularisation of fc.
- Prediction: weight norms fall faster (layer3 ~80 -> ~70 by 3064); train loss decline slows a bit
  (~4.25 by 3064); val ep25 (2850) >= 6%, ep30 >= 8%. If val still ~5% at ep30 with train loss falling,
  consider stronger wd (2e-3) or higher lr. If train loss rises / stalls badly, revert toward 7e-4.

## Decision 11 (update 3064) — v2 (wd 1e-3) results
- Val ep25 9.64% / 4.501 (ep20 4.95 / 4.948): big jump, pred >=6% met. Val-loss drop 0.45 vs train drop 0.38.
- Train loss 4.37 -> 3.99 (faster than predicted 4.25 — wd did not slow it). layer3 80.4 -> 59.6 (pred ~70;
  shrinking faster than expected). m/p rising (layer1 .045 -> .076, layer3 .017 -> .040). Grad norm 1.6 -> 2.5.
  Caveat: confounded with a natural "takeoff" phase.
- Kept v2. Prediction: ep30 val >= 13%, ep35 >= 17%; train loss ~3.6 by 3700. Watch: if val loss stops
  tracking train loss again, consider wd 2e-3; if grad norm/loss spikes as norms shrink, hold.

## Decision 12 (update 3705)
- Val ep30 6.22% / 5.215 (!) vs ep25 9.64 / 4.501. Pred >=13% badly missed. Train loss 3.99 -> 3.29
  (faster than pred 3.6). Big train/val divergence.
- Two hypotheses: (a) memorisation/overfit; (b) noisy iterate: conv norms have shrunk ~3x since init
  (layer3 135 -> 45, still falling ~1/50 upd), so effective lr on BN convs ~9x higher; relative step/update
  ~0.5%; val on instantaneous weights + lagging BN running stats is noisy. One val point can't separate them.
- Kept updater; analysis v3 wakes at ~4000 if ep35 val < 12%. At that decision: if val still bad ->
  likely act. Candidate actions: (a) stronger regularisation (wd 2e-3 on fc/head only? or all);
  (b) cap effective lr: lower wd back / lower lr on convs to stop norm collapse.
  Discriminator: if ep35 val loss >5 while train keeps dropping fast -> overfit (a). If val bounces
  around 8-12% -> noise (b), mostly fixed by cosine decay later.

## Decision 13 (update 4346)
- Val ep35 21.83% / 3.292 (ep30 6.22 / 5.215, ep25 9.64 / 4.501). => ep30 dip was noise of the high-lr
  iterate (hypothesis b), not overfit. Wake didn't fire (>12%). Train loss 3.29 -> 2.74.
- Norm shrink slowing (layer3 44.6 -> 37.5 in 640 upd; earlier ~10/640). m/p plateauing ~0.09-0.1.
  Grad norm ~3, stable.
- Kept v2. Single val points at high lr are +-5pt noisy: don't react to one point.
- Prediction: ep40/45 val in 20-30% band (noisy); train loss ~2.4 by 4988.

## Decision 14 (update 4988)
- Val ep40 24.75% / 3.195 (pred 20-30% met). Train loss 2.74 -> 2.26 (pred 2.4; faster).
- Gap train(LS) 2.3 vs val(CE) 3.2 widening, but val still improving. Norms: layer3 37.5 -> 32.6, fc.w flat ~22.
  m/p ~0.11-0.12 flat, grad norm ~3. Kept v2.
- Open question for later: wd 2e-3 for more regularisation if val stalls while train keeps falling over
  two val points (ep45, ep50). Prediction: ep45/50 val 27-35%, train loss ~1.9 by 5629.

## Decision 15 (update 5629)
- Val ep45 33.50% / 2.861 (pred 27-35% met). Train 2.26 -> 1.97 (pred 1.9 ~met). Val still gaining ~+9pt/5ep
  at ep40->45. Norms near equilibrium (layer3 32.6 -> 29.9; total param norm ~120 flat). Kept v2.
- Prediction: ep50/55 val 35-45%; train ~1.75 by 6270.

## Decision 16 (update 6270)
- Val ep50 43.78% / 2.176 (pred 35-45% met; +10pt). Train 1.97 -> 1.82 (pred 1.75; slowing).
  Val loss (2.18) now approaching train (LS) loss — no overfit sign. Norms flat. Kept v2.
- Prediction: ep55/60 val 45-55%; train ~1.7 by 6911.

## Ideas to consider with evidence
- Small dataset, from scratch: overfitting is the main risk later -> maybe higher wd (1e-3) if train loss
  falls far below val loss early. Watch train-vs-val gap at epochs 5,10,20.
- Peak lr 0.05 @ bs64 is fairly high; keep unless instability.
- Late in run: cosine reaches 0 so no EMA needed (and EMA would mismatch BN running stats).
