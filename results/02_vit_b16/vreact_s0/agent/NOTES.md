# NOTES

## Decision 1 (update 71)
- Baseline: loss 5.28 -> 5.235 by u70 (ln196=5.278). Zero head at lr ~1e-5 (warmup) is the bottleneck.
- Installed v1: AdamW, LLRD 0.85 (top layer 1e-4, patch embed ~1.2e-5), head lr 1e-3 with warmup to u200,
  backbone warmup to u570 then cosine to 0 at 11400. wd 0.05. Adam state carried from user opt.
- Analysis: per-depth grad norm, relative drift from init per depth, val, lrs.
- Predictions: train loss < 4.5 by u250, < 3 by ~u600; first val (u228) noticeably > baseline-like ~5-15%.
  Val acc should reach ~80%+ by epoch 20. Watch for: lower layers barely drifting (LLRD too strong for
  fine-grained domain shift) -> relax to 0.9; train loss -> label-smoothing floor (~1.0) early with
  val plateau -> overfitting, consider more wd / lower lr.
## Decision 2 (u142): loss 5.05 -> 3.88 over u71-141 (beat prediction). Drift baseline captured at u100
  (not u0). Kept everything; waiting for first val (u228).
## Decision 3 (u214): loss 3.81 -> 3.10. Backbone grad norms rising (enc layers ~1.7 at u200), drift
  monotone in depth (top 0.021, bottom 0.002 since u100). Kept. Still no val.
## Decision 4 (u285): first val u228 = 42.51% (val_loss 2.33). Train loss 2.99 -> 2.27. Kept.
  Next check: val u456 should be ~65-75%.
## Decision 5 (u356): train loss 2.22 -> 1.76. Val every 228 updates (next u456). Kept.
## Decision 6 (u428): train loss flattened ~1.65 (u370-427) while backbone lr still ramping. Kept; wait val u456.
## Decision 7 (u499): val u456 = 78.05% (loss 0.95), above 65-75 prediction. Train 1.38. Kept.
  Plan: watch for train loss hitting ~1.0 smoothing floor; then val trend decides (maybe raise wd / EMA tail).
## Decision 8 (u570): peak lr reached; train ~1.37-1.49 flat over u499-569. Kept updater; added wake on
  val drop >4 pts below best. Next decision u891 (vals at u684, u912).
## Decision 9 (u891): val u684 = 84.14% (loss 0.75). Train 1.28 -> 1.16. Smoothing floor ~0.85 (H of target).
  Kept. Predict val u912 ~85-87, u1140 ~87.
## Decision 10 (u1211): val 84.64 (u912), 86.42 (u1140); val_loss flat ~0.745 for 3 readings. Train 1.13 -> 1.06.
  Kept. Watch: if val_loss starts rising while train -> 0.9 floor, raise wd (0.05 -> 0.1-0.2) on backbone.
## Decision 11 (u1532): val u1368 86.29 (loss 0.779, up from 0.741). Train 1.015. Borderline overfit signal,
  single noisy reading. Kept; if u1596 & u1824 val_loss keep rising (>0.80) -> act (wd up / stronger LLRD).
  Plan: implement tail weight-averaging (EMA, copied into params at u11400) around epoch ~70.
## Decision 12 (u1852): val 87.18 (u1596, loss 0.736), 85.53 (u1824, loss 0.775). Train ~1.00 flat.
  Interpretation: high-lr noise plateau (lr still 95% of peak), not progressive overfit (val_loss oscillates,
  no trend). Gains expected as cosine decays. Kept. Next: if plateau persists to epoch ~30 with val <=87,
  consider pulling annealing earlier; EMA tail still planned.
## Decision 13 (u2173): val u2052 87.69 (best, loss 0.786). Train ~0.99 flat. Grad norms fell to ~2.1. Kept.
## Decision 14 (u2494): val u2280 87.44 (loss 0.770). Train 0.97 slowly falling. Plateau as predicted. Kept.
## Decision 15 (u2814): val 88.45 (u2508), 88.83 (u2736) - still improving at ~88% peak lr. Train ~0.965.
  No reason to pull annealing earlier. Kept.
## Decision 16 (u3135): val u2964 87.44 (loss 0.803). Band since ep14: 85.5-88.8, mean ~87.6. Train 0.95. Kept.
## Decision 17 (u3456): val 87.94 (u3192, loss 0.721), 87.69 (u3420, loss 0.795). Plateau ~87.9 since ep18.
  Installed v2 = v1 + weight EMA (start u3460, decay 0.998 ~500-update horizon) and EMA PROBES at u4104
  (ep36) and u4560 (ep40): EMA swapped into params right before validation, raw restored next step.
  Validation at those epochs measures EMA, not raw (reports tag ema_probe). ema_final=False for now.
  Purpose: measure how much of the plateau is high-lr noise. Prediction: EMA probe val ~89-90 (+1-2 over raw
  ~87.9). If EMA gain >= +1: enable ema_final, consider continuous EMA through end, and possibly
  lower peak lr / earlier annealing. If ~0: noise isn't the limiter; EMA tail only as minor tweak.
## Decision 18 (u3776): v2 installed fine (285 ms, +1.3 ms). EMA rel_dist 1.19% after 290 updates.
  val u3648 87.94 (loss 0.826). Kept; waiting for probe u4104 (seen at decision ~u4417).
## Decision 19 (u4097): val u3876 88.96 (best raw, loss 0.827). Train ~0.93. EMA rel_dist ~1.3% (stable).
  Kept; probe at u4104 is 7 updates away.
## Decision 20 (u4418): PROBE RESULT: EMA (ep36) 89.97% / loss 0.687 vs raw 88.96 (ep34, loss 0.827) and
  88.71 (ep38, loss 0.765). EMA gain ~+1.0-1.3 acc, -0.1 val loss -> high-lr noise limits raw iterate.
  v3: ema_final=True; probes every 4 epochs (ep40,44,...,96), raw vals at ep42,46,...,98,100.
  Plan: track EMA-probe trend; if EMA keeps rising, high-lr exploration is useful -> keep cosine.
  If EMA flat while raw flat, consider faster annealing. Near the end compare EMA(ep96) vs raw(ep98) to decide
  whether to keep ema_final (lagging EMA could lose to the annealed raw when lr->0).
## Decision 21 (u4738): EMA probe ep40 89.47 / loss 0.677 (ep36: 89.97 / 0.687). EMA loss still improving,
  acc within noise. Train ~0.92. Kept v3.
## Decision 22 (u5059): raw ep42 90.10 (loss 0.747), EMA ep44 90.23 (loss 0.679). Both rising as lr decays
  (lr 60% peak). Train 0.91. Kept v3.
## Decision 23 (u5379): raw ep46 89.09 (loss 0.780) - noise dip vs 90.10. Raw val_loss 0.75-0.83 while EMA
  ~0.68 consistently. Train 0.91. Kept v3; next EMA probe ep48 (u5472).
## Decision 24 (u5700): EMA ep48 90.48 / 0.685. EMA series acc 89.97, 89.47, 90.23, 90.48 (slow rise),
  loss ~0.68 flat. Train 0.90. Halfway, lr 51% peak. Kept v3.
## Decision 25 (u6021): raw ep50 88.83 (0.812), EMA ep52 90.48 (0.696). EMA ~90.3-90.5 plateau; lr 47% peak.
  Kept v3. Expect real gains in the low-lr phase (ep60-90).
## Decision 26 (u6341): raw ep54 88.83 (0.786). Raw since ep42: 90.10, 89.09, 88.83, 88.83 (flat/slightly down);
  EMA flat 90.2-90.5. Train 0.897. Kept. If EMA ep56/60 < 90 -> consider wd increase.
## Decision 27 (u6662): EMA ep56 89.72 / 0.701; raw ep58 88.83 / 0.801. EMA val_loss creeping up
  (0.677 ep40 -> 0.701 ep56) while train loss 0.90 -> 0.89: mild memorization, acc ~flat (EMA 89.7-90.5).
  Kept: wd toward 0 is a weak lever at these lrs; changes now are guesses. Low-lr phase ahead (lr 38% peak).
  EMA still +1 over raw consistently -> keep ema_final.
## Decision 28 (u6982): EMA ep60 90.36 / 0.714 (loss still creeping up, acc fine). Train 0.885. Kept.
## Decision 29 (u7303): raw ep62 89.97 (0.775), EMA ep64 90.61 (0.723, best probe acc). Train 0.887. Kept.
## Decision 30 (u7624): raw ep66 89.47 (0.786). Raw band 88.8-90.1, EMA band 89.7-90.6. Train 0.884. Kept.
## Decision 31 (u7944): EMA ep68 90.86 / 0.728 (best). Train 0.881. Kept.
## Decision 32 (u8265): raw ep70 89.97 (0.780), EMA ep72 90.23 (0.734). Gap narrowing (~0.3-0.9). Kept.
## Decision 33 (u8586): raw ep74 89.72 (0.774). Train 0.878. EMA rel_dist 0.42%. Kept.
## Decision 34 (u8906): EMA ep76 89.97 (0.742) = raw ep78 89.97 (0.759). Converging as lr->0 (lr 12% peak,
  EMA rel_dist 0.34%). EMA still lower loss. Kept ema_final.
## Decision 35 (u9227): EMA ep80 90.23 (0.745). Train 0.874. Kept.
## Decision 36 (u9548): raw ep82 90.36 (0.764) vs EMA ep80 90.23 (0.745): raw +1 image (1st time, noise);
  EMA still lower loss. Kept.
## Decision 37 (u9868): EMA ep84 90.10 (0.748), raw ep86 90.61 (0.754). Raw beat neighbouring EMA twice ->
  v4: ema_decay 0.998 -> 0.995 (horizon ~200 updates) to cut lag; ema_final kept. Effect tiny either way
  (EMA rel_dist 0.17%).
## Decision 38 (u10189): EMA ep88 89.97 (0.752). EMA rel_dist 0.07% -> EMA ~= raw; val differences are
  epoch-to-epoch noise. Kept v4.
## Decision 39 (u10509): raw ep90 89.85 (0.763), EMA ep92 89.97 (0.758). Settled ~90. Kept v4.
## Decision 40 (u10830, last): raw ep94 90.10 (0.759). EMA rel_dist 0.018% (~identical to raw). Kept v4
  with ema_final=True; one more probe at u10944 (ep96), final ep100 = EMA copy. Expect final val ~90 +/- 0.6.
- Ideas later:
  - EMA probe: val happens at u=228k after the step; could swap EMA into params at step 228k and restore
    at step 228k+1 to measure EMA val (one batch's grads computed at EMA point; small cost). compare drift by depth; consider raising backbone lr to 2e-4 if fitting slow.
