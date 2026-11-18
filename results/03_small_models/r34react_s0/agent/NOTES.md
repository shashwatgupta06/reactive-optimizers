# NOTES

## Decision 1 (update 142)
- Baseline: loss 5.31 -> 5.24 (ln196=5.278), still in warmup, grad norm ~1.4. Nothing to judge yet.
- Belief: small data (37 img/class) + from scratch => regularisation (effective lr*wd) is the main lever.
  Baseline lr*wd = 2.5e-5. New: lr 0.08, wd 8e-4 (lr*wd 6.4e-5, ~2.6x), no wd on BN/bias.
  Warmup extended to 1140 (absolute index), cosine to 0 at 22800. Global clip 5 (safety only).
- Own foreach Nesterov SGD (v1), momentum carried from torch SGD.
- Predictions: no instability (clip_frac ~0 after warmup; loss finite). Train loss falls slower
  than baseline would mid-run (more regularisation); val acc at epoch 20 should be >= ~10%.
  Watch: if val acc stalls < 5% by epoch 15 or loss oscillates/spikes, lower lr to 0.05.
## Decision 2 (update 285): v1 installed fine, 59.0 ms (vs 58.6). Loss 5.25 -> 5.16, clip_frac 0,
  gnorm ~1.3, lr 0.0198. Still warmup; kept. First val at epoch 5 (~update 570).
## Decision 3 (update 428): loss plateau ~5.15 (285-427), loss_std rising 0.04->0.085, gnorm 1.26,
  clip 0, lr 0.03. Cars-from-scratch is known to be slow to start; kept. If loss still >5.0 at
  update ~1000 (lr near peak) consider it a slow-start problem (maybe higher lr on BN/head).
## Decision 4 (update 570): loss ~5.11 (slow drift down), gnorm 1.18, clip 0, lr 0.04. Kept.
## Decision 5 (update 712): val ep5 = 2.54% (5x chance, beat my 1-2% guess). loss ~5.06, gnorm 1.09,
  clip 0, lr 0.049. Weight norm shrinking ~1.7/50 upd (wd at work). Kept.
## Decision 6 (update 855): loss ~4.98-5.0 (crossing 5.0), clip 0, lr 0.06, conv w norm falling
  ~1.8/50 upd (raises effective lr of BN-scaled convs). Kept. Watch for equilibrium of w norm.
## Decision 7 (update 998): loss ~4.96 (912-997), clip 0, lr 0.067, w norm 118. Kept. Ep10 val next.
## Decision 8 (update 1140, peak lr reached): loss ~4.93, clip 0, gnorm 1.0, w norm 112 (layer3 78->56).
  Ep10 val not yet seen. Kept updater; analysis now wakes on loss blow-up (+0.4 over best 50-mean) or
  clip_frac>0.3. Next decision 1781 (ep15 val 2.5-ish epochs earlier -> visible then).
  Prediction: ep10 val 4-6%, ep15 val 8-12%, train loss ~4.6 by 1781. If ep15 val < 6% -> cut wd to 5e-4.
## Decision 9 (update 1781): val ep10 3.93, ep15 5.46 (below my 8-12% prediction -> slower than I guessed).
  But train loss is accelerating: 4.89 (1140-1203) -> 4.48 (1717-1780), ~-0.05/64 upd and steepening.
  gnorm rising 1.0->1.45, clip 0. Conv w norms still falling (layer3 52->38), not yet at equilibrium.
  Decided NOT to apply my "cut wd if <6%" rule: 5.46 is borderline, the curve is in its take-off phase,
  and in a 200-epoch run early speed matters less than regularisation for the final number.
  New check: ep20 val (update 2280) >= 9%, ep25 >= 14%. If ep20 < 8% -> cut wd to 5e-4.
## Decision 10 (update 2422): ep20 val 9.39 (met >=9 check). Train loss 4.43 -> 3.86 over 641 upd.
  val_loss 4.48 vs train 3.9 (train has LS+aug). gnorm 1.5->1.95, clip 0. w norm 84.8, flattening
  (equilibrium near). Kept. Next: ep25 >=14%, ep30 >=20%. Train loss ~3.4 by 3064.
## Decision 11 (update 3064): ep25 val 19.04 (>=14 check passed comfortably). train loss 3.80 -> 3.10.
  val_loss 3.65. w norm at equilibrium ~84.3 (layer4 still slowly shrinking, fc growing 19->21.3).
  gnorm 2.0->2.4, clip 0. Kept. Next: ep30 val >=25%, ep35 >=30%. Train ~2.6 by 3705.
  Watch: train-val gap. Train floor with LS ~ 1.0; if train loss < 1.5 by ~ep80 while val plateau -> raise wd.
## Decision 12 (update 3705): ep30 val 34.9 (>=25 passed). train 2.98 -> 2.44; val_loss 2.77 ~ train
  (no overfit gap yet). w norm 86.3 slowly growing, gnorm ~2.4 flat, clip 0. Kept.
  Next: ep35 >=40, ep40 >=45. Train ~2.0 by 4346.
## Decision 13 (update 4346): ep35 val 42.51 (>=40 passed), val_loss 2.32. Train 2.50 -> ~2.15
  (slowing, ~0.035/64 upd). gnorm 2.45 flat, clip 0, w norm 87.8. Kept.
  Next: ep40 >=47, ep45 >=51. Train ~1.85 by 4988.
## Decision 14 (update 4988): ep40 val 51.02 (>=47 passed), val_loss 2.01. Train 2.12 -> ~1.95
  (~0.025/64). gnorm 2.45, clip 0, w norm 89.1 creeping up. Kept. Next: ep45 >=55, ep50 >=58.
## Decision 15 (update 5629): ep45 val 56.6 (>=55 passed), val_loss 1.82. Train 1.92 -> ~1.80.
  gnorm 2.5, clip 0, w norm 90.1. Kept. Next: ep50 >=59, ep55 >=61.
## Decision 16 (update 6270): ep50 val 55.96 (missed >=59; flat vs 56.6, within noise ~1.8pt sd).
  val_loss 1.81 (flat). Train 1.82 -> ~1.71, still falling. No overfit gap (train~val loss).
  Reading: high-lr noise floor, not overfitting. Gains should come as cosine decays. Kept.
  Next: ep55 >=57, ep60 >=59. If val flat (<58) through ep65 while train keeps dropping -> consider
  raising wd (overfit) ; if train also flat -> lr noise floor, fine.
## Decision 17 (update 6911): val ep55 51.27 (val_loss 2.27 spike), ep60 55.71 (val_loss 1.75, new best loss).
  Val acc plateau ~56 over ep45-60; val loss still trending down (1.82,1.81,[2.27],1.75). Train 1.65 -> 1.58.
  Ep55 spike = high-lr fluctuation (weights/BN stats), not trend. Train LS-loss 1.6 vs val CE 1.75:
  modest gap emerging. Kept (lr 0.067; most gain expected as cosine decays past ~ep100).
  Next: ep65 >=56, ep70 >=58. Raise wd only if val_loss starts rising while train falls.
## Decision 18 (update 7552): ep65 val 61.8, val_loss 1.597 (both best; plateau broken). Train ~1.52.
  gnorm 2.45, clip 0, w norm 91.6 flat. Kept. Next: ep70 >=61, ep75 >=63.
## Decision 19 (update 8194): ep70 val 59.26 / 1.663 (missed >=61 by noise; ep65 was 61.8). Train ~1.48.
  Kept. Running val mean ep60-70 ~59; expect rise as lr falls below ~0.05 (ep ~95).
## Decision 20 (update 8835): ep75 val 65.99 / 1.400 (best). Train ~1.44. lr 0.058. Kept.
  Next: ep80/85 avg >=65.
## Decision 21 (update 9476): ep80 val 65.86 / 1.387 (best loss). Train flat ~1.41 (lr-noise floor at
  lr 0.055). No overfit. Kept. Next: ep85/90 avg >=66; train should resume falling as lr decays.
## Decision 22 (update 10118): ep85 val 67.89 / 1.288 (best). Train 1.41 -> ~1.34 (resumed falling, as
  predicted). lr 0.051. Kept. Next: ep90/95 avg >=68, train ~1.29 by 10759.
## Decision 23 (update 10759): ep90 val 68.91 / 1.321. Train ~1.31. lr 0.047. Kept. Next: ep95/100 avg >=69.
## Decision 24 (update 11400, halfway): ep95 val 70.3 / 1.33 (best acc). Train ~1.29. lr 0.043. Kept.
  Next: ep100/105 avg >=70.
## Decision 25 (update 12041): ep100 73.35/1.251, ep105 70.81/1.291 (avg 72.1, passed >=70). Train ~1.24.
  lr 0.040. Kept. Next: ep110/115 avg >=72.5, train ~1.19 by 12682.
## Decision 26 (update 12682): ep110 val 75.63 / 1.187 (best both). Train ~1.215 (flat-ish). lr 0.036. Kept.
  Next: ep115/120 avg >=74.
## Decision 27 (update 13324): ep115 val 67.77 / 1.458 — spike (like ep55: 51.3/2.27 then recovered next ep).
  Train still falling 1.20 -> 1.17, no other anomaly. Single-point spike = BN/weight fluctuation at lr 0.033,
  not overfit trend (overfit would show gradually). Kept. If ep120 also >1.35 val_loss -> raise wd to 1e-3.
## Decision 28 (update 13965): ep120 val 77.03 / 1.053 (best both) — ep115 spike confirmed transient.
  Train ~1.15. lr 0.029. Kept. Next: ep125/130 avg >=76.
## Decision 29 (update 14606): ep125 val 79.06 / 1.014 (best). Train ~1.10. lr 0.025. Kept. Next: ep130/135 avg >=78.
## Decision 30 (update 15248): ep130 val 77.79 / 1.073 (noise vs 79.06). Train ~1.08. lr 0.022. Kept.
  Next: ep135/140 avg >=78.5.
## Decision 31 (update 15889): ep135 val 77.28 / 1.070. Val flat ep125-135 (79.1,77.8,77.3; loss ~1.01-1.07)
  while train 1.10 -> 1.04. Mild sign of fit outpacing generalisation, but val loss not rising. Kept:
  remaining cosine (lr 0.019 -> 0) usually yields several points; late wd change has little time to act.
  Next: ep140/145 avg >=78.5; ep150+ should exceed 80.
## Decision 32 (update 16530): ep140 val 80.96 / 0.927 (best both; plateau broken). Train ~1.02. lr 0.0156. Kept.
  Next: ep145/150 avg >=80.
## Decision 33 (update 17171): ep145 80.33/0.946, ep150 83.25/0.911 (best). Train ~1.00. lr 0.0127. Kept.
  Next: ep155/160 avg >=82.
## Decision 34 (update 17812): ep155 83.25/0.838 (best loss). Train ~0.975. lr 0.010. Kept. Next: ep160/165 avg >=83.
## Decision 35 (update 18454): ep160 85.28/0.785 (best both). Train ~0.96. lr 0.0077. Kept. Next: ep165/170 avg >=84.5.
## Decision 36 (update 19095): ep165 84.14/0.830 (noise vs 85.28). Train ~0.94. lr 0.0058. Kept. Next: ep170/175 avg >=84.5.
## Decision 37 (update 19736): ep170 86.04/0.761 (best both). Train ~0.93 (flattening). lr 0.004. Kept.
  Next: ep175/180 avg >=85.5.
## Decision 38 (update 20378): ep175 85.79/0.766 (~= ep170). Train ~0.917. lr 0.0025. Kept; converging.
## Decision 39 (update 21019): ep180 86.29/0.742 (best). Train ~0.91. lr 0.0014. Kept; converging.
## Decision 40 (update 21660, last): ep185 86.55/0.743 (best acc). Train ~0.906. lr 0.00056 -> 0. Kept to end.
  Expected final test ~85-87%.
- Ideas for later: if train loss gets near floor (~1.0 with LS) while val lags a lot -> raise wd further.
  If underfitting (train loss high late) -> lower wd. EMA risky because BN stats come from raw weights.
