# NOTES

## Decision 1 (update 142)
- Loss ~5.29 ≈ ln(196)=5.278: still at chance in warmup (lr factor 0.25 at u142). Normal.
- Installed v1 updater: own foreach SGD-Nesterov, identical hyperparameters to baseline, momentum
  carried from torch SGD state. Purpose: own state + per-role knobs (stem/conv/head/norm_bias).
- New analysis: per-stage grad norm, param norm, relative change per 50 updates (drel), val.
- Prediction: loss trajectory unchanged vs baseline; ms/update ≈ 105 (maybe slightly less).
  Loss should start falling below ~5.1 by update ~400-600 as warmup completes.

## Decision 2 (update 285)
- v1 installed fine, 105.1 ms (same). Loss 5.21-5.28, slowly drifting down; min 5.05.
- drel per 50 updates: stem.w 0.15-0.19, fc 0.14-0.24, body convs 0.005-0.01. Stem norm grows
  2.54->2.68 (scale-invariant via BN, self-limits). Nothing alarming.
- Kept everything. Prediction unchanged: loss <5.1 mean by ~600, clear descent during epochs 5-10.
  If loss still ≈5.2 at update ~800 (lr at peak), consider whether lr is too high/low.

## Decision 3 (update 428)
- Loss blocks fell 5.26 -> 5.17 (min 4.90). Learning has begun, on track. drel body 0.01-0.014,
  stem ~0.19, fc ~0.19. Conv weight norms slowly shrinking (wd), stem growing.
- Kept. Prediction: block means < 5.1 by ~600, < 4.9 by ~900. First val at epoch 5 (u570):
  expect val acc ~1-3%.

## Decision 4 (update 570, warmup done)
- Loss blocks 5.20 -> 5.09; prediction "<5.1 by 600" just met. Grad norm falling 1.55->1.35
  (fc grad dominating, falling). Body drel 0.012-0.014/50 upd at peak lr. No val yet (comes at u570).
- Kept. Prediction: ~4.9 by u900, ~4.6 by u1400. Val acc at ep5 ~1-3%, ep10 ~3-8%.
  If loss plateaus >5.0 through u1000 -> consider peak lr change.

## Decision 5 (update 712)
- Val ep5: 2.03% / loss 5.109 (prediction 1-3% ✓). Train loss 5.03-5.07, last block 4.97-5.02.
- Conv norms shrinking ~1.2% per 50 upd under wd (layer3 135->122) -> effective lr rising. OK.
- Kept. Prediction: train ~4.9 by u900; val ep10 (u1140) 3-8%.

## Decision 6 (update 855)
- Train loss blocks 4.93-5.02 (mean ~4.98); slightly behind "4.9 by u900" but falling ~0.1/300 upd.
  Stats stable; body drel ~0.014-0.017/50 upd. No new val.
- Kept. Wait for val ep10 (u1140) before judging lr. Prediction: train ~4.85 at u1140, val 3-8%.

## Decision 7 (update 998)
- Train blocks 4.88-4.98 (mean ~4.93). Slow steady descent. Stats unchanged. No val.
- Kept; val ep10 arrives at u1140 (next decision). Same prediction (val 3-8%).

## Decision 8 (update 1140)
- Train blocks 4.83-4.95 (mean ~4.89). Slightly behind 4.85 prediction but within noise.
  Grad norms rose a bit (1.09->1.31->1.23), early-layer grads up (stem 0.24->0.50): features forming.
- Val ep10 not yet seen (computed after u1140). Next decision at u1781 (long gap).
- Kept updater. analysis.py: wake if val ep10 <3%, val drop >3 pts, or loss interval mean > EMA+0.5.
- Prediction: val ep10 3-8%; train ~4.6-4.7 by u1781; val ep15 (u1710) 6-12%.

## Decision 9 (update 1781)
- Val ep10 3.30% (loss 5.052), ep15 5.58% (4.908): within predictions (3-8, 6-12 borderline).
- Train 64-upd blocks 4.86 -> 4.66; prediction 4.6-4.7 met. Descent accelerating a bit.
- Conv norms keep shrinking (layer3 108->94); drel rising (layer1 0.020->0.028/50 upd) =
  effective lr rising toward wd/lr equilibrium. Grad norms rising 1.26->1.50 (early layers).
- No overfit sign (val loss 4.91 vs train 4.70 w/ smoothing+aug). Kept everything.
- Prediction: val ep20 (u2280) 8-14%; train ~4.4-4.5 by u2422.

## Decision 10 (update 2422)
- Val ep20: 5.58% (same as ep15), loss 4.937 (ep15 4.908). Prediction 8-14% MISSED.
- Train blocks 4.66 -> 4.42 (prediction 4.4-4.5 met). So train improving, val flat: early
  generalization gap? But 5.58% = 44/788 imgs, SE ~0.8 pt; one point is weak evidence.
  Val also uses centre crops + BN stats from RRC; at high lr val is noisy.
- Effective lr still rising: conv norms shrinking (layer3 92->80), drel layer1 0.030->0.041.
- Kept for now. Prediction: val ep25 (u2850) >= 7%, train ~4.25 by u3064.
  If ep25 val <= 6.5% while train keeps falling -> treat as overfitting: raise wd (e.g. 1e-3 on
  convs) or other regularization. If train stalls too -> lr issue.

## Decision 11 (update 3064)
- Val ep25: 3.93%, loss 6.559 (!; was 4.94). Prediction (>=7%) MISSED badly.
- Train fell fast 4.40 -> 3.97 (prediction 4.25 beaten). So train/val diverging hard.
- Interpretation (uncertain): val loss > ln196 = confidently wrong. Either (a) early overfitting
  (37 img/class) or (b) high-noise regime making mid-run eval erratic (eff. angular step for
  BN-convs ~sqrt(2*lr*wd/(1+m)) ≈ 0.005/step, ~1.6x ImageNet recipe, at batch 64; layer1 drel
  0.017 -> 0.052 per 50 upd as norms shrink). Final test is after lr->0 so (b) matters less.
- Raising conv wd would raise effective lr further (not clearly right). fc (no BN after it) wd
  directly limits logit scale; fc norm 19 -> 20 growing. Candidate lever if overconfidence persists.
- Kept updater (one val point). Added wake if ep30 val acc <6 or loss >6 (reported at u3450).
- Prediction: ep30 val acc 5-10%, val loss back below 5.5. Train ~3.6 by u3705.

## Decision 12 (update 3705)
- Val ep30: 13.58%, loss 3.983 -> ep25 (3.93%/6.56) was an erratic mid-run eval, not overfit.
  Lesson: single val points at high lr are very noisy; don't react to one.
- Train 3.83 -> 3.25 (prediction 3.6 beaten). Learning in fast phase now. drel body 0.03-0.06.
- Kept updater. Made val-drop wake require two consecutive drops (avoid spurious wakes).
- Prediction: val ep35 (u3990) 15-25%; train ~2.9 by u4346.

## Decision 13 (update 4346)
- Val ep35: 20.69% / 3.586 (prediction 15-25 ✓). Train 3.30 -> 2.69 (pred ~2.9, beaten).
- Train-val loss gap: train(LS) 2.9 at u4000 vs val(no LS) 3.59. Watch it widen.
- Kept. Prediction: val ep40 (u4560) 24-32%; train ~2.2 by u4988.
- Watch item: when train loss nears LS floor (~1.0-1.3) while lr is still high, val may plateau ->
  consider stronger regularization or schedule shape then.

## Decision 14 (update 4988)
- Val ep40: 28.68% / 3.258 (pred 24-32 ✓). Train 2.58 -> 2.15 (pred 2.2 ✓).
- Gap train(LS)~2.4 vs val 3.26 at u4560 (was 2.9 vs 3.59 at u4000): ~0.85, slowly widening.
- Body drel ~0.07/50 upd, stable (equilibrium reached). Kept.
- Prediction: val ep45 (u5130) 31-38%, ep50 35-42%; train ~1.85 by u5629.

## Decision 15 (update 5629)
- Val ep45: 39.09% / 2.606 (pred 31-38, beaten). Train 2.10 -> 1.81 (pred 1.85 ✓).
- Gap at u5130: train ~2.0 vs val 2.61 (~0.6, narrower than before). fc norm plateaued ~27.1.
- Kept. Prediction: val ep50 (u5700) 40-47%, ep55 (u6270) 44-50%; train ~1.6 by u6270.

## Decision 16 (update 6270)
- Val ep50: 44.29% / 2.452 (pred 40-47 ✓). Train 1.83 -> 1.62 (pred 1.6 ✓), slowing.
- Gap at u5700: train ~1.78 vs val 2.45 (~0.67). Stable. drel stable ~0.07.
- Kept. Prediction: val ep55 (u6270, arrives now) 45-51%, ep60 (u6840) 48-54%; train ~1.45 by u6911.

## Ideas / plan
- Small dataset, from scratch: regularization matters. Candidates once evidence arrives:
  higher wd (1e-3) if train loss falls far below val loss; peak lr adjustments if loss stalls/spikes.
- Weight averaging is risky: test uses BN running stats from training forward passes (would
  mismatch averaged weights). Only consider late with tiny lr, or not at all.
- Final phase: cosine to 0 already; consider ending at exactly 0 for BN stat consistency.
