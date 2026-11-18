# NOTES

## Beliefs
- Baseline: SGD-N 0.9, lr 0.05 (batch 64; already ~2x Goyal linear-scaling), wd 5e-4, warmup 570, cosine.
- Small data (37 img/class) from scratch -> regularisation (wd) is likely the main lever; lr*wd sets
  effective lr of BN-scaled conv weights. Baseline lr*wd = 2.5e-5.
- Eval uses final params + training BN stats -> avoid weight averaging that would mismatch BN stats.

## Log
- D1 (u142): loss 5.30 -> 5.24 (ln196=5.278), still in warmup. Installed v1: own foreach SGD-N,
  same lr/schedule, wd 1e-3 on conv+fc (lr*wd = 5e-5). Momentum carried from user SGD.
  Prediction: no instability; by u~1140 (epoch 10) train loss slightly higher than baseline would be,
  conv pnorm grows less. Val at u570, 1140 is the first evidence. If val stalls / loss diverges, revert wd.
- D2 (u285): v1 installed fine, 105 ms. Loss ~5.24, std rising (0.05->0.09) = starting to fit some batches.
  Conv pnorm 211->208 (decay dominating early). Kept. Watch: loss should drop below ~5.0 by u~700.
- D3 (u428): loss 5.25 -> ~5.17 (last 14: 5.146). Conv pnorm 205->199. Kept. Next: first val at u570.
- D4 (u570): loss ~5.11-5.13 (plateau-ish 5.06-5.16). Conv pnorm 187 (falling ~4/50 upd). Warmup done. Kept.
  Val for epoch 5 not yet seen. If loss > 5.05 at u850, consider higher lr / head changes.
- D5 (u712): val@570 = 2.79% (pred 1-3%: ok). Loss 4.96 by u700 (pred <5.0 by 850: met early).
  Conv pnorm 182->173 still falling (~4.5/50). Kept. Next check: val@1140 expect ~6-10%.
- D6 (u855): loss ~4.95 (slow, ~-0.05/140 upd). Conv pnorm 161, still -4/50. Kept; decide on lr at val@1140.
- D7 (u998): loss ~4.92 (4.88-5.00). Conv pnorm 153. Kept; next decision u1140 sees val@1140.
- D8 (u1140): loss ~4.89 (4.86-4.95), slow. Conv pnorm 142. Val@1140 not yet seen (runs after this).
  Next scheduled decision is u1781 -> added one-shot wake at u1250 in analysis. Kept updater.
- D9 (u1250, wake): val@1140 = 4.06% (pred 4-8%, low end; below my 5% trigger). Loss ~4.87.
  Conv pnorm 132 (238 at start). Installed v2: peak lr 0.05->0.1 ramped over u1250-1550 (cosine
  shape unchanged, ends at 0 at u22800), wd 1e-3 -> 7.5e-4.
  Prediction: no instability (loss stays < 5.0 during ramp); loss < 4.6 by u1781; val@1710 >= 8%.
  Wrong if loss rises above ~4.95 sustained or val@1710 < 6% -> revert to 0.05 peak.
- D10 (u1781): val@1710 = 4.44% (pred >=8%: WRONG, below my 6% revert line). Loss 4.77 (pred <4.6: wrong),
  but loss trend improved late (4.84 -> 4.77 over last ~150 upd). No instability at lr 0.1.
  Conv pnorm 96 (still -3.7/50, decay-dominated). Not reverting: lr doubling did no harm, reverting
  costs time; the bottleneck seems intrinsic slow start. Added diagnostics: bn3 gain |mean| per stage,
  conv momentum-buffer norm. Next: val@2280. If still < 8% then consider a structural change
  (e.g. higher lr for BN/bias/head, or adaptive method for non-scale-invariant params).
- D11 (u2422): TAKEOFF. Loss 4.77 -> 4.32 (last 64 upd), accelerating. val@2280 = 5.33% (val loss 4.66),
  val lags train. Conv pnorm 62 (decline slowing ~2/50 -> approaching equilibrium). bn3 |gain| growing
  slowly (L1 .089, L4 .026). Kept: the run is now learning fast; no evidence for structural change.
  Prediction: val@2850 >= 12%, train loss < 4.0 by u3064. Watch for conv norm collapse / noisy loss.
- D12 (u3064): train loss 4.33 -> 3.77 (pred <4.0: ok). val@2850 = 4.19%, val loss 6.08 (> ln196!)
  (pred >=12%: WRONG). Interpretation: eval-mode instability at high effective lr (conv norm 46,
  shrinking; BN running stats lag fast-moving weights) -> confident wrong logits; not overfitting
  (train loss still ~3.8 w/ LS). Kept for now. Test: val@3420. If val loss still > 5 while train < 3.6,
  lower conv effective lr (reduce wd on conv so norm stops shrinking, or lower peak lr ~0.07).
- D13 (u3705): val@3420 = 16.24%, loss 3.86 (pred >=8% & <5: ok; the 6.08 was transient instability).
  Train loss 3.0 (pred 3.3-3.5: faster). Conv pnorm equilibrated at ~42.5. Train/val gap ~0.5 at u3420
  (train ~3.35 there). Kept. Watch gap: if by val@3990/4560 val loss stops falling while train
  keeps dropping -> raise wd (conv/head) to ~1.5e-3 for more regularisation.
  Prediction: val@3990 ~22-28%, train loss ~2.6 by u4346.
- D14 (u4346): val@3990 = 14.21%, loss 4.19 (pred 22-28%: WRONG). Train loss 2.55 (pred 2.6: ok).
  Gap train~2.75 vs val 4.19 at u3990 -> overfitting/regularisation trigger hit. Installed v3:
  wd conv/head 7.5e-4 -> 1.5e-3 ramped u4346-4646 (lr*wd ~1.4e-4 at peak). Expect conv norm to fall
  again toward ~36-38 (equilibrium ∝ (lr/wd)^(1/4)... ~0.84x), train loss decline slows.
  Prediction: train loss ~2.4-2.5 at u4988 (slower than without); val@4560 >= 18%, val@5130 >= 22%.
  Wrong if val keeps falling while train drops -> something else (eval-mode/BN noise?) is the issue.
- D15 (u4988): val@4560 = 41.37%, loss 2.44 (pred >=18%: ok, far above). Taken at start of wd ramp,
  so mostly not due to v3 -> the 14% at u3990 was eval noise at high lr. LESSON: single val points
  at high lr are very noisy (4.2 -> 16 -> 14 -> 41); don't react to one point.
  Train loss flat ~2.45 since wd increase (pred 2.4-2.5: ok). Conv pnorm 38.1 (pred 36-38: ok).
  Kept. Prediction: val@5130 >= 38%, train loss resumes slow decline as norm re-equilibrates.
- D16 (u5629): val@5130 = 37.56%, loss 2.67 (pred >=38%: marginal miss, within noise). Train loss
  ~2.42, slowly declining (last 64: 2.35). Conv pnorm ~38 equilibrium. Val/train gap small (val no-LS
  2.67 vs train LS 2.42) -> regularisation not excessive, not badly overfit. Kept.
  Prediction: val@5700/6270 in 40-50%, train ~2.3 by u6270. If two consecutive val < 35 -> wd 1e-3.
- D17 (u6270): val@5700 = 42.89%, loss 2.40 (pred 40-50: ok). Train ~2.25 (pred 2.3: ok).
  Conv pnorm 38.2 steady. Kept. Prediction: val@6840 45-52%, train ~2.15 by u6911.
- D18 (u6911): val@6270 = 40.36% (2.61), val@6840 = 37.06% (2.81) (pred 45-52: WRONG). Val acc
  plateau ~40% over 5 pts, val loss rising 2.40->2.61->2.81 while train 2.25->2.1 => overfitting.
  Installed v4: wd conv/head 1.5e-3 -> 2.5e-3 ramped u6911-7211 (lr*wd ~2e-4).
  Prediction: conv norm falls to ~33-34; train loss flattens/rises slightly (~2.1-2.2);
  val loss back below 2.6 and acc >= 42% by val@7980. Wrong if train loss rises > 2.35 and val
  doesn't improve -> too much; revert to 1.5e-3.
- D19 (u7552): train loss 2.10 -> 2.35 (at my revert threshold), val@7410 = 34.64% (loss 2.72, slightly
  better than 2.81). Conv norm 38.2 -> 34.5 (pred 33-34: ok), still settling -> transient; effective
  lr up. Kept v4 one more val point (val is noisy; transient not finished).
  Firm rule: if val@7980 < 38% AND train > 2.3 at u8194 -> revert wd to 1.5e-3.
- D20 (u8194): val@7980 = 40.23%, loss 2.45 (pred >=40 & <2.65: ok; revert rule not triggered).
  Train ~2.32 flat (pred 2.25-2.35: ok). Conv norm 33.7 settled. Kept v4.
  Val/train loss gap now ~0.13 (val no-LS 2.45 vs train LS 2.32) -> overfitting controlled.
  Expectation: acc improvements come with lr decay (lr 0.074 now). val@8550/9120 ~42-48%.
- D21 (u8835): val@8550 = 40.74%, loss 2.53 (pred 42-48: slight miss). Train 2.32 -> ~2.22.
  Val acc plateau ~40% (epochs 40-75) = high-lr noise floor; gains expected as lr decays (0.070 now).
  Kept. Prediction: val climbs past 45% by epoch ~90-100 (u10260-11400) as lr drops to ~0.05.
- D22 (u9476): val@9120 = 34.39%, loss 2.80 (noisy dip; series since ep40: 41,38,43,40,37,35,40,41,34).
  Train 2.22 -> 2.14. Single val points have ~±4pt noise at this lr; no trend to act on. Kept.
  Decision rule: if mean of val@ep85-95 < 42% while lr ≤ 0.055, consider faster lr decay / more wd.
- D23 (u10118): val@9690 = 44.16%, loss 2.29 (best val loss so far). Train ~2.07 (pred 2.05-2.10: ok).
  Kept. Prediction: val@10260/10830 43-50%.
- D24 (u10759): val@10260 = 46.07%, loss 2.30 (pred 43-50: ok; new best acc). Train ~2.0 (pred: ok).
  Kept. Prediction: val@10830/11400 45-52%; train ~1.95 by u11400.
- D25 (u11400): val@10830 = 48.35%, loss 2.15 (pred 45-52: ok). Rising trend 44.2->46.1->48.4.
  Train ~1.93 (pred 1.95: ok). Kept. Prediction: val@11970 48-54%, train ~1.85 by u12041.
- D26 (u12041): val@11400 = 43.91% (2.45), val@11970 = 59.90% (1.58!) (pred 48-54: exceeded).
  Huge val swings again (eval noise at lr ~0.05). Train ~1.83 (pred 1.85: ok). Kept.
  Prediction: val@12540 50-62% (noisy), upward trend continues as lr decays; train ~1.75 by u12682.
- D27 (u12682): val@12540 = 56.09%, loss 1.80 (pred 50-62: ok). Train ~1.73 (pred 1.75: ok). Kept.
  Prediction: val@13110 55-63%, train ~1.65 by u13324.
- D28 (u13324): val@13110 = 52.03%, loss 1.95 (pred 55-63: miss). Val loss 1.58->1.80->1.95 rising 2x,
  formally my "more regularisation" trigger, but 1.58 was likely a lucky high reading; avg val loss
  ep100-115 ~1.95 vs ep85-95 ~2.25 -> still improving on average. Train 1.73 -> 1.68. Kept.
  Rule: if val@13680 loss > 2.0 and acc < 52 -> raise wd (conv/head) to ~3.5e-3.
- D29 (u13965): val@13680 = 66.24%, loss 1.36 (pred 53-60: exceeded; best so far). Rule not triggered.
  Train ~1.58 (pred 1.6: ok). Kept. Prediction: val@14250 60-70%, train ~1.5 by u14606.
- D30 (u14606): val@14250 = 67.13%, loss 1.38 (pred 60-70: ok). Train ~1.50 (pred 1.5: ok). Kept.
  Prediction: val@14820 64-72%, train ~1.42 by u15248.
- D31 (u15248): val@14820 = 59.01%, loss 1.75 (pred 64-72: miss; one-off dip like ep115). Train ~1.43
  (pred 1.42: ok). Kept. Prediction: val@15390 63-72%. Rule: two consecutive < 60 -> reconsider.
- D32 (u15889): val@15390 = 69.04%, loss 1.29 (pred 63-72: ok; best). Train ~1.34 (pred 1.35: ok). Kept.
  Prediction: val@15960/16530 66-74%, train ~1.27 by u16530.
- D33 (u16530): val@15960 = 72.46%, loss 1.24 (pred 66-74: ok; best). Train ~1.29 (pred 1.27: ~ok). Kept.
  Prediction: val@17100 70-77%, train ~1.22 by u17171.
- D34 (u17171): val@16530 = 70.30% (1.28), val@17100 = 77.66% (1.00) (pred 70-77: ok; best).
  Train ~1.21 (pred 1.22: ok). Kept. Prediction: val@17670 74-80%, train ~1.16 by u17812.
- D35 (u17812): val@17670 = 80.84%, loss 0.92 (pred 74-80: slightly above; best). Train ~1.17
  (pred 1.16: ok). Kept. Prediction: val@18240 78-83%, train ~1.12 by u18454.
- D36 (u18454): val@18240 = 80.58%, loss 0.96 (pred 78-83: ok). Train ~1.11 (pred 1.12: ok). Kept.
  Prediction: val@18810/19380 80-85%, train ~1.07 by u19095.
- D37 (u19095): val@18810 = 85.03%, loss 0.79 (pred 80-85: ok; best). Train ~1.05 (pred 1.07: ok). Kept.
  Prediction: val@19380/19950 83-87%, train ~1.02 by u19736.
- D38 (u19736): val@19380 = 85.15%, loss 0.79 (pred 83-87: ok). Train ~1.015 (pred 1.02: ok). Kept.
  Prediction: val@19950/20520 84-88%; train ~0.99 by u20378.
- D39 (u20378): val@19950 = 86.93%, loss 0.70 (pred 84-88: ok; best). Train ~0.98 (pred 0.99: ok). Kept.
  Prediction: final val (ep200) ~86-89%.
- D40 (u21019): val@20520 = 88.45%, loss 0.69 (pred 86-89: ok; best). Train ~0.96 (pred: ok). Kept.
- D41 (u21660, last): val@21090 = 88.71%, loss 0.67 (best). Train ~0.95 flat. lr 0.00066 -> 0.
  Kept v4 to the end. Expected final val/test ~88-90%.
  Summary of what mattered: lr peak 0.05->0.1 (u1250) got the run out of the slow start;
  wd 1e-3 -> 1.5e-3 -> 2.5e-3 kept overfitting in check; val at high lr is very noisy (±5-10 pts),
  single points shouldn't drive changes.
- Plans: consider wd 2e-3..5e-3 if val gap (train loss vs val) grows large by epoch ~30;
  consider lr peak 0.1 if training looks too slow.
