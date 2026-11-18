# NOTES

## Decision 1 (update 71)
- Loss 5.45 -> 5.26 over 71 updates (ln196 = 5.28): slow, because of the zero-initialised head plus the slow 570-update warmup.
- Changed: my own foreach AdamW (Adam state carried over from the user optimizer). The head (classifier.*) gets a
  5e-3 peak with a 114-update warmup; the backbone stays at 1e-3 with a 570-update warmup, then cosine to 0. wd unchanged.
- Analysis: per-role grad/param norms, relative weight change between calls, lrs, validation. EVERY=57.
- Prediction: loss < 4.8 by update ~200 and < 4.0 by ~570. First validation (epoch 2, update ~228) should be well above chance.
  If the loss goes flat or spikes after the head lr jump, the head lr is too high.
- Plans: once warmup ends, compare val curves. Consider raising the backbone lr (Cars is far from ImageNet;
  a fine-grained task usually needs strong adaptation) if train loss lags. Watch for overfitting around 37 img/class
  (train loss << val loss) and use more wd/regularisation if so.

## Decision 2 (update 142)
- Loss went 5.26 -> 3.69 by update 141, beating my prediction of < 4.8 by update 200. The head lr jump caused no instability.
- At update 114 the early layers' grad norm (42) is much larger than the late layers' (10). Adam normalises this, so no action.
- Kept both programs. Waiting for the first validation (~228) and the end of the backbone warmup (570).

## Decision 3 (update 214)
- Loss 3.69 -> 3.12. Relative weight change per half epoch (dW/W): head 0.52, conv weights about 0.045, BN about 0.001.
- Kept. Still no validation. Next: judge the backbone lr from the epoch 2 and 4 validation results and the trend in train loss.

## Decision 4 (update 285)
- Validation at epoch 2: 35.66% accuracy, loss 2.65. Train loss is about 2.6, flat over updates 228-284 while the backbone lr ramps 4e-4 -> 5e-4.
- dW/W per half epoch: conv weights 0.055-0.058, head 0.32 (falling).
- Kept. Watch epoch 4 (~456) and epoch 6 (~684). If train loss falls well below val loss early, think about
  more regularisation (higher wd). If val improves slowly after the peak lr, think about a higher backbone lr.

## Decision 5 (update 356)
- Train loss steps down at epoch boundaries (228: 3.1 -> 2.6; 342: 2.63 -> 2.24) and is flat inside an epoch,
  so the model is starting to memorise repeated images. Expect overfitting pressure later; higher wd is the main lever.
- Kept. Decide on wd from val at epoch 4/6, comparing val loss with train loss.

## Decision 6 (update 428)
- In epoch 3 train loss rose 2.25 -> 2.55 while the backbone lr ramped 6e-4 -> 7e-4. The rising lr is roughly
  cancelling the progress, which hints that a 1e-3 backbone peak is on the high side rather than too low. Don't raise it.
- Kept; waiting for val at epoch 4 (456).

## Decision 7 (update 499)
- Validation at epoch 4: 31.09% accuracy, loss 2.948. Epoch 2 was 35.66% and 2.65. Meanwhile train loss went to about 2.2.
  So val got worse while the backbone lr rose to 8e-4: the pretrained features are being disrupted.
- Changed (version 2): head lr 5e-3 -> 2.5e-3; backbone peak lr 1e-3 -> 4e-4; early layers (stem, blocks 0-2)
  2e-4. The warmup still ends at 570 and the cosine schedule is unchanged. Adam state carried over.
- Prediction: val at epoch 6 (684) >= 40% and val loss < 2.6, then rising steadily.
  If val is still <= 35% at epoch 8, lower the backbone lr further (e.g. 2e-4 / 1e-4) or move toward SGD-style small updates.

## Decision 8 (update 570)
- After the lr cut, train loss falls steadily within the epoch (2.16 -> 1.81); before, it only dropped at epoch boundaries.
  dW/W per half epoch: early 0.018, late 0.036, head 0.117.
- Kept the updater. Analysis now wakes me if val < 38% at epoch >= 6, or if val falls > 4 points below its best.

## Decision 9 (update 891)
- Validation at epoch 6: 71.32% accuracy, loss 1.228 (epoch 4 was 31.09%). The lr cut clearly worked.
- Train loss is 1.39 (it includes label smoothing and augmentation), so no sign of overfitting yet. It falls about 0.06 per epoch.
- dW/W per half epoch is steady: early 0.015, late 0.032, head 0.095.
- Kept. Open question: would an even lower lr be better? Revisit if val plateaus while lr is still high
  (e.g. val flat over epochs 10-20). Late-run idea: a short high-to-zero tail. Weight averaging needs care
  because BN running stats come from the training forward passes.

## Decision 10 (update 1211)
- Val acc 71.32 / 73.86 / 72.21 at epochs 6 / 8 / 10; val loss flat at 1.21-1.23. Train loss 1.39 -> 1.27.
- Reading: a plateau at the high-lr noise floor, not overfitting (val loss isn't rising). Expect gains as the cosine decays.
- Kept. Watch for val loss rising while train falls (overfitting), which would call for more wd or a faster decay.
  If val stays flat through epoch 20 with no overfitting, consider a faster decay (spend more of the run at low lr).

## Decision 11 (update 1532)
- Val at epoch 12: 72.08% accuracy, loss 1.307 (up from 1.21-1.22). Accuracy flat at about 72% since epoch 6. Train loss 1.21.
  That is the start of overfitting at a high lr.
- Changed (version 3): all peak lrs x0.6 (head 1.5e-3, late 2.4e-4, early 1.2e-4). The cosine shape is unchanged.
  Why the lr and not wd: wd on BN-followed convs is mostly effective-lr control, and the earlier lr cut paid off a lot.
- Prediction: val loss back to about 1.2 or lower and accuracy 74-76% by epochs 16-18.
  Wrong if accuracy < 71 or val loss keeps climbing (> 1.35). If so, think about wd/regularisation of the head and the BN-free layers (conv_head, SE).

## Decision 12 (update 1852)
- Prediction confirmed: val 77.03% / 76.27% at epochs 14 / 16, val loss 1.115 / 1.118 (was 72.08% / 1.307). Train loss 1.10.
- Both lr cuts helped. Part of that may be an "early anneal" effect that the cosine would have delivered anyway,
  but lower peaks keep the weights closer to the pretrained ones, which suits small data.
- Kept. If val flattens again and val loss rises (> 1.2) by epochs 20-24, cut again (x0.6) or reshape the cosine to decay sooner.

## Decision 13 (update 2173)
- Val at epoch 18: 77.54% accuracy, loss 1.142. Train loss is flattening at about 1.085-1.09 (noise floor at the current lr).
- Kept. The trigger is unchanged (val loss > 1.2 while accuracy is flat).

## Decision 14 (update 2494)
- Val accuracy 77.03 / 76.27 / 77.54 / 77.41 over epochs 14-20 (a plateau). Train loss flat at 1.07-1.08 for 3 epochs.
  On the cosine the lr stays > 50% of peak until about update 6000, so the plateau would waste about 30 epochs.
- Changed (version 4): lrs x0.6 again (_S = 0.36; head 9e-4, late 1.44e-4, early 7.2e-5 peak).
- Prediction: val >= 78.5% at epochs 24-26, val loss <= 1.12, train loss drops to about 1.0.
  Wrong / too low if train loss drops fast while val stays at or below 77.5 and val loss rises above 1.2: then partly revert (x1.4).
- Late-run idea: EMA/SWA of the weights over the tail, swapped in at about update 11250, so the last ~150 near-zero-lr
  updates recompute the BN running stats for the averaged weights.

## Decision 15 (update 2814)
- Val at epoch 24: 79.57% accuracy, loss 1.086 (best so far). Epoch 22 (77.03%) came only 14 updates after the cut.
  Train loss 1.03-1.04, flattening. dW/W per half epoch: early 0.006, late 0.012, head 0.038.
- Prediction (>= 78.5%) confirmed. Kept. Track 3 lr cuts: each gave a step of +2 to +5 points.
- Next: if a plateau appears again by epochs 28-32, consider another x0.7, or bring the cosine decay forward.

## Decision 16 (update 3135)
- Val at epoch 26: 78.68% accuracy, loss 1.136 (within noise of 79.57% / 1.086). Train loss flat at about 1.02.
- Kept. Decide at the next decision (with epochs 28 and 30): if the mean of epochs 26-30 is <= 79.5, cut x0.7 or bring the decay forward.

## Decision 17 (update 3456)
- Val accuracy at epochs 26/28/30: 78.68 / 79.31 / 78.93 (mean 78.97, a plateau). Val loss 1.086 -> 1.136 -> 1.164 -> 1.16 (slowly rising).
  Train loss flat at about 1.02.
- Changed (version 5): lrs x0.7 (_S = 0.252; head 6.3e-4, late 1.0e-4, early 5e-5 peak; current lr about 0.85x of that).
- Prediction: val about 80% at epochs 32-34 and val loss <= 1.12. Wrong if val <= 78.5 and val loss > 1.2 (then lowering lr
  no longer helps, and the issue is overfitting rather than noise).
- Still pending: tail weight averaging (implement by about update 8000 at the latest).

## Decision 18 (update 3776)
- Val at epoch 32: 80.08% accuracy, loss 1.122 (best so far), so the prediction held. Train loss about 1.00.
- Kept.
- Averaging plan: SWA-style uniform average from about update 8000; copy it into the params at about update 11100 (cosine factor
  about 0.007 there), so about 300 low-lr updates refresh the BN running stats. Epoch 98's val (11172) then shows the effect.
  Keep the pre-swap weights so an extra decision can revert if val drops. Implement around update 7500-7900.

## Decision 19 (update 4097)
- Val at epoch 34: 79.31% accuracy, loss 1.131. Train loss flat at about 0.99. Gains per lr cut are shrinking (+4.5, +5, +2, +1).
- Kept. The cosine now decays meaningfully. Another cut is an option only if val drops with val loss > 1.18.
  SAM is not possible (the updater only gets grads; no extra forward pass).

## Decision 20 (update 4418)
- Val at epochs 36/38: 79.57 / 79.82, val loss 1.14 / 1.133. Train loss about 0.985. Stable, no overfitting drift.
- Kept. Next action: tail averaging around update 7500-7900.

## Decision 21 (update 4738)
- Val at epoch 40: 81.09% accuracy (best), loss 1.135. Train loss about 0.975-0.98. Slow improvement as the cosine decays.
- Kept.

## Decision 22 (update 5059)
- Val at epochs 42/44: 80.84 / 80.58, val loss 1.131 / 1.143. Train loss about 0.975. Stable plateau around 80.5-81.
- Kept.

## Decision 23 (update 5379)
- Val at epoch 46: 80.96% accuracy, loss 1.136. Train loss about 0.97-0.98.
- Changed (version 6): added tail averaging (inactive until 8000). Uniform avg over updates 8000-11099; at update 11100 the
  params are set to the average (pre-swap weights kept in self.pre); ~300 near-zero-lr updates then refresh the BN running stats.
  The lr schedule is unchanged. REVERT_SWAP=True in a later build restores the pre-swap weights.
- Check after 8000: no errors/rollback; analysis could report n_avg. After 11100: val at epoch 98 (11172) vs epoch 96 (10944).
  If it drops by > 0.8 points, revert (needs a wake/decision before about 11300).

## Decision 24 (update 5700)
- Version 6 installed without error. Val at epoch 48: 80.2% accuracy, loss 1.149. Train loss about 0.975. Stable at 80-81.
- Kept.

## Decision 25 (update 6021)
- Val at epochs 50/52: 81.98 / 80.33, val loss 1.147 / 1.155 (very slow upward creep). Train loss about 0.965.
- Kept.

## Decision 26 (update 6341)
- Val at epoch 54: 81.73% accuracy, loss 1.145. Train loss about 0.96. Steady at about 81.
- Kept.

## Decision 27 (update 6662)
- Val at epochs 56/58: 80.71 / 80.84, val loss 1.145 / 1.158. Train loss about 0.955. Steady.
- Kept. Averaging starts at 8000.

## Decision 28 (update 6982)
- Val at epoch 60: 80.58% accuracy, loss 1.122. Train loss about 0.95. Steady. Kept.

## Decision 29 (update 7303)
- Val at epochs 62/64: 81.6 / 80.71, val loss about 1.147. Train loss about 0.95. Kept; averaging starts at 8000.

## Decision 30 (update 7624)
- Val at epoch 66: 81.09% accuracy, loss 1.161. Train loss about 0.947. Kept.

## Decision 31 (update 7944)
- Val at epoch 68: 81.73% accuracy, loss 1.147. Train loss about 0.945. Kept. Check n_avg > 0 at the next decision (8265).

## Decision 32 (update 8265)
- Averaging works: n_avg = 265 at 8265, no errors, 14.3 ms per update. Val at epochs 70/72: 82.11 / 81.98 (best pair), loss about 1.15.
- Kept. The swap happens at 11100; the analysis wakes at the first validation after it (epoch 98, update 11172).

## Decision 33 (update 8586)
- Val at epoch 74: 82.11% accuracy, loss 1.135. n_avg 550 at 8550. Train loss about 0.943. Kept.

## Decision 34 (update 8906)
- Val at epochs 76/78: 81.98 / 81.6, val loss about 1.15. n_avg 892. Kept.
- Timing: the last scheduled decision is about 10829. The wake at the epoch 98 val (11172) is more than 100 updates after it, so it's allowed.
  A revert there leaves about 228 updates for the BN stats.

## Decision 35 (update 9227)
- Val at epoch 80: 82.49% accuracy (best), loss 1.155. n_avg 1177. Train loss about 0.938. Kept.

## Decision 36 (update 9548)
- Val at epoch 82: 82.49% accuracy, loss 1.165. n_avg 1519. Kept.

## Decision 37 (update 9868)
- Val at epochs 84/86: 81.35 / 81.6, val loss 1.181 / 1.166 (noise around 82). n_avg 1861. Kept.

## Decision 38 (update 10189)
- Val at epoch 88: 81.6% accuracy, loss 1.169. n_avg 2146. Kept. The weights moved little since 8000, so the swap is low-risk.

## Decision 39 (update 10509)
- Val at epochs 90/92: 82.36 / 82.36, val loss about 1.165. n_avg 2488. Kept.

## Decision 40 (update 10830, last scheduled)
- Val at epoch 94: 81.35% accuracy, loss 1.168. n_avg 2830. Kept; the swap goes ahead at 11100.
- The wake comes at the analysis call after the epoch 98 val (about 11229), leaving about 171 updates.
- Revert rule at the wake: revert if epoch 98 val < (mean of epochs 90-96) - 1.0 AND val loss is up. The val loss is a less noisy signal than accuracy.

## Decision 41 (update 11229, wake after the swap)
- The swap happened at 11100 (n_avg 3100). Epoch 96 (pre-swap): 82.11% / 1.147. Epoch 98 (post-swap): 81.73% / 1.147.
  The mean of epochs 90-96 is 82.05, so -0.32 points (noise) with identical val loss. Train loss unchanged at about 0.935.
- The revert rule was not triggered. Kept the averaged weights. The run ends at 11400; the BN stats get ~300 refresh updates after the swap.

