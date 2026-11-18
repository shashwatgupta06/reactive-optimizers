# NOTES

## Decision 1 (update 71)
- Baseline: AdamW lr 1e-4 everywhere, warmup 570, cosine. The loss went from 5.281 to 5.180 in 71 updates (the zero-init head learns slowly at a tiny warmup LR).
- Installed v1: our own foreach AdamW with
  - layer-wise LR decay 0.85 (top block / norm lr 2e-4; block 0 about 3.4e-5; embed about 2.4e-5)
  - head lr 2e-3, wd 0.05 on matrices, 0 on 1-D tensors, pos_embed and cls_token
  - warmup shortened to 300 (absolute index), cosine to 0 at 11400
  - global-norm clip at 5 (only for spikes)
  - Adam moments carried over from the user AdamW.
- Analysis: grad norm per section, the distance each section has moved relative to its weights at the first analysis call (update 100, not init), clip coefficient and val.
- Predictions: the train loss falls much faster, to below about 4.0 by update 300 and well below 2 by about epoch 5. Val acc should be above 50% at the first validation (epoch 2, update 228), and I expect it to reach 85-90%+ later.
  Wrong if the loss stalls or spikes, or if clip_coef is often below 1 → reduce top_lr/head_lr.
## Decision 2 (update 142): kept
- The loss fell from 5.18 to 3.42 in 71 updates (fast, as predicted). At update 100 the gradient norm before clipping was 36 (embed 39), so clipping at 5 scales every step by about 0.14 and acts as gradient normalisation, not just spike protection.
  This is a transient mismatch with the carried-over v (built from unclipped gradients), so the effective step is smaller for a while. I didn't loosen the clip, because a 7x jump in gradient scale against the adapted v could cause step spikes. Revisit if val looks slow.
## Decision 3 (update 214): kept
- Loss 2.56 (predicted <2.8). The gradient norm before clipping is 33-61, dominated by embed (55), and clip_coef ranges 0.08-0.15, so embed noise modulates every group's gradient scale.
  Removing the clip abruptly would inflate steps up to 10x until v adapts (tau about 1000). Even a gradual ramp over 1000 updates gives about 2.4x inflation. Decided to live with it unless val disappoints.
## Decision 4 (update 285): kept
- Val 67.5% at update 228 (pred >50%). Loss about 2.0 at 285 (pred 2.0-2.2). Peak LR is reached at 300. Watch val at 456/684 for the trend and gaps.
## Decision 6 (update 428): kept
- The loss plateaued at about 1.65-1.70 over updates 356-427 (pred 1.4-1.5 missed). This is typical at peak LR with augmentation noise.
  clip_coef 0.06-0.12. Waiting for val at 456 before acting.
## Decision 7 (update 499): kept
- Val 80.7% at update 456 (pred >=80%). Loss about 1.5. Next check is val at 684 (pred about 84-86%).
## Decision 8 (update 570): kept the updater; added wake triggers to analysis
- The loss plateaued at about 1.5 (pred 1.3-1.4 missed slightly). Gradient norm noisy (embed 24-91). The next scheduled decision is far away (891).
- Wake triggers: val drops more than 3 points below best, or a 50-update mean loss jumps by more than 0.5.
## Decision 9 (update 891): kept
- Val 86.9% at update 684 (pred 83-86%, beat it). The loss is slowly falling to 1.32. Movement is steady (b0-3 1.1%, b8-11 3%).
  The recipe is working, so I'm keeping it. Pred: val 88-90% by about update 1140.
## Decision 10 (update 1150, woken): kept
- Val 88.71 (update 912), then 85.53 (update 1140); val loss 0.52, then 0.61. The train loss is flat at about 1.27.
  The val noise std is about 1.2 points (788 images), so a 3.2-point drop is about 2 sigma, plausibly peak-LR oscillation.
  The cosine has barely started (factor 0.986). If the next two vals stay at or below 86, I'll lower the peak (scale all LRs by about 0.6).
- Raised the wake threshold to 4 points to cut noise wakes.
## Decision 12 (update 1532): kept
- Val 88.2% at update 1368 (pred 87-89%), so the dip at 1140 was noise. The loss is slowly falling (1.18). Vals: 67.5, 80.7, 86.9, 88.7, 85.5, 88.2.
## Decision 13 (update 1852): kept
- Vals 87.2 and 87.4 (updates 1596, 1824). Plateau at about 87.5-88.5 since epoch 8 while LR factor is above 0.95; the train loss is slowly falling (1.16).
  Belief: noise-limited at peak LR; gains come as the cosine decays (factor 0.79 at epoch 30, 0.5 at epoch 52).
  If val is still at or below 88.5 at epoch 30 with the train loss still falling, consider a faster decay.
## Decision 14 (update 2173): kept
- Val 88.96 at update 2052 (best; val loss best 0.556). The train loss is falling slowly (1.13-1.15). Grad norms are shrinking (clip_coef 0.12-0.25).
## Decision 15 (update 2494): kept
- Val 89.85 at update 2280 (new best, pred 88.5-90). The train loss is flat at about 1.12-1.13. The trend is still upward.
## Decision 16 (update 2814): kept
- Vals 89.97 (update 2508), 89.21 (update 2736). Plateau at about 89-90 with LR factor 0.88. The train loss is slowly falling (1.09-1.10).
## Decision 17 (update 3135): kept, decision point at 3456
- Vals 89.97, 89.21, 88.45 (slight decline). Val loss 0.547, 0.588, 0.590. The train loss is flat at 1.09 (far above the LS floor, so not memorised).
  It could be noise or early overfitting at high LR. At decision 3456 (after vals at 3192 and 3420): if both are at or below 89.2 and val loss is above 0.58,
  switch to a faster decay (e.g. reshape the remaining schedule so LR falls faster now, still ending at 0 at 11400).
## Decision 18 (update 3456): kept
- Vals 90.61 (val loss 0.511, best), 89.59. The faster-decay condition was not met, so the dip at epoch 26 was noise. The train loss is slowly falling (about 1.07).
## Decision 19 (update 3776): kept
- Val 89.34 at update 3648 (val loss 0.578). Range 88.5-90.6 since epoch 20. The train loss is about 1.05. Grad-norm readings range 7-30 (clip occasionally barely binding).
## Decision 20 (update 4097): kept (watching val loss)
- Val 89.21 at update 3876, val loss 0.630 (highest since epoch 10). The train loss is still falling (1.03).
  Val loss rising with flat acc means growing overconfidence/gap. If val loss is above 0.62 and acc at or below 89 at both of the next two vals, accelerate the decay.
## Decision 21 (update 4418): kept
- Vals 91.37 (best, update 4104) and 90.74 (val loss 0.526). Val loss recovered, so the overfit concern is dropped. The train loss is about 1.01. Recipe on track; the expected final is about 92+.
## Decision 22 (update 4738): kept
- Val 90.1 at update 4560 (val loss 0.611). Range 89.2-91.4 since epoch 28. The train loss is about 1.00.
## Decision 23 (update 5059): kept
- Vals 91.37 and 91.88 (best; val loss 0.54). The train loss is about 0.98. Rising with the decay as predicted.
## Decision 24 (update 5379): kept
- Val 91.88 at update 5244 (val loss 0.534, best). The train loss is about 0.97. Grad norms are shrinking (clip_coef 0.3-0.9).
## Decision 25 (update 5700): kept
- Val 93.15 at update 5472 (new best). The train loss is about 0.96. LR factor 0.52.
## Decision 26 (update 6021): kept
- Vals 92.39 and 93.02 (val loss 0.53). The train loss is about 0.95. LR factor 0.48.
## Decision 27 (update 6341): kept
- Val 93.40 at update 6156 (best). The train loss is about 0.94. The clip is sometimes non-binding (coef up to 1.0).
## Decision 28 (update 6662): kept
- Vals 92.77 and 93.27 (val loss 0.52, best). The train loss is about 0.92.
- Tail averaging: cosine to 0 already gives near-converged final weights, and the average can't be validated (val uses live params).
  The expected gain is at most about 0.2 with nonzero risk, so I'm leaning toward not doing it.
## Decision 29 (update 6982): kept
- Val 93.27 at update 6840 (val loss 0.5185, best). The train loss is about 0.91. The clip is often non-binding now.
## Decision 30 (update 7303): kept
- Vals 92.89 and 92.64 (val loss 0.479, best by far). The train loss is about 0.91. Plateau at about 93 within noise.
## Decision 31 (update 7624): kept
- Val 93.78 at update 7524 (best). The train loss is about 0.90.
## Decision 32 (update 7944): kept
- Val 93.91 at update 7752 (best; val loss 0.483). The train loss is about 0.895.
## Decision 33 (update 8265): kept
- Vals 93.27 and 93.65 (val loss about 0.49). The train loss is about 0.89. Plateau at about 93.5.
## Decision 34 (update 8586): kept
- Val 93.78 at update 8436 (val loss 0.4705, best). The train loss is about 0.887. Final decision: no tail averaging, keep v1 to the end.
## Decision 35 (update 8906): kept
- Vals 93.91 and 94.67 (best; val loss 0.47). The train loss is about 0.88.
## Decision 41 (update 10830, final): kept
- Vals at epochs 82-94 were all 93.9-94.7 (last 94.42, val loss 0.473). LR factor 0.007, so the model is converged. Expected final test about 94%.
- Ideas for later: raise or lower layer_decay depending on how far the lower blocks move, and add EMA of the weights near the end.
