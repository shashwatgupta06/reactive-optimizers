# Agent decisions: `mbreact_s0`
The agent's reply at each decision point, as written. `changed` lists the programs it edited.

## Decision 1 (update 71, scheduled; changed: analysis.py, updater.py)

I replaced the starting optimizer with my own AdamW that gives the new classifier head a much higher learning rate. I haven't run the new updater or analysis; the run builds them at the next update, and the next decision will show whether they installed.

1. **Observed:** In the first 71 updates the training loss only fell from 5.45 to 5.26, which is about chance (ln 196 = 5.28). There are no validation results yet. The analysis at update 50 showed a gradient norm of 28.7 and a parameter norm of 269.5. The run takes 14.4 ms per update. I think learning is slow because the head starts at zero and every group's learning rate is held down by the 570-update warmup.

2. **Changed:**
   - **`updater.py`:** a self-contained AdamW using batched `torch._foreach` operations. It carries over the running optimizer's momentum state and step count, so nothing restarts.
     - **Head (`classifier.*`):** peak learning rate 5e-3 (5× the starting value), reached after a short 114-update warmup.
     - **Backbone:** unchanged at 1e-3, with the same 570-update warmup.
     - **Everything else:** cosine decay to 0 at update 11400 for all groups. Weight decay is unchanged: 0.05 on weight matrices and conv kernels, 0 on BatchNorm and biases.
   - **`analysis.py`:**
     - Now runs every 57 updates (half an epoch).
     - Reports gradient and parameter norms separately for the early layers, the late layers and the head, plus how much the weights moved since the last report.
     - Also reports each group's current learning rate and the validation results.
   - **`NOTES.md`:** my plans and predictions.

3. **Prediction:** training loss below about 4.8 by update ~200 and below about 4.0 by update 570. The first validation (epoch 2, around update 228) should be well above chance. I'm wrong if the loss stalls or spikes just after the head's learning rate jumps; that would mean 5e-3 is too high for the head. Once warmup ends, I'll consider raising the backbone learning rate if training loss lags. Cars is a fine-grained task far from ImageNet, so the backbone probably needs to adapt a lot. I'll add regularisation if training loss falls well below validation loss, since 37 images per class makes overfitting likely.

## Decision 2 (update 142, scheduled; changed: nothing)

1. **Observed:** Training loss fell from 5.26 at update 71 to about 3.69 by update 141, with the steepest drop as the head reached its full learning rate (around update 114). I had predicted below 4.8 by update 200, so it is well ahead. There was no instability when the head's learning rate jumped. In the report at update 114, the early layers' gradient norm was 42 against 10 for the late layers, but Adam scales each parameter's step by its own gradient size, so this doesn't need action. Time per update is 14.2 ms, about the same as the starting program's 14.4 ms. There is no validation result yet.

2. **Kept:** both programs unchanged. The backbone learning rate is still warming up (about 2.5e-4 of its 1e-3 peak at this point), so it's too early to judge whether that peak is right. I added this decision's observations to `NOTES.md`.

3. **Prediction:**
   - **Validation:** well above chance at epoch 2 (around update 228), likely more than 20% accuracy.
   - **Training loss:** below about 3.0 by update 570, when the backbone warmup ends.

   I'm wrong if validation is near chance, or if training loss stalls above 3.5 through update 300; either would mean the backbone needs a higher learning rate.

## Decision 3 (update 214, scheduled; changed: nothing)

1. **Observed:** Training loss fell from 3.69 at update 141 to about 3.12 at update 213 (ln 196 = 5.28 is chance). The report at update 171 shows how much the weights moved over half an epoch:
   - **Head:** changed by 52% of its size. It is still learning fast at a learning rate of 5e-3.
   - **Backbone conv weights:** changed by about 4.5% (early and late layers alike).
   - **BatchNorm parameters:** changed by about 0.1%.

   The backbone learning rate was 3e-4 at that point and still rising. There is still no validation result. Time per update is 14.2 ms, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The backbone has not yet reached its 1e-3 peak (at update 570), and I can't judge that peak without validation accuracy. I recorded this decision in `NOTES.md`.

3. **Prediction:**
   - **Validation:** clearly above chance at epoch 2 (around update 228), likely 20–35% accuracy.
   - **Training loss:** below about 2.8 by update 350.

   I'm wrong if validation accuracy is below 15%, or if training loss stays above 3.0 at update 350. Either would mean the backbone needs a higher learning rate, which I would then raise.

## Decision 4 (update 285, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 35.66% accuracy at epoch 2 (update 228), with validation loss 2.65. That is at the top of my predicted 20–35% range.
   - **Training loss:** fell from 3.1 to 2.6 just after update 228. It then stayed flat around 2.55–2.70 up to update 284, while the backbone learning rate rose from 4e-4 to 5e-4. That flat stretch is below my target of 2.8 by update 350.
   - **Weight movement per half epoch:** the backbone conv weights changed by about 5.5–5.8%, rising with the learning rate. The head changed by 32%, down from 52%, so it is settling.
   - **Speed:** 14.1 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The backbone learning rate is only half way to its 1e-3 peak (reached at update 570). There is one validation point so far, and it is good. I'll judge the backbone's peak learning rate and weight decay on the epoch 4 and 6 validation results. I recorded this decision in `NOTES.md`.

3. **Prediction:**
   - **Validation:** about 50% or more at epoch 4 (around update 456), and about 60% at epoch 6 (around update 684).
   - **Training loss:** it should start falling again soon, to below 2.3 by update 456.

   I'm wrong if validation accuracy at epoch 4 is under 45%. That would point to a higher backbone learning rate. If instead validation loss rises while training loss keeps falling, that would point to more weight decay.

## Decision 5 (update 356, scheduled; changed: nothing)

1. **Observed:**
   - **Training loss:** hovered around 2.65–2.85 for most of epoch 2. It then dropped to about 2.24–2.33 as epoch 3 started (around update 342). The previous step came at update 228, from about 3.1 to 2.6. The loss falls when an epoch starts and is flat within it, which is the pattern of a model memorising images it has already seen.
   - **Weight movement per half epoch:** backbone conv weights 6.4–6.7%, rising with the learning rate (6e-4 at update 342). The head is steady at about 31%.
   - **Validation:** no new result since epoch 2 (35.66%).
   - **Speed:** 14.1 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The memorisation pattern suggests overfitting will become the main problem later. Stronger weight decay is my main lever for that. I'll wait for evidence before changing it: the epoch 4 validation, comparing validation loss with training loss. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** about 50% at epoch 4 (around update 456).
   - **Training loss:** below about 2.0 early in epoch 4.

   I'm wrong in one of two ways:
   - Validation loss stays near 2.65 while training loss falls below 2.0. That means overfitting is already setting in, and I'd raise weight decay.
   - Validation accuracy is under 45%. That means the problem is learning too slowly rather than overfitting.

## Decision 6 (update 428, scheduled; changed: nothing)

1. **Observed:**
   - **Training loss:** rose within epoch 3, from about 2.25 at the start to about 2.55 by updates 414–427. Over the same stretch the backbone learning rate ramped from 6e-4 to 7e-4. The rising learning rate seems to be cancelling the progress, which suggests the 1e-3 backbone peak is on the high side rather than too low.
   - **Weight movement per half epoch:** backbone conv weights about 7%. The head is down to 27%.
   - **Validation:** no new result since epoch 2 (35.66%).
   - **Speed:** 14.1 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. I've ruled out raising the backbone learning rate. Whether to lower it or add weight decay depends on the epoch 4 validation (update 456), which is only 28 updates away. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** about 45–50% at epoch 4.
   - **Training loss:** a step down when epoch 4 starts, then roughly flat until the peak learning rate at update 570.

   I'm wrong if validation accuracy at epoch 4 is at or below 40% (barely above 35.66%). That would mean the backbone learning rate is too high and is disrupting the pretrained features, and I would lower its peak.

## Decision 7 (update 499, scheduled; changed: updater.py)

I changed `updater.py`: lower learning rates for the head and the backbone, with the early layers getting half the backbone rate. I haven't run it; the run builds it at the next update. One validation drop is noisy evidence, but the accuracy fall and the loss rise point the same way.

1. **Observed:**
   - **Validation:** fell from 35.66% to 31.09% between epoch 2 and epoch 4, and validation loss rose from 2.65 to 2.95.
   - **Training loss:** over the same period it improved to about 2.2.
   - **Backbone learning rate:** reached 8e-4 at update 456. In my analysis at that point, the backbone weights were changing by about 8% every half epoch, and the head by 28%.

   This is what I said at the last decision would show the backbone learning rate is too high: generalisation gets worse while the model keeps fitting the training images.

2. **Changed (`updater.py`):**

   | Group | Peak learning rate before | Peak learning rate now |
   |---|---|---|
   | Head | 5e-3 | 2.5e-3 |
   | Backbone, later layers (blocks 3–5 and `conv_head`) | 1e-3 | 4e-4 |
   | Backbone, early layers (stem and blocks 0–2) | 1e-3 | 2e-4 |

   - **Why the early layers get less:** their general-purpose features are the most valuable to keep from pretraining.
   - **Warmup:** the backbone's current rate drops from about 8e-4 to about 3.5e-4. The warmup still ends at update 570, followed by the same cosine decay to 0 at update 11400.
   - **Unchanged:** weight decay. Adam's moments carry over from the running updater, so nothing restarts.

3. **Prediction:**
   - **Epoch 6 (update 684):** validation accuracy at or above 40%, with validation loss below 2.6.
   - **Later:** steady gains after that.

   I'm wrong if validation is still at or below 35% at epoch 8 (update 912). In that case I would cut the backbone learning rates further, to about 2e-4 for the later layers and 1e-4 for the early ones.

## Decision 8 (update 570, scheduled; changed: analysis.py)

1. **Observed:**
   - **Training loss:** since the change at update 499 it has fallen steadily within the epoch, from 2.16 to 1.81. Before, it only dropped at the start of each new epoch and rose within the epoch as the learning rate grew. That fits the earlier learning rates having been too high to make steady progress.
   - **Weight movement per half epoch (report at update 570):** early layers about 1.8% (was 7–8%), later layers about 3.6% (was 7–8%), head about 12% (was 23–28%).
   - **Validation:** no new result since epoch 4 (31.09%).
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Changed:**
   - **`updater.py`:** kept as is. Training loss is behaving well, and I need validation results to judge the change.
   - **`analysis.py`:** now asks for an extra decision if either of these happens:
     - validation accuracy is below 38% at epoch 6 or later;
     - validation accuracy falls more than 4 points below its best so far.

   The next scheduled decision isn't until update 891, after the epoch 6 validation (update 684), so without the wake-up a bad result would go unanswered for about 200 updates. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** at least 40% accuracy at epoch 6, with validation loss below 2.6.
   - **Training loss:** below about 1.6 by update 890.

   I'm wrong if the epoch 6 result is below 38%; the new check will wake me early in that case. If so, I'll cut the backbone learning rates further, roughly by half.

## Decision 9 (update 891, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 71.32% accuracy at epoch 6 (update 684), with validation loss 1.228. Epoch 4 was 31.09% and 2.95. This is far above my prediction of at least 40%.
   - **Training loss:** fell from about 1.60 to 1.39 over updates 570–890, about 0.06 per epoch. That number includes label smoothing and augmentation, while validation loss is plain cross-entropy on centre crops. So validation loss sitting below training loss doesn't mean anything is wrong, and there's no sign of overfitting yet.
   - **Weight movement per half epoch:** steady at about 1.5% for the early layers, 3.2% for the later layers and 9.5% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The current peak learning rates are 2.5e-3 for the head, 4e-4 for the later backbone layers and 2e-4 for the early layers. They are now at their peak and decay along the cosine schedule. The open question is whether even lower rates would do better. I'll only act on that if validation stalls while the learning rate is still high, for example if it stays flat from epoch 10 to 20. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** about 73–76% at epochs 8 and 10 (updates 912 and 1140).
   - **Training loss:** about 1.25 by update 1200.

   I'm wrong if validation falls below 67% at either point; the analysis program will wake me early if it drops more than 4 points below its best. That would mean the current learning rates are still disrupting the pretrained features, and I would cut them roughly in half again.

## Decision 10 (update 1211, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** accuracy has levelled off: 71.32%, 73.86% and 72.21% at epochs 6, 8 and 10. Validation loss is flat at 1.21–1.23. That is at the bottom of my predicted 73–76% band and well clear of the 67% that would have worried me.
   - **Training loss:** still falling slowly, from 1.39 to about 1.27 over updates 891–1210.
   - **Weight movement per half epoch:** steady at about 1.6% for the early layers, 3.4% for the later layers and 10% for the head.
   - **Learning rates:** almost at peak (2.48e-3 head, 3.97e-4 later layers, 1.98e-4 early layers).
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. With validation loss flat rather than rising, I read this as the limit set by noise at a high learning rate, not overfitting. Gains should come as the cosine schedule lowers the learning rate, which it has barely started to do.
   - **If validation loss starts rising while training loss keeps falling:** that's overfitting, and I'd add weight decay or make the learning rate decay faster.
   - **If validation stays flat through epoch 20 without overfitting:** I'd consider a faster decay so more of the run is spent at a low learning rate.

   I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** about 72–75% at epochs 12–14 (updates 1368 and 1596), with validation loss staying around 1.2.
   - **Training loss:** about 1.2 by update 1530.

   I'm wrong if validation loss rises above about 1.3 while training loss keeps falling. That would mean overfitting has started, and I would raise weight decay.

## Decision 11 (update 1532, scheduled; changed: updater.py)

I cut every learning rate to 0.6× of its current peak; the run picks up the new updater at the next update. The changed code hasn't run yet.

1. **Observed:**
   - **Validation:** 72.08% accuracy at epoch 12, about where it has been since epoch 6. Validation loss rose from 1.21–1.22 at epochs 8 and 10 to 1.307. That reaches the roughly 1.3 threshold I set last time for overfitting.
   - **Training loss:** still falling, from about 1.31 to 1.21.
   - **Weight movement per half epoch:** about 1.8% for the early layers, 3.6% for the later layers and 10.5% for the head. The learning rates are still close to their peaks.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   Rising validation loss, flat accuracy and falling training loss fit the start of overfitting while the learning rate is still high.

2. **Changed (`updater.py`):** all learning rates multiplied by 0.6. The cosine decay to 0 at update 11400 stays the same. Weight decay is unchanged. I recorded this in `NOTES.md`.

   | Group | Peak before | Peak now |
   |---|---|---|
   | Head | 2.5e-3 | 1.5e-3 |
   | Later backbone layers | 4e-4 | 2.4e-4 |
   | Early backbone layers | 2e-4 | 1.2e-4 |

   - **Why the learning rate:** the last cut (at update 499) took validation from 31% to 71%. Weight decay is too weak to matter here: 0.05 × 4e-4 is only 2e-5 per step. On conv layers followed by BatchNorm it mostly changes the effective step size rather than regularising.

3. **Prediction:**
   - **Validation loss:** back to about 1.2 or lower by epochs 16–18 (updates 1824–2052).
   - **Validation accuracy:** about 74–76% by then.
   - **Training loss:** a small immediate dip, then a slower decline.

   I'm wrong if accuracy falls below 71% or validation loss keeps climbing past about 1.35. In that case I would turn to regularising the parts not followed by BatchNorm: the classifier head, `conv_head` and the squeeze-and-excitation layers.

## Decision 12 (update 1852, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 77.03% at epoch 14 and 76.27% at epoch 16, up from 72.08%. Validation loss fell from 1.307 to 1.115–1.118. My prediction was validation loss of about 1.2 or lower and accuracy of 74–76% by epochs 16–18, so both came true early.
   - **Training loss:** fell from about 1.18 to 1.10.
   - **Weight movement per half epoch:** about halved: early layers about 0.9%, later layers about 1.9%, head about 5.7%.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   Both learning-rate cuts have paid off. Some of the gain may just be the noise reduction the cosine schedule would have delivered later anyway. Still, lower peaks also keep the weights closer to the pretrained ones, which suits a dataset this small.

2. **Kept:** both programs unchanged. Validation only just moved to a new level, so I need to see whether it keeps climbing at these rates or levels off again. If accuracy flattens and validation loss rises above about 1.2 by epochs 20–24, I'll either cut the learning rates again (×0.6) or reshape the cosine to decay sooner. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 76–79% at epochs 18–20 (updates 2052 and 2280), with validation loss staying at or below about 1.15.
   - **Training loss:** about 1.05 by update 2170.

   I'm wrong if validation loss rises above 1.2 while accuracy stays flat or drops. That would mean overfitting has returned at the current rates.

## Decision 13 (update 2173, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 77.54% at epoch 18, the best so far, after 77.03% and 76.27% at epochs 14 and 16. Validation loss ticked up from 1.118 to 1.142, but that is within noise and below the 1.2 I set as the overfitting threshold.
   - **Training loss:** has levelled off at about 1.08–1.09 for the last three epochs. That looks like the limit set by noise at the current learning rate rather than a sign that something is wrong.
   - **Weight movement per half epoch:** steady at about 1.0% for the early layers, 2.0% for the later layers and 6% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. Accuracy is still creeping up, validation loss hasn't crossed my threshold, and further gains should come as the cosine schedule lowers the learning rate. The trigger for another cut stays the same. If validation loss rises above 1.2 while accuracy stays flat, I'll cut the learning rates by 0.6× again or make the cosine decay sooner. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 77–79% at epochs 20 and 22 (updates 2280 and 2508), with validation loss at or below about 1.17.
   - **Training loss:** about 1.06 by update 2490.

   I'm wrong if validation loss rises above 1.2 while accuracy stays at or below 77%. That would mean overfitting has returned.

## Decision 14 (update 2494, scheduled; changed: updater.py)

I cut all learning rates by another 0.6× in `updater.py`; the run picks it up at the next update. The changed code hasn't run yet.

1. **Observed:**
   - **Validation:** levelled off: 77.03%, 76.27%, 77.54% and 77.41% at epochs 14, 16, 18 and 20. Validation loss is steady at about 1.14. That's in my predicted range but on the low edge (I predicted 77–79%).
   - **Training loss:** flat at about 1.07–1.08 for three epochs, so the model is stuck at the noise floor for the current learning rate.
   - **Weight movement per half epoch:** steady at about 1.1% for the early layers, 2.1% for the later layers and 6.2% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

   On the current cosine, the learning rate stays above half its peak until about update 6000, roughly 30 more epochs spent on this plateau.

2. **Changed (`updater.py`):** all learning rates multiplied by 0.6 again, so they are now 0.36× the levels set at update 499. The cosine decay to 0 at update 11400 stays the same. I recorded this in `NOTES.md`, along with an idea for the end of the run: average the weights over the last stretch, and swap the average in around update 11250 so the remaining updates refresh the BatchNorm statistics.

   | Group | Peak before | Peak now |
   |---|---|---|
   | Head | 1.5e-3 | 9e-4 |
   | Later backbone layers | 2.4e-4 | 1.44e-4 |
   | Early backbone layers | 1.2e-4 | 7.2e-5 |

   - **Why another cut:** both earlier cuts (at updates 499 and 1532) gave an immediate 4–5 point jump in validation accuracy without causing overfitting.
   - **The risk:** part of that gain is just noise reduction the cosine schedule would deliver later anyway, and a learning rate that is too low could let the model memorise the training images.

3. **Prediction:**
   - **Validation:** at least 78.5% at epochs 24–26 (updates 2736–2964), with validation loss at or below 1.12.
   - **Training loss:** down to about 1.0.

   I'm wrong if training loss drops quickly while validation stays at or below 77.5% and validation loss rises above 1.2. That would mean the learning rate is now too low and the model is memorising, and I would partly undo the cut (×1.4).

## Decision 15 (update 2814, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 79.57% at epoch 24, the best so far and above my prediction of at least 78.5%. Validation loss fell to 1.086, also within my prediction of at or below 1.12. The epoch 22 reading (77.03%) was taken only 14 updates after the cut, so it doesn't measure the change.
   - **Training loss:** fell from about 1.075 to 1.03–1.04, short of the 1.0 I predicted, and is flattening again.
   - **Weight movement per half epoch:** dropped to about 0.6% for the early layers, 1.2% for the later layers and 3.8% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   All three learning-rate cuts have now each been followed by a 2–5 point step up in validation accuracy.

2. **Kept:** both programs unchanged. Validation just reached a new best, and I need a few more points to see whether it keeps climbing at these rates. If it plateaus again by epochs 28–32, I'll either cut the learning rates by about 0.7× or make the cosine decay sooner. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 79–81% at epochs 26–28 (updates 2964 and 3192), with validation loss at or below about 1.09.
   - **Training loss:** about 1.02 by update 3135.

   I'm wrong if validation falls back to 77.5% or below, or validation loss rises above 1.15 while training loss keeps falling. That would mean the lower rates are starting to let the model memorise the training images.

## Decision 16 (update 3135, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 78.68% at epoch 26, with validation loss 1.136. Epoch 24 was 79.57% and 1.086. The accuracy is inside my predicted 79–81% band only within noise (one image is about 0.13 points). Validation loss is above the 1.09 I expected but below the 1.15 I set as the warning level.
   - **Training loss:** about 1.02, on target for my 1.02 by update 3135, but it has been flat for most of this stretch.
   - **Weight movement per half epoch:** steady at about 0.65% for the early layers, 1.25% for the later layers and 3.9% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. One slightly lower validation result can't separate a new plateau from noise. I'll decide at the next decision, which will have results through epoch 28. If the mean of the validation results from epoch 26 onward is at or below 79.5%, I'll either cut the learning rates by about 0.7× or make the cosine decay sooner. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 79–80% at epoch 28 (update 3192), with validation loss about 1.1.
   - **Training loss:** about 1.01 by update 3456.

   I'm wrong if validation falls to 78% or below, or validation loss rises above 1.15. That would mean the model has stalled or is beginning to overfit at these rates, and I'd move straight to another cut.

## Decision 17 (update 3456, scheduled; changed: updater.py)

I cut all learning rates by another 0.7× in `updater.py`; the run picks it up at the next update. The changed code hasn't run yet.

1. **Observed:**
   - **Validation:** 79.31% at epoch 28 and 78.93% at epoch 30. With epoch 26's 78.68%, that averages 78.97%, below the 79.5% I set as the plateau threshold.
   - **Validation loss:** crept up from 1.086 at epoch 24 to 1.136, 1.164 and 1.16.
   - **Training loss:** flat at about 1.01–1.03.
   - **Weight movement per half epoch:** steady at about 0.65–0.7% for the early layers, 1.3% for the later layers and 4% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   I had predicted 79–80% at epoch 28 with validation loss about 1.1. Accuracy fell in that range but validation loss was higher. It stayed just below the 1.2 I set as the overfitting threshold.

2. **Changed (`updater.py`):** all learning rates multiplied by 0.7, the action I committed to last time if the plateau rule triggered. The cosine decay to 0 at update 11400 stays the same. I recorded this in `NOTES.md`.

   | Group | Peak before | Peak now |
   |---|---|---|
   | Head | 9e-4 | 6.3e-4 |
   | Later backbone layers | 1.44e-4 | 1.0e-4 |
   | Early backbone layers | 7.2e-5 | 5e-5 |

   - **Why:** each of the three earlier cuts was followed by a 2–5 point step up.
   - **Still to do:** I still plan to average the weights over the last stretch of training and swap the average in near the end. I'll build that by about update 8000.

3. **Prediction:**
   - **Validation:** about 80% at epochs 32–34 (updates 3648 and 3876), with validation loss back to 1.12 or below.
   - **Training loss:** a small dip to about 1.0.

   I'm wrong if validation stays at or below 78.5% with validation loss above 1.2. That would mean lowering the learning rate no longer helps, and the remaining problem is overfitting rather than noise.

## Decision 18 (update 3776, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 80.08% at epoch 32, the best so far, with validation loss back down to 1.122 from about 1.16. My prediction was about 80% and validation loss at or below 1.12; this matches within noise.
   - **Training loss:** dipped from about 1.02 to about 0.99–1.01.
   - **Weight movement per half epoch:** down to about 0.45% for the early layers, 0.85–0.9% for the later layers and 2.8% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   This is the fourth learning-rate cut in a row that was followed by a step up in validation.

2. **Kept:** both programs unchanged. Validation just set a new best with lower loss, and I need the next couple of results to see whether it keeps climbing. I recorded a concrete plan in `NOTES.md` for the weight averaging I'll add around update 7500–7900:
   - Average the weights uniformly from about update 8000.
   - Swap the average into the model at about update 11100, where the learning rate is under 1% of its peak. The remaining ~300 updates then refresh the BatchNorm statistics for the averaged weights.
   - The epoch 98 validation (update 11172) will show whether the swap helped. I'll keep the pre-swap weights so I can revert if it hurts.

3. **Prediction:**
   - **Validation:** about 79.5–81% at epochs 34–36 (updates 3876 and 4104), with validation loss at or below about 1.13.
   - **Training loss:** about 0.99.

   I'm wrong if validation falls back to 78.5% or below with validation loss above 1.18. That would mean the gain from the cut was only a temporary noise reduction and overfitting is the limit. I would then look at regularisation instead of further cuts.

## Decision 19 (update 4097, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 79.31% at epoch 34, with validation loss 1.131. Epoch 32 was 80.08% and 1.122, so this is within noise and inside my predicted 79.5–81% band only within noise.
   - **Training loss:** flat at about 0.99, as predicted.
   - **Weight movement per half epoch:** steady at about 0.43% for the early layers, 0.85% for the later layers and 2.8% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

   The four learning-rate cuts were followed by gains of about 4.5, 5, 2 and 1 points, so each further cut is worth less.

2. **Kept:** both programs unchanged. The learning rates are now low (about 7.7e-5 for the later backbone layers), and the cosine schedule will lower them steadily from here. Another manual cut would mostly just bring that decay forward. I'd only cut again if validation drops with validation loss above about 1.18. I recorded this in `NOTES.md`.

   I also ruled out sharpness-aware minimisation (SAM), which is known to help fine-tuning on fine-grained tasks. It needs an extra forward and backward pass per update, and the updater only receives gradients.

   The plan to average weights over the last part of training and swap the average in near the end is unchanged. I'll implement it around update 7500–7900.

3. **Prediction:**
   - **Validation:** 79–81% at epochs 36–38 (updates 4104 and 4332), with validation loss at or below about 1.14.
   - **Training loss:** about 0.98–0.99.

   I'm wrong if validation falls to 78.5% or below with validation loss above 1.18. That would mean slow overfitting is taking over, and I'd consider another cut or a regulariser for the head.

## Decision 20 (update 4418, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 79.57% at epoch 36 and 79.82% at epoch 38, with validation loss 1.14 and 1.133. That is within my predicted 79–81% band and at or just under the 1.14 loss limit.
   - **Training loss:** flat at about 0.98–0.99, as predicted.
   - **Weight movement per half epoch:** steady at about 0.45% for the early layers, 0.85% for the later layers and 2.8% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

   There's no sign of overfitting drift: validation loss has been flat since epoch 32.

2. **Kept:** both programs unchanged. The model is stable at about 79.5–80%. Further manual cuts gave less and less each time, and the cosine schedule is now lowering the learning rate on its own (later backbone layers at about 7.3e-5, falling). My next planned change is the weight averaging over the last part of training, around update 7500–7900. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 79.5–81% through epochs 40–42 (updates 4560 and 4788), with validation loss at or below about 1.14.
   - **Training loss:** drifting down slowly to about 0.97 by update 4738.

   I'm wrong if validation falls to 78.5% or below with validation loss above 1.18. That would point to slow overfitting, and I'd consider another learning-rate cut or a regulariser for the head.

## Decision 21 (update 4738, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.09% at epoch 40, the best so far, with validation loss 1.135. That's at the top of my predicted 79.5–81% band, and the loss is within my 1.14 limit.
   - **Training loss:** edging down from about 0.985 to 0.975–0.98, close to the 0.97 I predicted.
   - **Weight movement per half epoch:** steady at about 0.4% for the early layers, 0.8% for the later layers and 2.7% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   One validation point is noisy (each image is about 0.13 points), but the trend from epoch 34 to 40 (79.31, 79.57, 79.82, 81.09) is upward with flat loss. That fits steady gains as the cosine schedule lowers the learning rate.

2. **Kept:** both programs unchanged. The run is improving with no sign of overfitting, so there's nothing to correct. My next planned change is still the weight averaging over the last part of training, around update 7500–7900. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80–82% at epochs 42–44 (updates 4788 and 5016), with validation loss at or below about 1.14.
   - **Training loss:** about 0.97 by update 5059.

   I'm wrong if validation falls back to 79% or below with validation loss above 1.16. That would mean 81.09% was a noise spike or that slow overfitting has started.

## Decision 22 (update 5059, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 80.84% at epoch 42 and 80.58% at epoch 44, with validation loss 1.131 and 1.143. Accuracy is in my predicted 80–82% band. The loss is about at the 1.14 limit and far below the 1.16 I set as the warning level.
   - **Training loss:** about 0.975, flat, slightly short of the 0.97 I predicted.
   - **Weight movement per half epoch:** steady at about 0.4% for the early layers, 0.8% for the later layers and 2.7% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   Since epoch 40, validation has settled at about 80.5–81%.

2. **Kept:** both programs unchanged. The run is stable with no overfitting drift, and the cosine schedule keeps lowering the learning rate (later backbone layers at about 6.4e-5, falling). My next planned change is still the weight averaging over the last part of training, around update 7500–7900. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80–82% at epochs 46–48 (updates 5244 and 5472), with validation loss at or below about 1.15.
   - **Training loss:** about 0.97 by update 5379.

   I'm wrong if validation falls to 79% or below with validation loss above 1.17. That would mean slow overfitting has set in, and I'd consider making the learning rate decay sooner.

## Decision 23 (update 5379, scheduled; changed: analysis.py, updater.py)

I added the weight averaging to `updater.py` and a matching report and wake-up to `analysis.py`. I haven't run any of it. The run builds them at the next update, and the averaging code itself doesn't execute until update 8000.

1. **Observed:**
   - **Validation:** 80.96% at epoch 46, with validation loss 1.136. Since epoch 40 it has been between 80.6% and 81.1%, with loss steady at 1.13–1.14. That's in my predicted 80–82% band, and the loss is within the 1.15 limit.
   - **Training loss:** about 0.97–0.98, as predicted.
   - **Weight movement per half epoch:** steady at about 0.4% for the early layers, 0.75–0.8% for the later layers and 2.6% for the head.
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Changed:** the learning rates and schedule are unchanged. I installed the averaging now, well before it acts, so there are decisions left to fix a problem.
   - **`updater.py`:** the weight averaging I had planned, inactive until update 8000.
     - From update 8000 to 11099 it keeps a running uniform average of all weights.
     - At update 11100 it saves the current weights and copies the average into the model. The learning rate there is under 1% of its peak, so the remaining ~300 updates barely move the weights but do refresh the BatchNorm running statistics for the averaged weights.
     - A switch (`REVERT_SWAP`) lets me restore the saved weights at a later decision if the average turns out worse.
     - The averaging state is saved and restored, so a rollback won't lose it.
   - **`analysis.py`:** reports how many updates have been averaged and whether the swap has happened. It also asks for an extra decision at the first validation after the swap (epoch 98, update 11172).

   I recorded the plan in `NOTES.md`.

3. **Prediction:**
   - **Until update 8000:** behaviour exactly as now. Validation about 80.5–82% at epochs 48–50, with training loss about 0.97.
   - **After update 8000:** no errors or rollbacks, and the report shows the average count climbing.
   - **After the swap at update 11100:** validation at epoch 98 is the same as or a few tenths higher than epoch 96.

   I'm wrong if validation drops by more than 0.8 points after the swap. In that case I'll set `REVERT_SWAP` before about update 11300 to restore the pre-swap weights.

## Decision 24 (update 5700, scheduled; changed: nothing)

1. **Observed:**
   - **New version:** the averaging version installed without errors and runs at the same speed (14.2 ms per update, against 14.4 ms for the starting program). As expected before update 8000, the reports show nothing averaged yet and no swap.
   - **Validation:** 80.2% at epoch 48, with validation loss 1.149. Since epoch 40 it has stayed between 80.2% and 81.1%. That's inside my predicted 80.5–82% band only within noise. The loss is just above the 1.14 I had in mind but well below the 1.17 I set as the warning level.
   - **Training loss:** about 0.975, as predicted.
   - **Weight movement per half epoch:** steady at about 0.4% for the early layers, 0.75% for the later layers and 2.5% for the head.

2. **Kept:** both programs unchanged. The run is stable and the cosine schedule keeps lowering the learning rate (later backbone layers at about 5.5e-5). The averaging will start on its own at update 8000. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80–82% at epochs 50–52 (updates 5700 and 5928), with validation loss at or below about 1.15.
   - **Training loss:** about 0.97 by update 6021.

   I'm wrong if validation falls to 79% or below with validation loss above 1.17. That would point to slow overfitting.

## Decision 25 (update 6021, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.98% at epoch 50, the best so far, and 80.33% at epoch 52. Both are in my predicted 80–82% band, and the swing between them is within noise.
   - **Validation loss:** 1.147 and 1.155. It's creeping up very slowly (from about 1.13 around epoch 40) and is just above my 1.15 limit, but well below the 1.17 warning level.
   - **Training loss:** dipped to about 0.96–0.97, as predicted.
   - **Weight movement per half epoch:** down to about 0.35% for the early layers, 0.65% for the later layers and 2.2% for the head.
   - **Averaging:** still waiting as designed (nothing averaged yet, no swap).
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. Accuracy is holding at about 80–82%. The rise in validation loss is small and mostly comes from the model becoming more confident, not from accuracy falling. The learning rate keeps decaying (later backbone layers at about 5e-5), and the averaging starts on its own at update 8000. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80–82% at epochs 54–56 (updates 6156 and 6384), with validation loss at or below about 1.17.
   - **Training loss:** about 0.96 by update 6341.

   I'm wrong if validation falls to 79% or below with validation loss above 1.18. That would point to real overfitting, and I'd consider making the learning rate decay sooner.

## Decision 26 (update 6341, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.73% at epoch 54, with validation loss 1.145. That's in my predicted 80–82% band and within the 1.17 loss limit. The last four results (80.2%, 81.98%, 80.33%, 81.73%) average about 81%.
   - **Training loss:** about 0.96, as predicted.
   - **Weight movement per half epoch:** down to about 0.3% for the early layers, 0.6% for the later layers and 2% for the head.
   - **Averaging:** still waiting as designed (nothing averaged yet, no swap).
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is steady at about 81% with no sign of overfitting, and the learning rate keeps decaying (later backbone layers at about 4.5e-5). The averaging starts on its own at update 8000. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80.5–82.5% at epochs 56–58 (updates 6384 and 6612), with validation loss at or below about 1.16.
   - **Training loss:** about 0.955 by update 6662.

   I'm wrong if validation falls to 79.5% or below with validation loss above 1.18. That would mean overfitting has set in late in the run.

## Decision 27 (update 6662, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 80.71% at epoch 56 and 80.84% at epoch 58, with validation loss 1.145 and 1.158. Accuracy is in my predicted 80.5–82.5% band. The loss is just above the 1.16 limit I gave in my last prediction (rounded, 1.158 ≈ 1.16) but within noise of it, and below the 1.18 warning level. Since epoch 48, validation has stayed at about 80.2–82%.
   - **Training loss:** drifting down to about 0.95–0.96, as predicted.
   - **Weight movement per half epoch:** down to about 0.3% for the early layers, 0.55% for the later layers and 1.9% for the head.
   - **Averaging:** still waiting as designed (nothing averaged yet, no swap).
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is stable with no overfitting drift. The learning rate keeps decaying (later backbone layers at about 4.1e-5), and the averaging starts on its own at update 8000. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80.5–82.5% at epochs 60–62 (updates 6840 and 7068), with validation loss at or below about 1.17.
   - **Training loss:** about 0.95 by update 6982.

   I'm wrong if validation falls to 79.5% or below with validation loss above 1.18. That would mean overfitting has set in late in the run.

## Decision 28 (update 6982, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 80.58% at epoch 60, in my predicted 80.5–82.5% band, at the low edge.
   - **Validation loss:** improved to 1.122, from 1.145–1.158 over epochs 54–58.
   - **Training loss:** about 0.95, as predicted.
   - **Weight movement per half epoch:** down to about 0.25% for the early layers, 0.5% for the later layers and 1.7% for the head.
   - **Averaging:** still waiting as designed (nothing averaged yet, no swap).
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

   Since epoch 48, validation has stayed at about 80.2–82%, with no sign of overfitting.

2. **Kept:** both programs unchanged. The run is stable, the learning rate keeps decaying (later backbone layers at about 3.6e-5), and the averaging starts on its own at update 8000. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80.5–82.5% at epochs 62–64 (updates 7068 and 7296), with validation loss at or below about 1.16.
   - **Training loss:** about 0.945 by update 7303.

   I'm wrong if validation falls to 79.5% or below with validation loss above 1.18. That would mean overfitting has set in late in the run.

## Decision 29 (update 7303, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.6% at epoch 62 and 80.71% at epoch 64, with validation loss 1.146 and 1.148. Both are in my predicted 80.5–82.5% band and within the 1.16 loss limit. Since epoch 48, validation has stayed at about 80.2–82%.
   - **Training loss:** about 0.95. That's a little above the 0.945 I predicted, but within noise.
   - **Weight movement per half epoch:** down to about 0.22% for the early layers, 0.42% for the later layers and 1.5% for the head.
   - **Averaging:** still waiting as designed (nothing averaged yet, no swap).
   - **Speed:** 14.2 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is stable with no overfitting, and the learning rate keeps decaying (later backbone layers at about 3.2e-5). The weight averaging starts on its own at update 8000, about 700 updates from now. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80.5–82.5% at epochs 66–68 (updates 7524 and 7752), with validation loss at or below about 1.16.
   - **Training loss:** about 0.945 by update 7624.
   - **After update 8000:** no errors or rollbacks when the averaging begins, and the next report shows the average count climbing.

   I'm wrong if validation falls to 79.5% or below with validation loss above 1.18. That would mean overfitting has set in late in the run.

## Decision 30 (update 7624, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.09% at epoch 66, in my predicted 80.5–82.5% band.
   - **Validation loss:** 1.161, just above the 1.16 limit I predicted but within noise of it, and below the 1.18 warning level. It has hovered at about 1.12–1.16 since epoch 40.
   - **Training loss:** about 0.945–0.95, as predicted.
   - **Weight movement per half epoch:** down to about 0.2% for the early layers, 0.37% for the later layers and 1.3% for the head.
   - **Averaging:** still waiting as designed (nothing averaged yet, no swap).
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

   Since epoch 48, validation has stayed at about 80.2–82%.

2. **Kept:** both programs unchanged. The run is stable with no overfitting, and the learning rate keeps decaying (later backbone layers at about 2.8e-5). The averaging starts on its own at update 8000, between the next decision (update 7944) and the one after (about update 8265). The later one will show whether it's running. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80.5–82.5% at epoch 68 (update 7752), with validation loss at or below about 1.17.
   - **Training loss:** about 0.945 by update 7944.
   - **At the decision around update 8265:** no errors or rollbacks from the averaging, and the report shows the average count climbing.

   I'm wrong if validation falls to 79.5% or below with validation loss above 1.18. That would mean overfitting has set in late in the run.

## Decision 31 (update 7944, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.73% at epoch 68, with validation loss 1.147. That's in my predicted 80.5–82.5% band and within the 1.17 loss limit. Since epoch 48, validation has stayed at about 80.2–82%.
   - **Training loss:** about 0.945, as predicted.
   - **Weight movement per half epoch:** down to about 0.18% for the early layers, 0.33% for the later layers and 1.1% for the head.
   - **Averaging:** still waiting as designed (nothing averaged yet, no swap).
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is stable with no overfitting, and the learning rate keeps decaying (later backbone layers at about 2.4e-5). The weight averaging starts on its own at update 8000, 56 updates from now. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 80.5–82.5% at epoch 70 (update 7980), with validation loss at or below about 1.17.
   - **At the next decision (update 8265):**
     - The averaging will have run with no errors or rollbacks.
     - The report will show about 265 updates averaged.
     - Time per update will stay at about 14.3 ms; the averaging adds only a few small batched operations per update.

   I'm wrong if the report still shows nothing averaged, or if an error or rollback is reported. Either would mean the averaging code is broken, and I'd fix it at that decision.

## Decision 32 (update 8265, scheduled; changed: nothing)

1. **Observed:**
   - **Averaging:** working as designed. The report shows 265 updates averaged at update 8265, exactly the predicted count. There were no errors or rollbacks, and speed is unchanged at 14.3 ms per update (14.4 ms for the starting program).
   - **Validation:** 82.11% at epoch 70 and 81.98% at epoch 72, the best pair of results so far. Validation loss is 1.146 and 1.157.
   - **Training loss:** about 0.944.
   - **Weight movement per half epoch:** down to about 0.15% for the early layers, 0.28% for the later layers and 1% for the head.

2. **Kept:** both programs unchanged. The run is stable, the averaging is accumulating, and the learning rate keeps decaying (later backbone layers at about 2e-5). Nothing needs to change until the swap at update 11100. The analysis program will then wake me at the first validation after the swap (epoch 98, update 11172) to compare it with epoch 96. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81–83% at epochs 74–76 (updates 8436 and 8664), with validation loss at or below about 1.17.
   - **Training loss:** about 0.94 by update 8586.
   - **Averaging:** the count climbs by about 321 per decision with no errors.

   I'm wrong if validation falls to 80% or below with validation loss above 1.18, or if any error or rollback appears from the averaging code.

## Decision 33 (update 8586, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 82.11% at epoch 74, equal to the best result so far and in my predicted 81–83% band.
   - **Validation loss:** 1.135, slightly better than 1.146–1.157 at epochs 70–72.
   - **Training loss:** about 0.94, as predicted.
   - **Averaging:** counting up exactly as expected (550 updates averaged at update 8550). No errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.13% for the early layers, 0.24% for the later layers and 0.85% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is stable at about 82%, the learning rate keeps decaying (later backbone layers at about 1.6e-5), and the averaging is accumulating. The next planned event is the swap at update 11100. The analysis program will wake me at the first validation after it (epoch 98, update 11172) to compare it with epoch 96. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81–83% at epochs 76–78 (updates 8664 and 8892), with validation loss at or below about 1.16.
   - **Training loss:** about 0.94 by update 8906.
   - **Averaging:** the count reaches about 870 with no errors.

   I'm wrong if validation falls to 80% or below with validation loss above 1.18, or if the averaging code reports an error or rollback.

## Decision 34 (update 8906, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.98% at epoch 76 and 81.6% at epoch 78, with validation loss 1.156 and 1.147. Both are in my predicted 81–83% band and within the 1.16 loss limit. Since epoch 68, validation has stayed at about 81.6–82.1%.
   - **Training loss:** about 0.94, as predicted.
   - **Averaging:** 892 updates averaged at update 8892, on track for the count I expected, with no errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.1% for the early layers, 0.18% for the later layers and 0.65% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is stable, the learning rate keeps decaying (later backbone layers at about 1.3e-5), and the averaging is accumulating.

   I checked that the plan for checking the swap still works with the remaining decisions:
   - The last scheduled decision is at update 10829.
   - The wake after the swap comes with the epoch 98 validation at update 11172, 343 updates later, which satisfies the 100-update minimum between decisions.
   - If I revert there, about 228 updates remain to refresh the BatchNorm statistics.

   I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81–83% at epochs 80–82 (updates 9120 and 9348), with validation loss at or below about 1.16.
   - **Training loss:** about 0.94 by update 9227.
   - **Averaging:** the count reaches about 1225 with no errors.

   I'm wrong if validation falls to 80% or below with validation loss above 1.18, or if the averaging code reports an error or rollback.

## Decision 35 (update 9227, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 82.49% at epoch 80, the best result so far, with validation loss 1.155. That's in my predicted 81–83% band and within the 1.16 loss limit. Since epoch 68, validation has stayed at about 81.6–82.5%.
   - **Training loss:** about 0.935–0.94, as predicted.
   - **Averaging:** 1177 updates averaged at update 9177, on track for the 1225 I expected by update 9227, with no errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.08% for the early layers, 0.15% for the later layers and 0.5% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is stable and slowly improving, the learning rate keeps decaying (later backbone layers at about 1e-5), and the averaging is accumulating. The swap at update 11100 and the wake to check it at update 11172 are still on schedule. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81.5–83% at epochs 82–84 (updates 9348 and 9576), with validation loss at or below about 1.16.
   - **Training loss:** about 0.935 by update 9548.
   - **Averaging:** the count reaches about 1545 with no errors.

   I'm wrong if validation falls to 80.5% or below with validation loss above 1.18, or if the averaging code reports an error or rollback.

## Decision 36 (update 9548, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 82.49% at epoch 82, equal to the best result so far and in my predicted 81.5–83% band.
   - **Validation loss:** 1.165, just above the 1.16 limit I predicted but below the 1.18 warning level. It has hovered at about 1.13–1.165 since epoch 40.
   - **Training loss:** about 0.935–0.94, as predicted.
   - **Averaging:** 1519 updates averaged at update 9519, on track for the 1545 I expected by update 9548, with no errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.05% for the early layers, 0.1% for the later layers and 0.36% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The run is stable at about 82%, the learning rate is now small (later backbone layers at about 7e-6), and the averaging is accumulating. Four scheduled decisions remain, at about updates 9868, 10188, 10509 and 10829. The swap at 11100 and the wake to check it at 11172 still fit after them. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81.5–83% at epochs 84–86 (updates 9576 and 9804), with validation loss at or below about 1.17.
   - **Training loss:** about 0.935 by update 9868.
   - **Averaging:** the count reaches about 1865 with no errors.

   I'm wrong if validation falls to 80.5% or below with validation loss above 1.19, or if the averaging code reports an error or rollback.

## Decision 37 (update 9868, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.35% at epoch 84 and 81.6% at epoch 86, a little below the 82.49% at epochs 80–82. Both are inside my predicted 81.5–83% band only within noise; each image is worth about 0.13 points, so the gap is about 7–9 images.
   - **Validation loss:** 1.181 at epoch 84, above my 1.17 limit but within my 1.19 warning level. It eased back to 1.166 at epoch 86.
   - **Training loss:** about 0.932–0.94, as predicted.
   - **Averaging:** 1861 updates averaged at update 9861, matching the expected count, with no errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.04% for the early layers, 0.07% for the later layers and 0.25% for the head.
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged. The dip is within the noise of a 788-image validation set, and it came while the weights were barely moving, so it isn't a trend worth acting on. At this stage, gains can only come from the weight average swapped in at update 11100. Three scheduled decisions remain (about 10189, 10509 and 10829), plus the wake after the swap at update 11172. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81–83% at epochs 88–90 (updates 10032 and 10260), with validation loss at or below about 1.18.
   - **Training loss:** about 0.935 by update 10189.
   - **Averaging:** the count reaches about 2189 with no errors.

   I'm wrong if validation falls to 80.5% or below with validation loss above 1.19, or if the averaging code reports an error or rollback.

## Decision 38 (update 10189, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.6% at epoch 88, with validation loss 1.169. That's in my predicted 81–83% band and within the 1.18 loss limit. Since epoch 68, validation has stayed at about 81.35–82.5%.
   - **Training loss:** about 0.93–0.94, as predicted.
   - **Averaging:** 2146 updates averaged at update 10146, matching the expected count, with no errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.03% for the early layers, 0.05% for the later layers and 0.17% for the head.
   - **Speed:** 14.4 ms per update, the same as the starting program.

2. **Kept:** both programs unchanged. The learning rate is now tiny (later backbone layers at about 3.3e-6), and the run will finish close to where it is. Since update 8000 the weights have moved only a few tenths of a percent per half epoch, so the average stays close to the current weights. That makes the swap at update 11100 low-risk, and its likely effect is a small gain from smoothing out the last stretch. Two scheduled decisions remain (about 10509 and 10829), plus the wake after the swap at update 11172. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81–83% at epoch 90 (update 10260), with validation loss at or below about 1.18.
   - **Training loss:** about 0.935 by update 10509.
   - **Averaging:** the count reaches about 2509 with no errors.

   I'm wrong if validation falls to 80.5% or below with validation loss above 1.19, or if the averaging code reports an error or rollback.

## Decision 39 (update 10509, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 82.36% at both epoch 90 and epoch 92, near the best of 82.49%. Validation loss was 1.17 and 1.16. Both are within my predicted 81–83% band and the 1.18 loss limit.
   - **Training loss:** about 0.935, as predicted.
   - **Averaging:** 2488 updates averaged at update 10488, matching the expected count, with no errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.015% for the early layers, 0.03% for the later layers and 0.1% for the head.
   - **Speed:** 14.4 ms per update, the same as the starting program.

2. **Kept:** both programs unchanged. The learning rate is nearly zero (later backbone layers at about 1.8e-6), and the weights have essentially stopped moving. The only remaining planned event is the swap at update 11100. One scheduled decision is left (update 10830). After it, the analysis program wakes me at the epoch 98 validation (update 11172) to compare the post-swap result with epoch 96. I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81.5–83% at epochs 94–96 (updates 10716 and 10944), with validation loss at or below about 1.18.
   - **Averaging:** the count reaches about 2830 by update 10830 with no errors.
   - **After the swap:** validation at epoch 98 is the same as or a few tenths higher than epoch 96.

   I'm wrong if validation drops by more than 0.8 points after the swap. In that case I'll restore the pre-swap weights at the wake decision.

## Decision 40 (update 10830, scheduled; changed: nothing)

1. **Observed:**
   - **Validation:** 81.35% at epoch 94, with validation loss 1.168. That's just under my predicted 81.5–83% band but within noise of it, and within the 1.18 loss limit. Since epoch 68, validation has stayed at about 81.35–82.5%; the mean over epochs 90–94 is 82.0%.
   - **Training loss:** about 0.935.
   - **Averaging:** 2830 updates averaged at update 10830, exactly the expected count, with no errors or rollbacks.
   - **Weight movement per half epoch:** down to about 0.006% for the early layers, 0.01% for the later layers and 0.04% for the head. The learning rate is nearly zero (later backbone layers at about 7e-7).
   - **Speed:** 14.3 ms per update, against 14.4 ms for the starting program.

2. **Kept:** both programs unchanged, and the swap at update 11100 goes ahead.
   - **How the swap works:** the average covers updates 8000–11099, a stretch where the weights moved very little. The ~300 near-zero learning rate updates after the swap will refresh the BatchNorm running statistics for the averaged weights. The pre-swap weights are kept.
   - **Timing correction:** validation results reach the analysis at its next call, not the one where they're measured. So the wake for the epoch 98 result comes at about update 11229, not 11172. That is still more than 100 updates after this decision, and leaves about 171 updates for the BatchNorm statistics if I revert.
   - **Revert rule:** at that decision I'll restore the pre-swap weights only if epoch 98 is more than 1 point below the mean of epochs 90–96 and validation loss has also gone up. Accuracy on 788 images is too noisy to act on alone.

   I recorded this in `NOTES.md`.

3. **Prediction:**
   - **Validation:** 81.5–83% at epoch 96 (update 10944).
   - **After the swap:** epoch 98 validation is the same as or a few tenths above the epoch 90–96 average of about 82%, with validation loss about 1.16 or slightly lower.

   I'm wrong if epoch 98 falls below about 81% with validation loss above about 1.18. That would mean the averaged weights are worse than the final ones, and I'll revert at the wake.

## Decision 41 (update 11229, your analysis asked: first validation after the weight-average swap; changed: nothing)

The run is nearly finished. I kept the averaged weights. Validation sits at about 82%, up from 35.66% at the first check (epoch 2). The averaging made no measurable difference: the result after the swap is within noise of before.

1. **Observed:**
   - **The swap:** happened at update 11100, averaging 3100 sets of weights (updates 8000–11099). The reports show the expected one-off jump in the weights at the swap (1.1% for the head). After it the weights hardly move, and training loss is unchanged at about 0.935.
   - **Validation:**

     | | Accuracy | Validation loss |
     |---|---|---|
     | Epoch 96 (before the swap) | 82.11% | 1.147 |
     | Epoch 98 (after the swap) | 81.73% | 1.147 |
     | Mean of epochs 90–96 | 82.05% | — |

   - **Against my prediction:** I predicted epoch 98 would match or slightly beat that mean. It was 0.32 points below, about 2–3 images out of 788, with identical validation loss. That is within noise, so I can't tell whether the averaging helped or hurt.

2. **Kept:** the averaged weights. My revert rule needed a drop of more than 1 point together with higher validation loss, and neither happened. Reverting on a noise-sized difference would just swap one equivalent set of weights for another. The last ~300 updates at a near-zero learning rate will finish refreshing the BatchNorm statistics for these weights before the test at update 11400. I recorded this in `NOTES.md`.

3. **Prediction:** test accuracy at update 11400 about 81–83%, in line with recent validation, since the weights and BatchNorm statistics are now essentially fixed. There's no further decision to check this against.

**What changed over the run:**
- The changes that mattered were the four learning-rate cuts. They took the head from 5e-3 down to about 6e-4 and the backbone from 1e-3 to about 1e-4, with the early layers at half the backbone rate.
- The first cut lifted validation from 31% to 71%. Each later cut was followed by a further gain of 1–5 points.
- The weight averaging at the end contributed nothing measurable.
