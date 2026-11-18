# Agent decisions: `dreact_s0`
The agent's reply at each decision point, as written. `changed` lists the programs it edited.

## Decision 1 (update 71, scheduled; changed: analysis.py, updater.py)

I replaced the starting optimizer with my own AdamW (v1) and expanded the analysis program. Nothing has run yet: the new code takes over at the next update and must survive its 20-update trial.

1. **Observed:** In 71 updates the training loss went from 5.281 to 5.180, barely moving. The new 196-class head starts at zero, and the warmup is so slow that at update 71 the learning rate is only 12% of its 1e-4 peak (about 1.25e-5). The gradient norm at update 50 was 6.46, almost all of it from the head. There are no validation results yet.

2. **Changed:**
   - **`updater.py`**: a new AdamW, written to process tensors in batches rather than one at a time.
     - **Learning rate by depth:** the last block and the final norm get 2e-4, and each block below gets 0.85× the one above. That puts block 0 at about 3.4e-5 and the patch/position embeddings at about 2.4e-5. Stanford Cars is a narrow task far from DINOv2's training data, so the upper blocks need to adapt more than the lower ones.
     - **Head:** learning rate 2e-3, so the new classifier learns quickly instead of holding the whole network back.
     - **Warmup:** cut from 570 to 300 updates, then cosine decay to 0 at update 11400. The schedule is written in terms of the absolute update number.
     - **Unchanged from the start:** weight decay 0.05 on weight matrices, and none on 1-D tensors, the position embedding or the class token.
     - **Clipping:** the total gradient norm is clipped at 5, which should only catch spikes.
     - **Carry-over:** the existing AdamW momentum and variance estimates are copied over, so nothing restarts from zero.
   - **`analysis.py`**: now reports gradient norms for each section (embeddings, blocks 0–3, 4–7, 8–11, head), how far each section has moved, the gradient norm before clipping, how much clipping scaled it down, the learning-rate multiplier, and validation results. The "how far moved" figure is measured from the weights at update 100, not from the original starting weights.
   - **`NOTES.md`**: records this plan and the predictions below.

3. **Prediction:**
   - The training loss should be below about 4.0 by update 300 and well below 2 by about epoch 5.
   - The first validation (update 228) should be above 50%, and I expect it to reach 85–90% or more later.
   - **What would prove me wrong:** the loss stalling or spiking, or the clipping scaling down often. If that happens, I'll lower the head and top-block learning rates.
   - **Later:** I'll adjust the 0.85 depth factor based on how far the lower blocks move, and consider averaging the weights over the final stretch of training.

## Decision 2 (update 142, scheduled; changed: nothing)

I kept both programs unchanged at this decision.

1. **Observed:** The new updater is working. The training loss fell from 5.18 to 3.42 in 71 updates, against about 0.1 per 71 updates under the starting recipe. The time per update is 123 ms, about the same as the starting program's 125 ms. One surprise: at update 100 the total gradient norm before clipping was 36, mostly from the embeddings (39). So the clip at 5 is scaling every update down by about 0.14, which turns it into gradient normalisation rather than spike protection.

2. **Kept:** Adam mostly ignores a constant scale on the gradients. The only cost is a temporary one: its variance estimates still include the unclipped gradients from the starting optimizer, so the effective step is somewhat smaller for a while. Removing the clip now would suddenly make the gradients about 7× larger than those estimates, which risks sharply oversized steps. The fast drop in loss suggests the smaller step isn't hurting much, so I'm leaving the clip at 5 for now.

3. **Prediction:**
   - The training loss should be below about 2.8 by update 214.
   - The first validation (update 228) should be above 50%.
   - **What would prove me wrong:** validation below about 40%, or the loss flattening above 3. If that happens, I'll raise the clip threshold gradually, which effectively raises the learning rate.

## Decision 3 (update 214, scheduled; changed: nothing)

I kept both programs unchanged again. The first validation result arrives at update 228, just after this decision.

1. **Observed:**
   - The training loss is 2.56 at updates 207–213, ahead of my prediction of about 2.8 by update 214.
   - The total gradient norm before clipping rose from 33 to 61 and is dominated by the patch/position-embedding section (55 at update 200). So the clip at 5 scales every update down by a factor that swings between 0.08 and 0.15.
   - The new head has moved a lot from where it was at update 100. The blocks have moved only 0.2–0.7% of their size, and their lower layers least, as the depth-scaled learning rates intend.
   - The time per update is 123 ms.

2. **Kept:** Because one section dominates the gradient norm, its noise changes how much every other section's gradient gets scaled down. That is a mild cost; Adam handles a steady scale on the gradients well.

   Removing the clip would make the gradients up to about 10× larger than what Adam's variance estimates were built on. Those estimates adapt over roughly 1,000 updates, so steps would be oversized for a long time. Even raising the clip gradually over 1,000 updates would still inflate steps by about 2.4×. Training is going well, so the risk isn't justified.

3. **Prediction:**
   - Validation at update 228 should be at least 50%.
   - The training loss should be about 2.0–2.2 by update 285.
   - **What would prove me wrong:** validation below 40%, or the loss flattening. Either would make me revisit the clip and learning rates.

## Decision 4 (update 285, scheduled; changed: nothing)

I kept both programs unchanged; the first validation came in above my prediction.

1. **Observed:**
   - The first validation, at update 228 (epoch 2), was 67.5% accuracy with loss 1.61. I had predicted at least 50%.
   - The training loss is about 2.0 at updates 271–284, inside my predicted 2.0–2.2.
   - The time per update is still 123 ms.
   - The clip keeps scaling every update down to about a tenth (0.10 at update 250).
   - The blocks are moving steadily: the top four blocks have moved about 1% of their size, the bottom four about 0.35%.

2. **Kept:** Learning is fast and stable, and the learning rate reaches its peak at update 300. Nothing in the evidence points to a better learning rate or depth factor yet.

3. **Prediction:**
   - Validation should be at least 80% at update 456 and about 85% at update 684.
   - **What would prove me wrong:**
     - Validation flattening near 75% would point to too little adaptation; I'd raise the learning rates of the lower blocks.
     - Validation going down while the training loss keeps falling would point to overfitting. In that case I'd lower the peak learning rate or add weight averaging.

## Decision 5 (update 356, scheduled; changed: nothing)

I kept both programs unchanged. There was no validation since the last decision; the next one is at update 456.

1. **Observed:**
   - The training loss fell from about 2.0 to 1.66 over updates 285–355, and is still dropping.
   - The learning rate reached its peak at update 300.
   - The total gradient norm before clipping is about 78, so the clip scales every update down to about 0.06. That scaling has been steady over the last two reports, so Adam sees a consistent gradient scale.
   - Movement is growing smoothly: the top four blocks have moved about 1.5% of their size, the bottom four about 0.5%.
   - The time per update is 123 ms.

2. **Kept:** There is no new validation result to act on, and training is stable at the peak learning rate. A change now would only muddy the reading at update 456.

3. **Prediction:**
   - Validation at update 456 should be at least 80%.
   - The training loss should be about 1.4–1.5 by update 428.
   - **What would prove me wrong:** validation below 75% at update 456. That would prompt me to give the lower blocks more learning rate, by raising the depth factor from 0.85 toward 0.9.

## Decision 6 (update 428, scheduled; changed: nothing)

I kept both programs unchanged. My training-loss prediction missed, but the validation result at update 456, just after this decision, is the better test.

1. **Observed:**
   - The training loss levelled off at about 1.65–1.70 over updates 356–427, ending at 1.57. I had predicted 1.4–1.5 by update 428, so that prediction missed.
   - The total gradient norm before clipping dropped to 43, so the clip now scales updates to 0.12 instead of 0.06.
   - Movement keeps growing steadily: the top four blocks have moved 1.7% of their size, the bottom four 0.6%.
   - There was no validation since the last decision.

2. **Kept:** A pause in the loss at peak learning rate is common, given the random crops and flips. Before deciding whether the learning rates are too high or too low, I want the validation accuracy, which comes in 28 updates.

3. **Prediction:**
   - Validation at update 456 should be at least 80%.
   - **What would prove me wrong:** validation below 75%. That together with the loss pause would suggest the peak learning rate is too high, and I'd lower the top and head learning rates by about 30%.
   - If validation is at least 80%, I'll keep the recipe.

## Decision 7 (update 499, scheduled; changed: nothing)

I kept both programs unchanged; validation came in at 80.7%, just meeting my prediction.

1. **Observed:**
   - Validation at update 456 (epoch 4) was 80.7% accuracy with loss 0.78. I had predicted at least 80%.
   - Compared with 67.5% and loss 1.61 at epoch 2, accuracy is still rising quickly.
   - The training loss dropped from about 1.65 to about 1.50 over this window.
   - The gradient norm before clipping fell to 32, so the clip now scales updates to 0.16 instead of 0.06–0.12.
   - Training is stable, and the time per update is 123 ms.

2. **Kept:** I don't see underfitting or instability. Validation accuracy is well above the training-loss level would suggest; that loss is measured on randomly cropped, label-smoothed images. The current recipe is on track. Weight averaging near the end remains the main idea to revisit later.

3. **Prediction:**
   - Validation should be about 83–86% at update 684.
   - The training loss should be about 1.3–1.4 by update 570.
   - **What would prove me wrong:** validation at or below 81% at update 684. That would mean learning has stalled, and I'd lower the peak learning rate or give the lower blocks more learning rate.

## Decision 8 (update 570, scheduled; changed: analysis.py)

I kept the training code unchanged and added early-warning triggers to `analysis.py`, because the next scheduled decision is not until update 891.

1. **Observed:**
   - The training loss stayed around 1.5, ending at 1.42 over updates 563–569. I had predicted 1.3–1.4 by now, so it's slightly behind.
   - The gradient norms are noisy: the embedding section read 24 at update 500 and 91 at update 550.
   - The total gradient norm before clipping is about 48, so the clip scales updates to about 0.10. That is in line with earlier readings.
   - The last validation was 80.7% at update 456; the next is at update 684.

2. **Changed (analysis only):**
   - **Validation drop:** the analysis now asks for an early decision if validation accuracy falls more than 3 points below its best so far. The starting best is 80.71%, the last result, because the analysis program had no record of earlier ones.
   - **Loss jump:** it also asks for an early decision if the training loss is not a finite number, or if its average over 50 updates jumps by more than 0.5.

   The updater is unchanged. A pause in the training loss at peak learning rate is expected, and the most recent validation result supports the recipe.

3. **Prediction:**
   - Validation should be about 83–86% at update 684 and about 85–88% at update 912.
   - **What would prove me wrong:** validation at or below 81% at update 684. That would point to the peak learning rate being too high for the fine-grained details, and I'd lower the top and head learning rates.

## Decision 9 (update 891, scheduled; changed: nothing)

I kept both programs unchanged. Validation came in at 86.9%, above my prediction.

1. **Observed:**
   - Validation at update 684 (epoch 6) was 86.9% accuracy with loss 0.62. I had predicted 83–86%.
   - Over the last three readings, accuracy has gone 67.5% → 80.7% → 86.9%.
   - The training loss is falling slowly, from 1.41 to 1.32 over updates 570–890.
   - The clip keeps scaling updates to 0.09–0.12, and none of the early-warning triggers fired.
   - Movement is steady: the bottom four blocks have moved about 1.1% of their size, the top four about 3%.
   - The time per update is 123.5 ms.

2. **Kept:** The recipe is still improving at a good rate and shows no instability or overfitting. Validation loss is still falling (0.78 → 0.62). The early-warning triggers will ask for a decision if validation drops by more than 3 points. I still plan to revisit weight averaging nearer the end, when the learning rate is lower.

3. **Prediction:**
   - Validation should be about 87–89% at update 912 and about 88–90% at update 1140.
   - **What would prove me wrong:** validation falling below 84%, which would trigger an early decision. In that case I'd lower the peak learning rates.

## Decision 10 (update 1150, your analysis asked: val dropped to 85.53 from best 88.71; changed: analysis.py)

This decision was triggered by the validation-drop alarm: accuracy fell from 88.7% to 85.5%. I kept the training code unchanged and only made the alarm less sensitive.

1. **Observed:**
   - Validation was 88.7% (loss 0.52) at update 912, then 85.5% (loss 0.61) at update 1140. That second result triggered the early decision.
   - The training loss has been flat at about 1.25–1.32 since update 900.
   - The learning rate is still essentially at its peak (0.986 of it), and the clip scales updates to 0.09–0.19.
   - With only 788 validation images, one reading at about 87% varies by about ±1.2 points by chance. So a 3.2-point drop is about two of those, which could be noise or the usual bouncing around at peak learning rate.

2. **Kept the updater; changed the alarm:**
   - A single drop near the peak learning rate isn't strong enough evidence to change the recipe. The learning rate will only start falling noticeably from here, and that falling phase is where most of the final accuracy is settled.
   - I raised the alarm threshold in `analysis.py` from a 3-point to a 4-point drop below the best, so noise alone is less likely to trigger it.
   - In `NOTES.md` I recorded a plan: if the next two validations both stay at or below 86%, I'll cut all learning rates to about 0.6 of their current values.

3. **Prediction:**
   - Validations at updates 1368 and 1596 should come back to about 87–89%.
   - **What would prove me wrong:** two more readings at or below 86%, or validation loss rising above 0.65. Either would mean the peak learning rate is too high for this stage, and I'd make the 0.6× cut.

## Decision 11 (update 1211, scheduled; changed: nothing)

I kept both programs unchanged. There was no new validation result, so nothing has tested the plan from the last decision yet.

1. **Observed:**
   - The training loss dropped slightly, from about 1.27 to about 1.22 over updates 1150–1210.
   - The clip scales updates to 0.12, in its usual range.
   - The learning rate is at 0.98 of its peak, and the blocks keep moving steadily with no sign of instability.

2. **Kept:** The open question is whether the drop to 85.5% at update 1140 was noise or a real effect of the high learning rate. The validations at updates 1368 and 1596 will answer it. My rule from the last decision still stands: if both are at or below 86%, I'll cut all learning rates to about 0.6 of their current values. Otherwise I'll keep the recipe.

3. **Prediction:**
   - Validation should be about 87–89% at update 1368.
   - The training loss should keep drifting down, to about 1.15–1.2 by update 1532.
   - **What would prove me wrong:** validation at or below 86% at update 1368. That would be the first of the two results that trigger the learning-rate cut.

## Decision 12 (update 1532, scheduled; changed: nothing)

I kept both programs unchanged. Validation recovered to 88.2%, so the drop at update 1140 looks like noise.

1. **Observed:**
   - Validation at update 1368 was 88.2% with loss 0.57, inside my predicted 87–89%.
   - The sequence so far is 67.5, 80.7, 86.9, 88.7, 85.5, 88.2%.
   - The training loss is slowly falling, from 1.26 to 1.18 over updates 1211–1531.
   - The gradient norm before clipping went down (21–52), so the clip now scales updates to 0.10–0.24 instead of about 0.10.
   - The learning rate is at 0.97 of its peak.

2. **Kept:** The 85.5% result was a dip within noise, not a trend, so the learning-rate cut I'd set up does not apply. Validation accuracy is flat at about 88% while the learning rate is near its peak. That's expected: the gains should come as the learning rate decays. I'm not changing a recipe that is working. My next planned step is to add weight averaging for the last part of training, once I've decided how to use it safely without being able to test it on validation.

3. **Prediction:**
   - Validation should be about 88–90% at updates 1596 and 1824.
   - **What would prove me wrong:** two readings at or below 86%. That would bring back the plan to cut all learning rates to about 0.6 of their current values.

## Decision 13 (update 1852, scheduled; changed: nothing)

I kept both programs unchanged. Validation has stopped climbing at about 87.5–88.5%, but I think that's expected while the learning rate is near its peak.

1. **Observed:**
   - Validation was 87.2% (loss 0.59) at update 1596 and 87.4% (loss 0.58) at update 1824.
   - Validation has stayed in roughly the 87–89% band since epoch 8, apart from the one-off dip to 85.5%.
   - The training loss is still slowly falling, from about 1.19 to 1.16.
   - The learning rate is at 0.95 of its peak, and the clip scales updates to 0.07–0.15.
   - Movement is steady: the bottom four blocks have moved 1.6% of their size, the top four 4.6%.

2. **Kept:** The training loss is still well above its floor under label smoothing and random crops, so the model isn't overfitting. The flat validation fits an optimizer that is bouncing around at a high learning rate. If that's right, accuracy should rise as the cosine decay lowers the learning rate: to 0.79 of the peak around epoch 30 and 0.5 around epoch 52. A sudden learning-rate cut now would trade long-run fitting for a short-term gain I can already expect from the schedule.

3. **Prediction:**
   - Validation should stay at about 87–89% through epoch 20, then start rising slowly as the learning rate falls.
   - **What would prove me wrong:** validation still at or below 88.5% at epoch 30 while the training loss keeps falling. That would mean the high-learning-rate phase is wasting updates, and I'd move to a faster decay.

## Decision 14 (update 2173, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 89.0%.

1. **Observed:**
   - Validation at update 2052 (epoch 18) was 89.0% with loss 0.556. Both are the best so far.
   - The training loss keeps falling slowly, to about 1.13–1.15.
   - The gradients are getting smaller: the total norm before clipping was 20–41, so the clip now scales updates to 0.12–0.25 instead of about 0.10. This change is gradual, so Adam's estimates of gradient size keep pace with it.
   - The learning rate is at 0.93 of its peak, and the time per update is 123.6 ms.

2. **Kept:** Validation accuracy and loss are both improving slowly while the learning rate is still near its peak, which is what I expected. Nothing points to a change that would gain a point or more right now. The faster-decay option stays on hold until the epoch-30 check.

3. **Prediction:**
   - Validation should be about 88.5–90% at updates 2280 and 2508.
   - Validation should be about 89.5% or better by epoch 30 (about update 3420).
   - **What would prove me wrong:** validation falling back to about 87% or below with validation loss above 0.6. That would mean the high learning rate is now doing harm, and I'd move to a faster decay.

## Decision 15 (update 2494, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 89.9%.

1. **Observed:**
   - Validation at update 2280 (epoch 20) was 89.9% with loss 0.554, both new bests and inside my predicted 88.5–90%.
   - The last three readings were 87.4, 89.0 and 89.9%, so accuracy is still trending up.
   - The training loss is flat at about 1.12–1.13.
   - The learning rate is at 0.91 of its peak, and the clip scales updates to 0.10–0.28.
   - Training is stable, and the time per update is 123.4 ms.

2. **Kept:** Validation is still rising while the learning rate is near its peak. A change now would need evidence that something is going wrong, and there is none. The epoch-30 check for a faster decay stays in the plan.

3. **Prediction:**
   - Validation should be about 89–91% at updates 2508 and 2736.
   - **What would prove me wrong:** validation falling back below 88% with validation loss above 0.6. That would make me consider a faster decay.

## Decision 16 (update 2814, scheduled; changed: nothing)

I kept both programs unchanged; validation is holding at about 89–90%.

1. **Observed:**
   - Validation was 90.0% (loss 0.547) at update 2508 and 89.2% (loss 0.588) at update 2736. Both are inside my predicted 89–91%.
   - The training loss is slowly falling, to about 1.08–1.11.
   - The learning rate is at 0.88 of its peak.
   - The clip scales updates to 0.08–0.22, which is noisy but has no trend, and training is stable.

2. **Kept:** Validation has risen to about 89–90% and is now holding there, while the learning rate decays slowly. Everything I've seen fits the view that further gains will come as the learning rate falls. The one-reading drop to 89.2% is within the ±1.2-point noise of a single validation, and the validation loss is still near its best. I still don't see evidence for a change that would gain a point or more.

3. **Prediction:**
   - Validation should be about 89–91% at the epoch-26 and epoch-28 evaluations.
   - It should be about 90% or better by epoch 30.
   - **What would prove me wrong:** validation still at or below 89% at epoch 30 with the training loss falling. That would point to the decay being too slow, and I'd move to a faster decay.

## Decision 17 (update 3135, scheduled; changed: nothing)

I kept both programs unchanged. Validation has slipped slightly for three readings in a row, so I've set a firm rule for the next decision.

1. **Observed:**
   - Validation went 90.0 → 89.2 → 88.5% (epochs 22, 24, 26), and validation loss went 0.547 → 0.588 → 0.590.
   - The training loss is flat at about 1.08–1.10.
   - The learning rate is at 0.85 of its peak, and the clip scales updates to 0.12–0.28.

2. **Kept, with a firm rule for the next decision:**
   - A 1.5-point fall over three noisy readings is suggestive but not conclusive; each reading varies by about ±1.2 points.
   - The training loss is still far above its floor under label smoothing and random crops, so the model is not memorising the training set.
   - **If** both validations at updates 3192 and 3420 come back at or below 89.2% with validation loss above 0.58, I'll reshape the schedule at update 3456 so the learning rate falls faster from then on, still reaching 0 at update 11400.

3. **Prediction:**
   - If the dip is noise, validation will be about 89–90.5% at update 3192 or 3420.
   - **What would prove me wrong:** both readings at or below 89.2% with validation loss still rising. That would mean the high learning rate has become harmful, and I'd make the faster-decay change.

## Decision 18 (update 3456, scheduled; changed: nothing)

I kept both programs unchanged. The new validation results did not meet the faster-decay condition, so the dip at epoch 26 was noise.

1. **Observed:**
   - Validation was 90.6% (loss 0.511) at update 3192, both new bests, and 89.6% (loss 0.546) at update 3420.
   - My rule called for both to be at or below 89.2% with loss above 0.58, and neither was. The slide to 88.5% at epoch 26 was noise.
   - The training loss is slowly falling, to about 1.05–1.08.
   - The learning rate is at 0.81 of its peak, and the clip scales updates to 0.13–0.23.

2. **Kept:** Validation is still improving slowly while the learning rate decays. I see no sign of overfitting or instability, so no change is justified.

3. **Prediction:**
   - Validation should be about 89.5–91% over epochs 32–36, then rise further as the learning rate passes half its peak (around epoch 52).
   - **What would prove me wrong:** two readings at or below 88.5%, or validation loss above 0.6. Either would make me reconsider the faster decay.

## Decision 19 (update 3776, scheduled; changed: nothing)

I kept both programs unchanged; validation is holding within its usual range.

1. **Observed:**
   - Validation at update 3648 (epoch 32) was 89.3% with loss 0.578. That's inside the 88.5–90.6% band it has stayed in since epoch 20.
   - The training loss is slowly falling, to about 1.04–1.08.
   - The gradients are getting smaller. The total norm before clipping ranged from 7 to 30, so in some readings the clip barely acted (0.72 and 0.44 instead of about 0.2).
   - The learning rate is at 0.78 of its peak.

2. **Kept:** Accuracy is steady at about 89.5–90% while the learning rate is still high, and nothing has gone wrong. The clip acting less often only means updates get somewhat larger as the gradients shrink. That shift is gradual, and the falling learning rate offsets it. I still see no evidence for a change that would gain a point or more; the expected gain is from the learning-rate decay ahead.

3. **Prediction:**
   - Validation should be about 89–91% through epoch 40.
   - It should rise toward 91–92% as the learning rate drops below about half its peak (past epoch 52).
   - **What would prove me wrong:** two readings at or below 88.5%, or validation loss above 0.6. Either would make me revisit the faster decay.

## Decision 20 (update 4097, scheduled; changed: nothing)

I kept both programs unchanged, but validation loss is now rising, so I've set a rule for when to act.

1. **Observed:**
   - Validation at update 3876 (epoch 34) was 89.2% accuracy, but its loss rose to 0.630. That's the highest since epoch 10, up from 0.511 at epoch 28.
   - The training loss keeps falling, to about 1.02–1.04.
   - The learning rate is at 0.74 of its peak, and the clip scales updates to 0.11–0.33.

2. **Kept, with a rule:**
   - Accuracy has held steady while validation loss rises, which means the model is growing more confident than it should be. That is common in long fine-tuning runs, and accuracy hasn't fallen yet.
   - One reading isn't enough to justify cutting the learning rate early. That would give up fitting time the long tail of the schedule could still use.
   - I recorded this rule in `NOTES.md`: if the next two validations both have loss above 0.62 and accuracy at or below 89%, I'll make the learning rate fall faster from then on.

3. **Prediction:**
   - Validation accuracy should be about 89–90.5% at updates 4104 and 4332, with validation loss back near 0.55–0.60.
   - **What would prove me wrong:** validation loss staying above 0.62 with accuracy at or below 89% on both readings. That would trigger the faster decay.

## Decision 21 (update 4418, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 91.4%, and the concern about rising validation loss has passed.

1. **Observed:**
   - Validation at update 4104 (epoch 36) was 91.4% with loss 0.597, the best accuracy so far.
   - At update 4332 (epoch 38) it was 90.7% with loss 0.526. The jump to 0.630 at epoch 34 did not persist, so the overconfidence concern is dropped.
   - The training loss is about 1.00–1.03.
   - The learning rate is at 0.70 of its peak, and training is stable.

2. **Kept:** Accuracy is now rising as the learning rate decays, which is what I expected. Neither condition for the faster-decay rule was met. I see no evidence that a change would beat the current path.

3. **Prediction:**
   - Validation should be about 90.5–92% through epoch 50.
   - It should reach about 92% or better in the last third of training, as the learning rate approaches 0.
   - **What would prove me wrong:** two readings at or below 89%, or validation loss above 0.62 again. Either would make me reconsider the faster decay.

## Decision 22 (update 4738, scheduled; changed: nothing)

I kept both programs unchanged; validation is holding steady.

1. **Observed:**
   - Validation at update 4560 (epoch 40) was 90.1% with loss 0.611.
   - Since epoch 28, readings have stayed between 89.2% and 91.4%.
   - The training loss is about 0.99–1.03, still falling very slowly.
   - The learning rate is at 0.66 of its peak, and the clip scales updates to 0.15–0.43.
   - Training is stable.

2. **Kept:** Neither of the warning conditions I set (two readings at or below 89%, or validation loss above 0.62) has been met. The model is not overfitting, and the expected gain still lies in the second half of the decay.

   I'm considering one change for later: averaging the weights over roughly the last 10 epochs and using that average as the final model. It is low-risk once the learning rate is small. I'd decide around update 9000.

3. **Prediction:**
   - Validation should be about 90–91.5% through epoch 50.
   - It should rise to about 91.5–92.5% by epoch 70, as the learning rate drops below 0.35 of its peak.
   - **What would prove me wrong:** two readings at or below 89%, or validation loss staying above 0.62. Either would prompt the faster decay.

## Decision 23 (update 5059, scheduled; changed: nothing)

I kept both programs unchanged. Validation set a new best of 91.9%.

1. **Observed:**
   - Validation was 91.4% at epoch 42 and 91.9% at epoch 44, both with loss 0.54. Accuracy is a new best.
   - The training loss is about 0.97–1.00.
   - The learning rate is at 0.61 of its peak, and the clip scales updates to 0.15–0.47.
   - Training is stable.

2. **Kept:** Accuracy is rising as the learning rate decays, just as predicted, and validation loss is steady. There's no sign of overfitting and no reason to change course. I'll keep the weight-averaging idea for the last ~10 epochs to decide around update 9000.

3. **Prediction:**
   - Validation should be about 91–92.5% through epoch 55.
   - It should reach about 92–93% by epoch 80.
   - **What would prove me wrong:** two readings at or below 90%, or validation loss above 0.62. Either would mean the gains have reversed, and I'd look at the faster decay.

## Decision 24 (update 5379, scheduled; changed: nothing)

I kept both programs unchanged; validation matched its best of 91.9%.

1. **Observed:**
   - Validation at update 5244 (epoch 46) was 91.9% with loss 0.534. That ties the best accuracy and is the lowest validation loss since epoch 28.
   - The training loss is slowly falling, to about 0.97.
   - The gradients keep shrinking, so the clip now scales updates to 0.3–0.9, and at times it barely acts. The change is gradual, and the falling learning rate (now 0.57 of its peak) offsets the larger steps it allows.
   - The time per update is 123.4 ms.

2. **Kept:** The decay is delivering the gains I predicted, and both validation accuracy and loss are at or near their best. There's no evidence a change would do better. I'll keep the averaging idea to decide around update 9000.

3. **Prediction:**
   - Validation should be about 91.5–92.5% through epoch 56.
   - It should reach about 92–93% by epoch 80.
   - **What would prove me wrong:** two readings at or below 90.5%, or validation loss above 0.6. Either would prompt a closer look at the decay.

## Decision 25 (update 5700, scheduled; changed: nothing)

I kept both programs unchanged. Validation jumped to a new best of 93.2%, above my predicted range.

1. **Observed:**
   - Validation at update 5472 (epoch 48) was 93.2% accuracy with loss 0.556. I had predicted 91.5–92.5%.
   - The training loss is slowly falling, to about 0.95–0.97.
   - The learning rate is at 0.52 of its peak, and the clip scales updates to 0.2–0.48.
   - Training is stable at the halfway point of the run.

2. **Kept:** The learning-rate decay is producing steady gains. Validation has risen over the last six readings, apart from a noise-sized dip at epoch 40, and nothing points to a better alternative. I'll still decide on averaging the weights over the last ~10 epochs around update 9000.

3. **Prediction:**
   - Validation should be about 92–93.5% through epoch 60.
   - It should reach about 93%+ by the end.
   - **What would prove me wrong:** two readings at or below 91%, or validation loss above 0.6. Either would mean the trend has reversed.

## Decision 26 (update 6021, scheduled; changed: nothing)

I kept both programs unchanged; validation is holding near its best.

1. **Observed:**
   - Validation was 92.4% (loss 0.560) at update 5700 and 93.0% (loss 0.532) at update 5928. That's close to the best of 93.2%, and both readings are inside my predicted 92–93.5%.
   - The training loss is slowly falling, to about 0.94–0.96.
   - The learning rate is at 0.48 of its peak, and the clip scales updates to 0.25–0.45.
   - Training is stable.

2. **Kept:** Accuracy is holding near 93% as the learning rate decays, and nothing is going wrong. No change is likely to gain a point or more at this stage. I'll decide on averaging the weights over the last ~10 epochs around update 9000.

3. **Prediction:**
   - Validation should be about 92.5–93.5% through epoch 65.
   - It should be about 93–94% by the end of the run.
   - **What would prove me wrong:** two readings at or below 91.5%, or validation loss above 0.6. Either would mean the gains have reversed.

## Decision 27 (update 6341, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 93.4%.

1. **Observed:**
   - Validation at update 6156 (epoch 54) was 93.4% with loss 0.552, inside my predicted 92.5–93.5%.
   - The training loss is about 0.94.
   - The gradients have shrunk enough that the clip sometimes doesn't act at all: the total norm before clipping was 4.4–19, and one reading had no scaling. The shift is gradual, and the learning rate (now 0.44 of its peak) keeps falling, so the effective step stays under control.
   - Training is stable.

2. **Kept:** Accuracy is still rising slowly as the learning rate decays, and there's no sign of a problem. I still plan to decide on averaging the weights over the last ~10 epochs around update 9000.

3. **Prediction:**
   - Validation should be about 92.5–94% through epoch 65.
   - It should be about 93–94% at the end.
   - **What would prove me wrong:** two readings at or below 91.5%, or validation loss above 0.6. Either would mean the gains have reversed.

## Decision 28 (update 6662, scheduled; changed: nothing)

I kept both programs unchanged. I'm also leaning against the weight-averaging idea I'd been carrying.

1. **Observed:**
   - Validation was 92.8% (loss 0.524) at update 6384 and 93.3% (loss 0.520) at update 6612. Validation loss is at its best.
   - The training loss is slowly falling, to about 0.92.
   - The learning rate is at 0.39 of its peak, and the clip scales updates to 0.12–0.90.
   - Training is stable.

2. **Kept:** Accuracy is holding at about 93% as the learning rate decays.

   On averaging the weights over the last ~10 epochs: the learning rate will be almost 0 by then, so the final weights will already be nearly settled, and the likely gain is at most about 0.2 points. The averaged weights would only replace the live ones after the last update, and validation is measured on the live weights, so I could never check that the average helps. That untested risk isn't worth the small gain, so I'm leaning against it.

3. **Prediction:**
   - Validation should be about 92.5–94% through epoch 70.
   - It should be about 93–94% at the end.
   - **What would prove me wrong:** two readings at or below 91.5%, or validation loss above 0.6. Either would mean the gains have reversed.

## Decision 29 (update 6982, scheduled; changed: nothing)

I kept both programs unchanged; validation is steady at 93.3%.

1. **Observed:**
   - Validation at update 6840 (epoch 60) was 93.3% with loss 0.519. Validation loss is at its best.
   - Accuracy has held at about 93% for six epochs.
   - The training loss is about 0.90–0.93.
   - The gradients have shrunk further: the total norm before clipping was 2.4–19, and in three of the six readings the clip didn't act at all. This is gradual, and the learning rate (now 0.35 of its peak) keeps falling, so step sizes stay under control.
   - Training is stable.

2. **Kept:** The recipe is steady at about 93% with validation loss still improving. Nothing suggests a change would gain a point or more this late. I'm still not planning to average the weights.

3. **Prediction:**
   - Validation should be about 92.5–94% for the rest of the run.
   - It should end at about 93–94%.
   - **What would prove me wrong:** two readings at or below 91.5%, or validation loss above 0.6. Either would mean the gains have reversed.

## Decision 30 (update 7303, scheduled; changed: nothing)

I kept both programs unchanged; validation is holding at about 93%, and validation loss reached its best yet.

1. **Observed:**
   - Validation was 92.9% (loss 0.528) at update 7068 and 92.6% (loss 0.479) at update 7296. That loss is by far the best so far.
   - Accuracy has stayed between 92.6% and 93.4% for 12 epochs, which is within the noise of a single reading.
   - The training loss is about 0.90–0.92.
   - The learning rate is at 0.30 of its peak, the clip scales updates to 0.2–1.0, and training is stable.

2. **Kept:** The model is steady at about 93%, and falling validation loss suggests it's getting better calibrated without overfitting. With 36 epochs left and the learning rate still decaying, nothing in the evidence points to a change that would gain a point or more.

3. **Prediction:**
   - Validation should be about 92.5–94% for the rest of the run.
   - It should end at about 93–94%.
   - **What would prove me wrong:** two readings at or below 91.5%, or validation loss above 0.6. Either would mean the gains have reversed.

## Decision 31 (update 7624, scheduled; changed: nothing)

I kept both programs unchanged. Validation set a new best of 93.8%.

1. **Observed:**
   - Validation at update 7524 (epoch 66) was 93.8% with loss 0.491, the best accuracy so far.
   - The training loss is about 0.89–0.90.
   - The learning rate is at 0.26 of its peak, the clip scales updates to 0.35–1.0, and training is stable.

2. **Kept:** Accuracy is creeping up as the learning rate decays, and validation loss is near its best. Nothing points to a change this late.

3. **Prediction:**
   - Validation should be about 93–94.5% for the rest of the run.
   - It should end at about 93.5–94.5%.
   - **What would prove me wrong:** two readings at or below 92%, or validation loss above 0.6. Either would mean the gains have reversed.

## Decision 32 (update 7944, scheduled; changed: nothing)

I kept both programs unchanged. Validation set another new best of 93.9%.

1. **Observed:**
   - Validation at update 7752 (epoch 68) was 93.9% with loss 0.483, the best accuracy so far.
   - The training loss is about 0.89–0.90.
   - The learning rate is at 0.23 of its peak. The clip often doesn't act now, with the total gradient norm before clipping at 1.8–18.
   - Training is stable.

2. **Kept:** Accuracy is still rising slowly as the learning rate decays, and validation loss is near its best. With 30 epochs left, there's no evidence for a change.

3. **Prediction:**
   - Validation should be about 93–94.5% for the rest of the run.
   - It should end at about 93.5–94.5%.
   - **What would prove me wrong:** two readings at or below 92%, or validation loss above 0.6. Either would mean the gains have reversed.

## Decision 33 (update 8265, scheduled; changed: nothing)

I kept both programs unchanged; validation is holding at about 93.5%.

1. **Observed:**
   - Validation was 93.3% (loss 0.491) at update 7980 and 93.7% (loss 0.485) at update 8208. Both are within noise of the best, 93.9%.
   - The training loss is about 0.89.
   - The learning rate is at 0.19 of its peak, and the clip mostly doesn't act now.
   - Training is stable.

2. **Kept:** The model has settled at about 93.5% with validation loss near its best. A change in the final 28 epochs would have little time to pay off and could undo the gains.

3. **Prediction:**
   - Validation should be about 93–94.5% through the end of the run.
   - The final result should be about 93.5–94.5%.
   - **What would prove me wrong:** two readings at or below 92%, or validation loss above 0.6.

## Decision 34 (update 8586, scheduled; changed: nothing)

I kept both programs unchanged, and I've decided to run this setup unchanged to the end.

1. **Observed:**
   - Validation at update 8436 (epoch 74) was 93.8% with loss 0.471, the best validation loss so far.
   - The training loss is flat at about 0.88–0.89.
   - The learning rate is at 0.15 of its peak, and the clip is mostly not acting.
   - Training is stable.

2. **Kept:** Accuracy is holding at about 93.5–93.9% and validation loss is still improving. I've also ruled out averaging the weights at the end, since its benefit can't be checked on validation and the final weights will already be nearly settled.

3. **Prediction:**
   - Validation should be about 93–94.5% for the remaining checks.
   - The final result should be about 93.5–94.5%.
   - **What would prove me wrong:** two readings at or below 92%, or validation loss above 0.6.

## Decision 35 (update 8906, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 94.7%.

1. **Observed:**
   - Validation was 93.9% (loss 0.482) at update 8664 and 94.7% (loss 0.471) at update 8892. That second reading is the best accuracy so far.
   - The training loss is about 0.88.
   - The learning rate is at 0.12 of its peak, and training is stable.

2. **Kept:** Accuracy is still creeping up as the learning rate decays toward zero. With 22 epochs left, a change would have little time to help and could only put this progress at risk.

3. **Prediction:**
   - Validation should be about 93.5–95% for the remaining checks.
   - The final result should be about 94% (93.5–95%).
   - **What would prove me wrong:** two readings at or below 92.5%.

## Decision 36 (update 9227, scheduled; changed: nothing)

I kept both programs unchanged; validation is holding near its best.

1. **Observed:**
   - Validation at update 9120 (epoch 80) was 94.4% with loss 0.480, close to the best of 94.7%.
   - The training loss is about 0.88.
   - The learning rate is at 0.09 of its peak, the clip is mostly not acting, and training is stable.

2. **Kept:** The run is converging as planned, and a change in the last 19 epochs would have little room to pay off.

3. **Prediction:**
   - Validation should be about 93.8–95% for the remaining checks.
   - The final result should be about 94–94.5%.
   - **What would prove me wrong:** two readings at or below 93%.

## Decision 37 (update 9548, scheduled; changed: nothing)

I kept both programs unchanged; the run is converging as planned.

1. **Observed:**
   - Validation at update 9348 (epoch 82) was 94.5% with loss 0.464, the best validation loss so far.
   - The training loss is about 0.87–0.88.
   - The learning rate is at 0.07 of its peak, and training is stable.

2. **Kept:** The model is settling near its best, and there are only 16 epochs left for any change to pay off.

3. **Prediction:**
   - Validation should be about 94–95% for the remaining checks.
   - The final result should be about 94–94.5%.
   - **What would prove me wrong:** two readings at or below 93%.

## Decision 38 (update 9868, scheduled; changed: nothing)

I kept both programs unchanged; validation is steady at about 94–94.5%.

1. **Observed:**
   - Validation was 94.5% (loss 0.475) at update 9576 and 94.0% (loss 0.466) at update 9804. Both are within noise of the best, 94.7%.
   - The training loss is about 0.87–0.88.
   - The learning rate is at 0.05 of its peak, and the clip mostly isn't acting.

2. **Kept:** The model has converged as planned. With about 13 epochs left, no change is worth the risk.

3. **Prediction:**
   - The final validation readings should be about 94–95%.
   - The final test accuracy should be about 94%.
   - **What would prove me wrong:** a reading at or below 93%.

## Decision 39 (update 10189, scheduled; changed: nothing)

I kept both programs unchanged; the run has settled.

1. **Observed:**
   - Validation at update 10032 (epoch 88) was 93.9% with loss 0.471.
   - Over the last six readings, accuracy has stayed between 93.9% and 94.7%.
   - The training loss is about 0.87.
   - The learning rate is at 0.03 of its peak, so the weights are barely moving.

2. **Kept:** The model has converged. Any change now would have almost no time to pay off.

3. **Prediction:**
   - The final validation readings should be about 93.8–94.8%.
   - The final test accuracy should be about 94%.

## Decision 40 (update 10509, scheduled; changed: nothing)

I kept both programs unchanged; validation is at its best level.

1. **Observed:**
   - Validation was 94.5% at update 10260 and 94.7% at update 10488, with loss about 0.47. That second reading ties the best accuracy so far.
   - The training loss is about 0.87.
   - The learning rate is at 0.016 of its peak, so the weights are essentially frozen.

2. **Kept:** The model has converged with validation at its best. Any change now could only add risk.

3. **Prediction:** The final test accuracy should be about 94–94.5%, in line with the last eight validation readings (93.9–94.7%).

## Decision 41 (update 10830, scheduled; changed: nothing)

This was the last scheduled decision, and I kept both programs unchanged. The run should end with the updater I installed at update 71.

1. **Observed:**
   - Validation at update 10716 (epoch 94) was 94.4% with loss 0.473.
   - Every reading from epoch 82 to 94 was between 93.9% and 94.7%.
   - The training loss is flat at about 0.87.
   - The learning rate is at 0.007 of its peak, so the remaining 570 updates will barely change the weights.

2. **Kept:** The model has converged and validation is steady at about 94.4%.

   The only change I made all run was at update 71, replacing the starting optimizer with one that:
   - gives the new classifier head a 10× higher learning rate (2e-3);
   - scales the learning rate by depth: 2e-4 at the top block, 0.85× for each block below;
   - shortens the warmup to 300 updates, followed by cosine decay to 0;
   - clips the total gradient norm at 5.

   I intended the clip only to catch spikes, but for about the first half of the run it scaled down almost every update. It still worked well.

3. **Prediction:**
   - The final test accuracy should be about 94% (93.5–95%), matching the validation range of the last 14 epochs.
   - Validation was 67.5% at epoch 2, rose to about 87–90% by epochs 6–30, and to about 94% from epoch 66 onward. I can't compare against the starting recipe, because it was replaced at update 71 and never ran in full.
