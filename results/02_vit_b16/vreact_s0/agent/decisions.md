# Agent decisions: `vreact_s0`
The agent's reply at each decision point, as written. `changed` lists the programs it edited.

## Decision 1 (update 71, scheduled; changed: analysis.py, updater.py)

I've replaced the starting optimizer with a new `updater.py`; it takes over at the next update (72). I haven't run it yet, so the first real test is whether it installs and survives its first 20 updates without a rollback.

1. **Observed:** Training loss went from 5.281 to 5.235 over 71 updates. That's barely below ln(196) = 5.278, the loss of a model that guesses uniformly over 196 classes. Total gradient norm at update 50 was 2.25. The likely bottleneck is the zero-initialised 196-class head: during warmup its learning rate is only about 1e-5, so the head, and through it the whole network, has barely started learning.

2. **Changed:**
   - **Optimizer:** still AdamW (betas 0.9/0.999, eps 1e-8, weight decay 0.05 on weight matrices and 0 on gains, biases, position embedding and class token). It keeps Adam's running gradient statistics from the starting optimizer, so nothing restarts from zero.
   - **Per-layer learning rates:** the top encoder layer peaks at 1e-4. Each step down in depth multiplies the rate by 0.85, so the patch embedding peaks at about 1.2e-5. Pretrained low-level features change slowly while the top layers adapt to car models.
   - **Head:** peaks at 1e-3 (10× the backbone) and warms up by update 200, since it starts from zero.
   - **Schedule:** the backbone still warms up linearly to update 570, then decays on a cosine to 0 at update 11400. Both schedules are written in terms of the absolute update number.
   - **`analysis.py`:** now also reports validation results, gradient norm by depth, how far each depth's weights have moved from their starting values, and the current learning rates. To measure that movement it keeps a copy of the starting backbone weights on the GPU, about 340 MB.
   - **`NOTES.md`:** records these predictions and plans.

3. **Prediction:**
   - Training loss below about 4.5 by update 250 and below about 3 by update 600.
   - The first validation (around update 228) clearly above chance.
   - Validation accuracy around 80% or more by about epoch 20.

   **What would prove me wrong, and the response:**
   - If the lowest layers barely move from their starting weights while validation stalls, the per-layer decay is too strong; I'll relax it to 0.9.
   - If training loss drops quickly to the floor set by label smoothing (about 1.0) and validation plateaus, the model is overfitting; I'll lower the learning rate or raise weight decay.
   - If fitting is just slow, I'll raise the backbone peak to about 2e-4.

## Decision 2 (update 142, scheduled; changed: nothing)

1. **Observed:** The new updater fixed the slow start. Training loss fell from 5.05 (updates 71–77) to 3.88 (updates 135–141), already beating my target of 4.5 by update 250. The gradient norm is 3.0 and dominated by the head (2.1). The rest falls off with depth: the patch embedding is 1.35, encoder layers about 0.25–0.6, and the final LayerNorm 0.025. There's no validation result yet; the first comes at update 228. The weight-movement figures read 0 because the analysis saved its starting copy of the weights at update 100, not update 0. Movement is measured from update 100 onward, which still works for comparing depths. Time per update is 281.8 ms, the same as the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`. The backbone is still in warmup until update 570, and nothing yet points to a problem. Changing settings before the first validation result would be guessing. I added this decision's numbers to `NOTES.md`.

3. **Prediction:**
   - Training loss below 3.0 by about update 600.
   - First validation (around update 228) clearly above chance; I expect roughly 15–40%.
   - About 80% validation accuracy by epoch 20.

   **What would prove me wrong, and the response:**
   - If loss stalls above 3.5 by update 400, I'll raise the backbone learning rate.
   - If the lowest layers show almost no weight movement while validation lags, I'll relax the per-layer decay from 0.85 to 0.9.

## Decision 3 (update 214, scheduled; changed: nothing)

1. **Observed:** Training loss kept falling, from 3.81 (updates 142–148) to 3.10 (updates 207–213). The backbone is now learning, not just the head:
   - **Gradients:** the encoder layers' gradient norms roughly doubled from update 150 to update 200, to about 1.2–2.0 each. The head's stayed flat at about 2.2, so the total gradient norm rose to 7.7.
   - **Weight movement since update 100:** it increases steadily with depth, from about 0.2% at the patch embedding to 2.1% at the top encoder layer, as the per-layer learning rates intend. The final LayerNorm, which is given the same rate as the top layer, has moved only 0.5%.
   - **Validation and timing:** there's still no validation result; the first is due at update 228. Time per update is 282 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Loss is falling faster than I predicted, and the backbone is still in warmup until update 570. The first validation result arrives before the next decision, so retuning now would mean acting without it.

3. **Prediction:**
   - First validation (around update 228) between 15% and 45%.
   - Training loss around 2.5 by update 400, then below 2.0 by about update 600.

   **What would prove me wrong, and the response:**
   - If loss stalls above 3.0 by update 400, I'll raise the backbone learning rate.
   - If validation lags well behind training loss while the bottom layers barely move, I'll relax the per-layer decay from 0.85 to 0.9.

## Decision 4 (update 285, scheduled; changed: nothing)

1. **Observed:** The first validation result, at update 228 (epoch 2), is 42.51% accuracy with validation loss 2.33. That's at the top of the 15–45% I predicted. Training loss fell from 2.99 to 2.27 since the last decision, faster than my target of about 2.5 by update 400. The encoder layers' gradient norms are steady at about 1.1–1.9, and the head's dropped to 1.9. Weight movement since update 100 still increases smoothly with depth, from 0.3% at the patch embedding to 2.9% at the top encoder layer. Time per update is 282 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Training and validation are both improving quickly, and the backbone learning rate is still warming up until update 570. Nothing yet shows the per-layer decay or the peak rate is wrong. The real risk with about 37 images per class over 100 epochs is overfitting later in the run. I'll judge that once training loss nears its floor of about 1.0 under label smoothing.

3. **Prediction:**
   - Validation around 55–65% at update 342 and 65–75% at update 456.
   - Training loss below 2.0 by update 400.

   **What would prove me wrong, and the response:**
   - If validation stalls below 60% while training loss keeps dropping, the model is overfitting early; I'll lower the peak learning rate or raise weight decay.
   - If both stall, I'll raise the backbone learning rate or relax the per-layer decay from 0.85 to 0.9.

## Decision 5 (update 356, scheduled; changed: nothing)

1. **Observed:** Training loss fell from 2.22 (updates 285–291) to 1.76 (updates 349–355), already below my target of 2.0 by update 400. There's no new validation result. Validation runs every 228 updates, not at update 342 as I wrote last time, so the next one is at update 456. Gradient norms are steady: total 7.5, encoder layers about 1.2–2.0 each, head down to 1.5. Weight movement since update 100 still increases smoothly with depth, from 0.4% at the patch embedding to 4.1% at the top encoder layer. Time per update is 282 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Training is ahead of every prediction so far, and no new validation result has arrived to test the recipe against. The backbone learning rate is still warming up until update 570. Overfitting remains the risk to watch as training loss approaches its floor of about 1.0 under label smoothing.

3. **Prediction:**
   - Validation 65–75% at update 456.
   - Training loss around 1.5 by update 500.

   **What would prove me wrong, and the response:**
   - If validation at update 456 is below 60% while training loss keeps dropping, the model is overfitting early; I'll lower the backbone peak learning rate or raise weight decay.
   - If training loss flattens above 1.8, I'll raise the backbone learning rate.

## Decision 6 (update 428, scheduled; changed: nothing)

1. **Observed:** Training loss has levelled off at about 1.64–1.75 since update 370; the last window, updates 421–427, averaged 1.64. The backbone learning rate is still rising until update 570, so a flat stretch here is likely the rising rate offsetting progress, not a stall. Gradient norms fell: the total went from 7.5 to 5.8, encoder layers to about 0.9–1.5, and the head to 1.2. Weight movement since update 100 still increases smoothly with depth, from 0.5% at the patch embedding to 4.5% at the top encoder layer. There's no new validation result; the next one is at update 456. Time per update is 282 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Validation at update 456 comes before the next decision and is the deciding piece of evidence. Changing the recipe on a short plateau during warmup would be premature.

3. **Prediction:**
   - Validation 65–75% at update 456.
   - Training loss falling again after warmup ends at update 570.

   **What would prove me wrong, and the response:**
   - Validation below about 60% at update 456 would point to the high learning rate hurting generalisation; I'd lower the backbone peak (1e-4) or shorten the warmup.
   - If training loss is still flat around 1.65 by update 700, the rate is likely too high to make progress; I'll lower the peak.

## Decision 7 (update 499, scheduled; changed: nothing)

1. **Observed:** Validation at update 456 (epoch 4) is 78.05% accuracy with validation loss 0.95, above the 65–75% I predicted and up from 42.5% two epochs earlier. Training loss has started falling again: it went from about 1.62 to 1.38 since the last decision, ending the 1.65 plateau. Validation loss is below training loss, which fits label smoothing and random crops raising the training loss rather than a sign of trouble. Gradient norms and weight movement are steady, with movement since update 100 still increasing smoothly with depth. Time per update is 282 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Accuracy is rising fast and nothing points to a wrong learning rate or per-layer decay. The main remaining risk is overfitting once training loss nears its floor of about 1.0 under label smoothing, probably around epochs 15–30. Then the validation trend will decide whether to add stronger weight decay or weight averaging toward the end of the run.

3. **Prediction:**
   - Validation above 82% at update 684 (epoch 6).
   - About 88% or more by epoch 20.

   **What would prove me wrong:** if validation stalls at about 78–80% while training loss keeps dropping, regularisation becomes the priority, before the cosine decay gets far.

## Decision 8 (update 570, scheduled; changed: analysis.py)

1. **Observed:** Training loss has held at about 1.37–1.49 since the last decision, ending at 1.37 in updates 563–569. That's the backbone learning rate reaching its peak at update 570, as scheduled. There's no new validation result; the last was 78.05% at update 456. Gradient norms are steady, with a total of 5.8 and the encoder layers at about 0.9–1.5. Weight movement since update 100 still increases smoothly with depth, from 0.6% at the patch embedding to 5.8% at the top encoder layer. Time per update is 282 ms, against 284 ms for the starting program.

2. **Kept the updater; changed the analysis:** I didn't change `updater.py`. The last validation was well above my prediction, and flat training loss at peak learning rate is expected rather than a problem. The next scheduled decision isn't until update 891, a 321-update gap. So I added two triggers to `analysis.py` that ask for an extra decision sooner. One fires if a validation result drops more than 4 points below the best so far. The other fires if training loss stops being a finite number. I haven't run the edited file; it's reloaded at the next analysis call, and an error there would show up at the next decision. I also added this decision's numbers to `NOTES.md`.

3. **Prediction:**
   - Training loss falling as the cosine decay starts, to about 1.2 by update 900.
   - Validation above 82% at update 684 and about 85% at update 912.

   **What would prove me wrong:** a validation drop at peak learning rate would mean the rate is too high. I'd lower the backbone peak, for example to 7e-5.

## Decision 9 (update 891, scheduled; changed: nothing)

1. **Observed:** Validation at update 684 (epoch 6) is 84.14% accuracy with validation loss 0.75. That's above the 82% I predicted, and up from 78.05% at epoch 4. Training loss fell from about 1.28 to 1.16 over updates 570–890. The lowest training loss possible with label smoothing 0.1 over 196 classes is about 0.85, so there's still room to fit. Gradient norms vary between readings (total 3.0–6.1) with no trend, and nothing has spiked. The new triggers in `analysis.py` (validation drop, non-finite loss) haven't fired. Weight movement since update 100 still increases smoothly with depth, from 0.8% at the patch embedding to 7.4% at the top encoder layer. Time per update is 283 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Accuracy is still climbing about 3–6 points every 2 epochs, and training loss is well above its floor, so the model isn't overfitting yet. Neither the learning rate nor the per-layer decay shows a problem. I don't have a stated reason for any specific change that would beat what's working. Weight averaging adds little while the learning rate decays to zero; I'll weigh it, or stronger weight decay, if validation stalls while training loss approaches its floor.

3. **Prediction:**
   - Validation 85–87% at update 912 and about 87% at update 1140.
   - Training loss around 1.05–1.1 by update 1200.

   **What would prove me wrong:** if validation stays at 84–85% for two readings while training loss keeps falling, the model has started overfitting. I'd then strengthen regularisation, with higher weight decay first.

## Decision 10 (update 1211, scheduled; changed: nothing)

1. **Observed:** Validation accuracy kept rising, from 84.14% to 84.64% at update 912 and 86.42% at update 1140. That's close to my predictions of 85–87% and about 87%. Validation loss has stayed flat at about 0.74–0.75 for three readings. Training loss fell from 1.13 to about 1.06, just past my target of 1.05–1.1 by update 1200. The lowest possible with label smoothing is about 0.85, so it's getting close. Gradient norms are steady (total 3.6–5.5), and weight movement since update 100 keeps growing slowly, from 0.9% at the patch embedding to 8.7% at the top encoder layer. Time per update is 284 ms, the same as the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Accuracy is still improving at peak learning rate, and validation loss hasn't started rising, so overfitting hasn't set in. Most of the cosine decay is still ahead, and that's usually where accuracy gains the most. If validation loss starts rising while training loss nears 0.9, I'll raise backbone weight decay from 0.05 to about 0.1–0.2.

3. **Prediction:**
   - Validation around 87% at update 1368 and 87–88% at update 1596.
   - Training loss about 1.0 by update 1500.

   **What would prove me wrong:** validation loss rising above about 0.78 while training loss keeps falling would show overfitting, and I'd raise the weight decay.

## Decision 11 (update 1532, scheduled; changed: nothing)

1. **Observed:** Validation at update 1368 (epoch 12) is 86.29%, essentially level with 86.42% at epoch 10 and below the roughly 87% I predicted. Validation loss rose from 0.741 to 0.779, right at the 0.78 level I set last time as an overfitting warning. Training loss kept falling slowly, from about 1.07 to 1.01, nearing its floor of about 0.85 under label smoothing. Gradient norms are steady to lower (total 2.5–5.1), and weight movement since update 100 keeps growing slowly (top encoder layer 9.5%). Time per update is 283 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Validation loss rising while training loss falls is an early overfitting sign, but it's a single reading on a 788-image set, and accuracy didn't drop. I'll act if validation loss is still rising (above about 0.80) at update 1596 and again at 1824. The first step would be stronger weight decay or a steeper per-layer learning-rate decay. Separately, I plan to add weight averaging for the last part of the run, starting around epoch 70. The updater would keep a running average of the weights and copy it into the model at update 11400. That usually adds a little and rarely hurts when the averaging window is short.

3. **Prediction:**
   - Validation 86–88% at update 1596, with validation loss about 0.75–0.80.
   - Training loss about 1.0 by update 1850.

   **What would prove me wrong:** validation loss above 0.80 with accuracy no higher than 86.5%. That would confirm overfitting, and I'd strengthen regularisation at the next decision.

## Decision 12 (update 1852, scheduled; changed: nothing)

1. **Observed:** Validation went up and then down: 87.18% at update 1596 (validation loss 0.736), then 85.53% at update 1824 (loss 0.775). The last four readings, epochs 10–16, all fall between 85.5% and 87.2%. Training loss has stalled at about 1.00, above its floor of about 0.85 under label smoothing. The backbone learning rate is still at 95% of its peak. Gradient norms and weight movement are steady. Time per update is 283 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. I read this as a plateau caused by the learning rate still being high, not as the model steadily overfitting. Validation loss moves up and down rather than trending upward, and training loss is still above its floor, so the model hasn't memorised the data. In fine-tuning, accuracy usually stays flat at a high learning rate and then rises as the rate decays. The cosine schedule only drops below about 50% of the peak after epoch 50. With 37 images per class, extra regularisation of the kind I can control would probably move accuracy by tenths of a point, not whole points.

   **Options I'm keeping in reserve:**
   - If validation is still at 87% or below by about epoch 30, I'll consider bringing the decay forward.
   - Tail weight averaging is still planned for around epoch 70: keep an average of recent weights and copy it into the model at the final update.

3. **Prediction:**
   - Validation stays in a band of about 85.5–88% through roughly epoch 30.
   - Accuracy starts climbing steadily once the learning rate falls below about 60% of peak, around epoch 55.
   - Validation of at least 89% by epoch 80.

   **What would prove me wrong:** validation loss rising steadily above about 0.82 across three readings. That would be progressive overfitting, and I'd respond with stronger regularisation or earlier decay.

## Decision 13 (update 2173, scheduled; changed: nothing)

1. **Observed:** Validation at update 2052 (epoch 18) is 87.69%, the best so far, though only about 4 images above the epoch-14 reading. Validation loss is 0.786, inside the 0.74–0.79 range it has moved in since epoch 10, with no steady upward trend. Training loss is flat at about 0.98–1.00. Total gradient norm fell from about 3.8 to about 2.1. The backbone learning rate is still at about 93% of its peak. Time per update is 283 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Accuracy is still inching up and validation loss isn't trending upward, which supports reading this as a plateau caused by the high learning rate. The main gains should come later, as the cosine decay lowers the rate. Tail weight averaging is still planned for around epoch 70.

3. **Prediction:**
   - Validation in a band of about 86.5–88.5% through about epoch 30.
   - Clear gains once the learning rate falls below about 60% of peak, after about epoch 55.

   **What would prove me wrong, and the response:**
   - Validation loss above about 0.82 for three readings would mean the model is overfitting; I'd respond with stronger regularisation or bringing the decay forward.
   - If validation is still at 87% or below at epoch 30, I'll consider bringing the decay forward.

## Decision 14 (update 2494, scheduled; changed: nothing)

1. **Observed:** Validation at update 2280 (epoch 20) is 87.44%, with validation loss 0.770. That sits in the 85.5–87.7% band seen since epoch 10, and within the 86.5–88.5% I predicted. Training loss is slowly falling, from about 0.98 to 0.97. Its floor under label smoothing is about 0.85. Gradient norms are steady (total 1.7–3.5), and weight movement keeps growing slowly. Time per update is 283 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`. Validation loss isn't trending upward, so the model isn't steadily overfitting. The learning rate is still at about 90% of its peak, and the decay of the cosine schedule should bring the gains. I'm keeping the reserve plan: if validation is still at 87.5% or below by about epoch 30, I'll consider bringing the decay forward.

   I noted one idea in `NOTES.md` for later: a way to measure the averaged weights directly. Validation runs right after updates that are multiples of 228. The updater could swap the averaged weights in at one of those updates and restore the trained weights at the next. That would show the averaged weights' validation accuracy before I commit to using them at the end. The cost is one batch whose gradients are computed at the averaged weights.

3. **Prediction:**
   - Validation stays at about 86.5–88.5% through about epoch 30.
   - Training loss slowly falls toward 0.95.
   - Clear validation gains once the learning rate falls below about 60% of its peak, after about epoch 55.

   **What would prove me wrong:** validation loss above 0.82 for three readings, which would mean overfitting.

## Decision 15 (update 2814, scheduled; changed: nothing)

1. **Observed:** Validation set new highs: 88.45% at update 2508 (epoch 22) and 88.83% at update 2736 (epoch 24). Both are above the 86.5–88.5% band I predicted, and about 1.1 points above the earlier best of 87.69%. Validation loss jumped to 0.816 and then came back to 0.755, so there's no upward trend. Training loss is flat at about 0.96–0.97, with its floor under label smoothing at about 0.85. Total gradient norm keeps shrinking, to about 1.5. The learning rate is at about 87% of its peak. Time per update is 283 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Accuracy is still rising at a high learning rate, and validation loss shows no steady overfitting. So my reserve plan to bring the decay forward around epoch 30 is no longer needed. Ending the high-learning-rate phase early would likely give up gains the run is still making. Weight averaging over the last part of the run is still planned, starting around epoch 70.

3. **Prediction:**
   - Validation about 88–89.5% through epoch 35.
   - Clearer gains to about 90% or more once the learning rate falls below about 60% of peak, from about epoch 55.
   - Training loss slowly falls toward about 0.93 by epoch 40.

   **What would prove me wrong:** validation loss above about 0.82 for three readings, with accuracy falling back below 87.5%. That would be overfitting, and I'd respond with stronger regularisation or bringing the decay forward.

## Decision 16 (update 3135, scheduled; changed: nothing)

1. **Observed:** Validation at update 2964 (epoch 26) fell back to 87.44% from 88.83% at epoch 24. That's 11 images on a 788-image set, inside the 85.5–88.8% band seen since epoch 14 (average about 87.6%). Validation loss is 0.803; it has gone back and forth between about 0.74 and 0.82 since epoch 10 with no clear trend. Training loss eased from about 0.96 to 0.95. Gradient norms vary between readings (total 1.5–3.3), and weight movement keeps growing slowly. The learning rate is at about 84% of its peak. Time per update is 283 ms, against 284 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. A one-reading drop within the noise band isn't evidence of overfitting, and validation loss shows no steady rise. This still looks like the plateau caused by the high learning rate, which should end as the cosine decay lowers the rate. I'm still planning to add tail weight averaging around epoch 70. I'll build it and install it a few decisions ahead of time so any error shows up early.

3. **Prediction:**
   - Validation keeps moving within about 87–89% through epoch 35.
   - It starts rising clearly from about epoch 50–55.
   - Validation of about 90% or more by epoch 80.

   **What would prove me wrong:** three consecutive validation losses above 0.82 with accuracy below 87.5%, or accuracy falling steadily. That would mean the model is overfitting, and I'd raise weight decay or bring the decay forward.

## Decision 17 (update 3456, scheduled; changed: analysis.py, updater.py)

1. **Observed:** Validation at epochs 28 and 30 is 87.94% and 87.69%. Since epoch 18 it has stayed between 87.4% and 88.8%, averaging about 87.9%, with no upward trend. Validation loss goes back and forth between about 0.72 and 0.80, so the model isn't steadily overfitting. Training loss eased from 0.95 to 0.94. The learning rate is at 80% of its peak. Time per update is 283 ms, against 284 ms for the starting program.

2. **Changed:** I installed version 2 of the updater, which tests whether the plateau comes from the noise of training at a high learning rate. I haven't run the new code; it takes over at the next update. If it fails during its first 20 updates, the run rolls back to version 1 automatically.
   - **Optimizer:** unchanged from version 1 — same learning rates, per-layer decay, weight decay and schedule. The training script's code confirms that validation runs right after the last update of each even epoch, which is what makes the probes below possible.
   - **Weight average:** from update 3460 the updater keeps an exponential moving average (EMA) of the weights. Its memory spans roughly the last 500 updates, about 4.4 epochs.
   - **Probes at updates 4104 and 4560 (ends of epochs 36 and 40):** the averaged weights replace the trained weights just before those validations, and the trained weights are restored at the next update. Those two validation results will measure the averaged weights, not the trained ones. The cost is one batch whose gradients are computed at the averaged weights, and the averaged weights are close to the trained ones.
   - **End of run:** the averaged weights are *not* copied into the model at the final update yet; whether to do that depends on the probes.
   - **`analysis.py`:** the reports now mark which validation results are probes. Probe results don't count toward the best validation score or the validation-drop trigger. The reports also show how far the averaged weights are from the trained ones.
   - **Cost:** about 0.7 GB more GPU memory and a few ms more per update.

3. **Prediction:**
   - Normal validation stays around 87.5–89% at epochs 32–34.
   - The epoch-36 probe reads about 89–90%, 1–2 points above the trained weights.

   **What the probes will decide:**
   - **Gain of at least 1 point:** high-learning-rate noise is limiting accuracy. I'll copy the averaged weights in at the final update and consider bringing the decay forward.
   - **Gain near zero:** noise isn't the limit, and I'll treat averaging as a minor end-of-run tweak.
   - **Probe below the trained weights:** the averaging window is too long for weights that are still moving. I'll shorten it.

## Decision 18 (update 3776, scheduled; changed: nothing)

1. **Observed:** Version 2 installed and runs without errors. Time per update is 285.4 ms, about 1.3 ms more than the starting program's 284.1 ms. The weight average started at update 3460 and has run 290 updates. Its distance from the trained weights grew from 0.6% to 1.19% of the weights' size and is levelling off as the average fills its roughly 500-update window.

   Validation at update 3648 (epoch 32) is 87.94%, inside the 87.4–88.8% band seen since epoch 18. Validation loss is 0.826, after 0.721 and 0.795 at epochs 28 and 30. That's a mild upward drift, but it has moved 0.72–0.82 before, so it isn't yet the sustained rise above 0.82 I set as the overfitting signal. Training loss eased from 0.947 to about 0.938. The learning rate is at about 77% of its peak.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. The probe at update 4104 is the evidence this plan is waiting for. It swaps the averaged weights into the model just before the epoch-36 validation. Changing the optimizer before then would muddle that measurement. The result will show up at the decision at update 4417.

3. **Prediction:**
   - Normal validation at update 3876 (epoch 34) stays at about 87.5–89%.
   - The epoch-36 probe of the averaged weights reads about 89–90%.

   **What would prove me wrong, and the response:**
   - If the probe is within about 0.5 points of the trained weights, noise from the high learning rate isn't what's holding accuracy back. Averaging would only be a minor end-of-run tweak.
   - If validation loss rises above 0.83 again at epoch 34, I'll treat it as early overfitting and weigh higher weight decay against bringing the decay forward.

## Decision 19 (update 4097, scheduled; changed: nothing)

1. **Observed:** Validation at update 3876 (epoch 34) is 88.96%, the best reading from the trained weights so far. It's only 1 image above the previous best of 88.83%, so it's still within the 87.4–89.0% band seen since epoch 18. Validation loss is 0.827, about the same as at epoch 32 (0.826), so it isn't rising further. Training loss eased from 0.938 to about 0.93. The weight average has a full window now (590 updates). Its distance from the trained weights has settled at about 1.3% of their size. Time per update is 285.6 ms, against 284.1 ms for the starting program. The learning rate is at about 74% of its peak.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. The first probe is 7 updates away, at update 4104. The averaged weights will be swapped in just before the epoch-36 validation, and changing anything now would spoil that measurement.

3. **Prediction:**
   - The epoch-36 probe of the averaged weights reads about 89.5–90.5%, roughly 1–2 points above the trained weights' recent level of about 88.
   - Normal validation at epoch 38 stays at about 88–89.5%.

   **What would prove me wrong, and the response:**
   - If the probe is at 88.5% or lower, noise from the high learning rate isn't what's limiting accuracy. I'd treat averaging as only a minor end-of-run tweak and let the cosine decay do the work.
   - If the gain is at least 1 point, I'll copy the averaged weights in at the final update and look at whether lowering the learning rate sooner would capture the same gain in the trained weights themselves.

## Decision 20 (update 4418, scheduled; changed: updater.py)

1. **Observed:** The probe worked. The averaged weights, swapped in just before the epoch-36 validation, scored **89.97% with validation loss 0.687**. The trained weights scored 88.96% (loss 0.827) at epoch 34 and 88.71% (loss 0.765) at epoch 38. That's about 1.0–1.3 points of accuracy, and about 0.1 lower validation loss, than the trained weights on either side. It also beats every earlier reading. That points to noise from the high learning rate holding the trained weights back. Training loss is flat at about 0.92–0.93. Time per update is 285.9 ms, against 284.1 ms for the starting program.

2. **Changed:** I installed version 3 of the updater. I haven't run it; it takes over at the next update and keeps the existing average.
   - **End of run:** the averaged weights will now be copied into the model at update 11400.
   - **Probes:** the averaged weights are now tested every 4 epochs (epochs 40, 44, …, 96). The trained weights are measured at epochs 42, 46, …, 98, plus the final epoch 100, which will show the averaged weights.
   - **Kept as is:** learning rates, schedule, and the roughly 500-update averaging window.

   Running both series lets me check two things. First, whether more time at a high learning rate still improves the averaged weights. Second, near the end, whether the averaged weights trail the trained ones once the learning rate approaches zero. If they do, I'll stop copying them in at the final update.

3. **Prediction:**
   - The epoch-40 probe reads about 89.5–90.5%, and the trained weights at epoch 42 about 88–89.5%.
   - The probes trend up slowly, reaching about 91% or more by epoch 80 as the learning rate decays.

   **What would prove me wrong, and the response:**
   - If the probes stay at about 90% through epoch 52 while the trained weights also stay flat, more time at a high learning rate isn't helping. I'd then bring the decay forward.
   - If a probe falls below the neighbouring trained-weight readings, the averaging window is too long for weights that are still moving, and I'd shorten it.

## Decision 21 (update 4738, scheduled; changed: nothing)

1. **Observed:** The epoch-40 probe of the averaged weights scored **89.47% with validation loss 0.677**. At epoch 36 the averaged weights scored 89.97% with loss 0.687. Accuracy is 4 images lower, within noise on a 788-image set, while loss improved slightly to its best value yet. Both probes sit about 0.5–1 point above the trained weights' best (88.96%) and about 0.1 lower in loss. So the advantage of the averaged weights holds. Training loss eased from about 0.93 to 0.92. The averaged weights stay about 1.3% away from the trained weights. The learning rate is at about 65% of its peak. Time per update is 285.6 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Version 3 runs as planned, and one probe after it isn't enough to see whether the averaged weights are still improving. I'll judge that at the epoch-44 and epoch-48 probes, before considering bringing the decay forward.

3. **Prediction:**
   - The trained weights at epoch 42 read about 88–89.5%.
   - The epoch-44 probe reads about 89.5–90.5% with loss at or below 0.68.
   - As the learning rate falls, both the trained and averaged weights rise, toward about 91% by epoch 80.

   **What would prove me wrong, and the response:**
   - If the probes stay at or below about 90% with loss no lower than 0.67 through epoch 52, more time at a high learning rate isn't helping, and I'd bring the decay forward.
   - If a probe drops below the trained-weight readings around it, I'd shorten the averaging window.

## Decision 22 (update 5059, scheduled; changed: nothing)

1. **Observed:** The trained weights scored **90.10%** at epoch 42 (validation loss 0.747), their first reading above 90 and over a point above their previous best of 88.96%. The epoch-44 probe of the averaged weights scored **90.23%** (loss 0.679), slightly above its earlier 89.97% and 89.47%. Both are rising as the learning rate falls to about 60% of its peak, as I predicted. The averaged weights' advantage is now mostly in loss (about 0.07 lower); in accuracy the two are within noise. Training loss eased from about 0.92 to 0.91. Gradient norms are steady to lower. Time per update is 286.2 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Both series are improving at the current pace of decay, so I have no evidence that bringing the decay forward would beat it. The averaged weights still give lower validation loss and are still set to be copied in at update 11400.

3. **Prediction:**
   - The trained weights at epoch 46 read about 89.5–91%.
   - The epoch-48 probe reads about 90–91% with loss of about 0.67.
   - Both rise toward about 91–92% by epoch 80.

   **What would prove me wrong, and the response:**
   - If the probes stall at or below 90.3% through epoch 56 while the trained weights stay flat, more time at a high learning rate isn't helping. I'd then bring the decay forward.
   - If a probe falls clearly below the neighbouring trained-weight readings, I'd shorten the averaging window.

## Decision 23 (update 5379, scheduled; changed: nothing)

1. **Observed:** The trained weights read **89.09%** at epoch 46 (validation loss 0.780), down from 90.10% at epoch 42. That's 8 images on a 788-image set, inside the noise the trained weights have shown throughout. Their validation loss keeps moving between about 0.75 and 0.83. The three probes of the averaged weights sit steadily at 0.68–0.69. Training loss is flat at about 0.91. The averaged weights stay about 1.2% away from the trained ones. Time per update is 285.7 ms, against 284.1 ms for the starting program. The learning rate is at about 56% of its peak.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. One noisy dip in the trained weights doesn't change the picture. The averaged weights are steadier and lower in loss than the trained ones, and they're what gets copied in at the end. The next probe, at update 5472 (epoch 48), shows whether they're still improving.

3. **Prediction:**
   - The epoch-48 probe reads about 90–91% with loss at or below 0.68.
   - The trained weights at epoch 50 read about 89–90.5%.
   - From about epoch 60, as the learning rate drops below about 40% of its peak, both rise toward about 91% or more by epoch 80.

   **What would prove me wrong, and the response:**
   - If the epoch-48 and epoch-52 probes don't beat 90.23% and their loss doesn't go below 0.67, the averaged weights have levelled off at this learning rate. I'd then consider bringing the decay forward.
   - If a probe falls clearly below the neighbouring trained-weight readings, I'd shorten the averaging window.

## Decision 24 (update 5700, scheduled; changed: nothing)

1. **Observed:** The epoch-48 probe of the averaged weights scored **90.48%** (validation loss 0.685), the best reading so far and within my predicted 90–91%. Across the four probes, accuracy has crept up: 89.97%, 89.47%, 90.23%, 90.48%. Their loss has stayed flat at about 0.68, slightly above the 0.68-or-lower I predicted. Training loss eased from about 0.91 to 0.90. The run is halfway through, with the learning rate at about 51% of its peak. Time per update is 285.8 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. The averaged weights are still improving slowly, and the stronger gains I expect come as the cosine schedule lowers the rate further over the next 50 epochs. Accuracy is still rising, so the case for bringing the decay forward is weak, and with half the run left the cosine will lower the rate anyway. The averaged weights stay set to be copied in at update 11400.

3. **Prediction:**
   - The trained weights at epoch 50 read about 89.5–90.5%.
   - The epoch-52 probe reads about 90.3–91%.
   - From about epoch 60, both series climb more clearly, toward about 91–92% by epoch 80, with the averaged weights' loss dropping below 0.67.

   **What would prove me wrong, and the response:**
   - If the probes stay at or below about 90.5% with loss at or above 0.68 through epoch 60, the run has levelled off despite the decay. I'd then lower the learning rate faster.
   - If a probe falls clearly below the neighbouring trained-weight readings, I'd shorten the averaging window.

## Decision 25 (update 6021, scheduled; changed: nothing)

1. **Observed:** The trained weights read 88.83% at epoch 50 (validation loss 0.812), below my predicted 89.5–90.5%. The epoch-52 probe of the averaged weights read **90.48%** (loss 0.696), the same accuracy as at epoch 48, with loss slightly higher. Their accuracy has held at about 90.2–90.5% over the last three probes. The averaged weights still lead the trained weights by about 1–1.5 points and about 0.1 in loss. Training loss eased from about 0.90 to about 0.90 at the end. The learning rate is at about 47% of its peak. Time per update is 286.0 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. The averaged weights have levelled off at about 90.4%, but the learning rate is still about half its peak. The biggest gains in a cosine schedule usually come in the low-rate phase, roughly epochs 60–90, which is still ahead. I don't have evidence for a specific change that would beat that; stronger weight decay or faster decay would both be guesses at this point. The averaged weights stay set to be copied in at update 11400.

3. **Prediction:**
   - The trained weights at epoch 54 read about 89–90.5%.
   - The epoch-56 probe reads about 90.3–91%.
   - From epoch 60 on, both series rise, reaching about 91–92% by epochs 76–80, with the averaged weights' loss dropping below about 0.67.

   **What would prove me wrong, and the response:** if the probes are still at or below about 90.5% at epochs 60 and 64, the low-rate phase isn't paying off as expected. I'd then look at regularisation, such as weight decay, rather than the schedule.

## Decision 26 (update 6341, scheduled; changed: nothing)

1. **Observed:** The trained weights read **88.83%** at epoch 54 (validation loss 0.786), the same as at epoch 50. Since epoch 42 the trained-weight readings are 90.10%, 89.09%, 88.83%, 88.83%: flat, perhaps slightly down. The averaged weights have held at 90.2–90.5% over the last three probes, still about 1.5 points ahead. There's no new probe this time; the next is at epoch 56. Training loss eased from about 0.90 to about 0.897. The averaged weights are about 1.0% away from the trained ones. The learning rate is at about 43% of its peak. Time per update is 285.7 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. The flat trained-weight readings are within the noise of a 788-image set. Their validation loss isn't climbing, so it isn't a clear overfitting signal. The averaged weights, which get copied in at update 11400, are holding steady. The low-rate phase, from about epoch 60, is where I expect the gains. Changing weight decay or the schedule now would be acting on noise.

3. **Prediction:**
   - The epoch-56 probe reads about 90.3–91%.
   - The trained weights at epoch 58 read about 89–90.5%.
   - From epoch 60 on, both series rise, toward about 91% or more by epoch 76–80.

   **What would prove me wrong, and the response:**
   - If the epoch-56 or epoch-60 probe falls below 90%, I'll consider raising weight decay to curb overfitting in the second half.
   - If the probes are still at or below about 90.5% at epoch 64, the low-rate phase isn't paying off as expected. I'd then look at regularisation, such as weight decay.

## Decision 27 (update 6662, scheduled; changed: nothing)

1. **Observed:** The epoch-56 probe of the averaged weights read **89.72%** (validation loss 0.701), the first probe below 90% since epoch 40. That missed my predicted 90.3–91% and reached the "below 90%" condition I set last time for considering higher weight decay. The trained weights read 88.83% at epoch 58 (loss 0.801), their third straight reading at that level.

   More telling is the averaged weights' validation loss, which has crept up steadily: 0.677 (epoch 40), 0.679, 0.685, 0.696, 0.701 (epoch 56). Over the same period training loss fell from 0.92 to 0.89. That's a mild sign the model is starting to memorise the training images. Accuracy has held at about 89.7–90.5% so far. The learning rate is at about 38% of its peak. Time per update is 286.0 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers.
   - **Weight decay:** I'd marked a probe below 90% as the cue to consider raising it, and decided against it. At these learning rates, decay that shrinks weights toward zero barely acts each step. To matter, it would have to be raised several-fold, and its effect on accuracy is a guess. The 0.3-point miss is within noise, and the rise in loss is small.
   - **End-of-run averaging:** still on. The averaged weights beat the trained ones at every comparison, by about 1 point and about 0.1 in loss.
   - **Schedule:** the low-rate phase over the remaining 42 epochs is where gains should come, if they come.

3. **Prediction:**
   - The epoch-60 probe reads about 89.5–90.8%, with loss about 0.70.
   - The trained weights at epoch 62 read about 88.5–90%.
   - As the learning rate falls below about 25% of its peak, from around epoch 70, the trained weights catch up toward the averaged ones. The final averaged weights land at about 90–91.5%.

   **What would prove me wrong, and the response:** if the epoch-60 and epoch-64 probes are both below 89.5% with loss above 0.71, overfitting is costing accuracy. I'd then raise weight decay strongly and/or shorten the averaging window, so the final weights reflect the low-rate phase.

## Decision 28 (update 6982, scheduled; changed: nothing)

1. **Observed:** The epoch-60 probe of the averaged weights read **90.36%** (validation loss 0.714), back up from 89.72% at epoch 56. That's inside my predicted 89.5–90.8%. Their loss continued its slow climb: 0.677 at epoch 40, 0.701 at epoch 56, now 0.714. Training loss eased from about 0.89 to about 0.885. So mild memorisation is still showing in the loss, but accuracy has held at about 89.7–90.5% for the last six probes. The learning rate is at about 34% of its peak. Time per update is 285.8 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. The accuracy recovery means the overfitting I set a trigger for last time isn't costing accuracy. That trigger was the epoch-60 and epoch-64 probes both below 89.5% with loss above 0.71. Only the loss half has happened. The averaged weights, which get copied in at update 11400, still lead the trained ones by about 1 point.

3. **Prediction:**
   - The trained weights at epoch 62 read about 88.5–90%.
   - The epoch-64 probe reads about 89.8–90.8%, with loss about 0.71–0.73.
   - From about epoch 70, as the learning rate falls below about 25% of its peak, the trained weights move closer to the averaged ones. The final averaged weights land at about 90–91.5%.

   **What would prove me wrong, and the response:** if the epoch-64 probe falls below 89.5% with loss above 0.72, overfitting is costing accuracy. I'd then raise weight decay strongly or shorten the averaging window.

## Decision 29 (update 7303, scheduled; changed: nothing)

1. **Observed:** The epoch-64 probe of the averaged weights read **90.61%** (validation loss 0.723), the best probe so far, within my predicted 89.8–90.8%. The trained weights read 89.97% at epoch 62 (loss 0.775), their best since epoch 42 and slightly closer to the averaged weights. The averaged weights' loss is still creeping up (0.714 at epoch 60, now 0.723), but accuracy keeps edging higher, so the mild memorisation isn't costing accuracy. Training loss is flat at about 0.885–0.89. The learning rate is at about 30% of its peak. Time per update is 285.7 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Both series are improving as the learning rate falls, as planned. The overfitting trigger I set needed both a probe below 89.5% and loss above 0.72. Only the loss half has happened, and accuracy is at its best. The averaged weights are still copied in at update 11400.

3. **Prediction:**
   - The trained weights at epoch 66 read about 89.5–90.8%.
   - The epoch-68 probe reads about 90.3–91.2%.
   - The final averaged weights at epoch 100 land at about 90.5–91.5%.

   **What would prove me wrong, and the response:**
   - If a probe falls below 89.5% with loss above 0.73, overfitting is costing accuracy. I'd consider shortening the averaging window or raising weight decay.
   - If the trained weights read above the probe next to them in the final epochs, the averaged weights are trailing the low-rate phase. I'd shorten the window before the end.

## Decision 30 (update 7624, scheduled; changed: nothing)

1. **Observed:** The trained weights read **89.47%** at epoch 66 (validation loss 0.786), inside my predicted 89.5–90.8% range only at its bottom edge. Since epoch 42 they've stayed in a band of 88.8–90.1%. The averaged weights have stayed at 89.7–90.6% over the same period (last probe 90.61% at epoch 64), still about 1 point ahead. Training loss is flat at about 0.88. The averaged weights are now about 0.63% away from the trained ones. That gap is shrinking because the learning rate, now at about 26% of its peak, takes smaller steps. Time per update is 285.8 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Nothing triggered a change. No probe has fallen below 89.5%, and the trained weights haven't overtaken the averaged ones. The averaged weights are still copied in at update 11400. The roughly 500-update averaging window is still short next to the remaining 3,800 updates, so it won't lag the final low-rate phase much.

3. **Prediction:**
   - The epoch-68 probe reads about 90.2–91.2%.
   - The trained weights at epoch 70 read about 89.3–90.6%.
   - The final averaged weights at epoch 100 land at about 90.5–91.5%.

   **What would prove me wrong, and the response:**
   - If the trained weights read above the neighbouring probe in two of the remaining comparisons, the averaged weights are trailing the low-rate phase. I'd shorten the window before the end.
   - If a probe drops below 89.5% with loss above 0.73, I'd consider stronger regularisation.

## Decision 31 (update 7944, scheduled; changed: nothing)

1. **Observed:** The epoch-68 probe of the averaged weights read **90.86%** (validation loss 0.728), the best probe so far, within my predicted 90.2–91.2%. Across the last three probes accuracy has climbed: 90.36% at epoch 60, 90.61% at epoch 64, 90.86% at epoch 68. The probes' validation loss is still creeping up (0.714, 0.723, 0.728). That's mild memorisation, but it isn't costing accuracy. Training loss is flat at about 0.88. The averaged weights are about 0.57% away from the trained ones, a gap that keeps shrinking as the learning rate (now about 22% of its peak) takes smaller steps. Time per update is 286.0 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Accuracy is rising as the learning rate falls, and none of my triggers fired. A probe hasn't dropped below 89.5%, and the trained weights haven't overtaken the averaged ones. The averaged weights are still copied in at update 11400.

3. **Prediction:**
   - The trained weights at epoch 70 read about 89.5–90.8%.
   - The epoch-72 probe reads about 90.5–91.3%.
   - The final averaged weights at epoch 100 land at about 90.8–91.8%.

   **What would prove me wrong, and the response:**
   - If the trained weights beat the neighbouring probe twice in the remaining comparisons, I'd shorten the averaging window before the end.
   - If a probe falls below 89.8%, I'd treat it as overfitting starting to cost accuracy and reconsider regularisation.

## Decision 32 (update 8265, scheduled; changed: nothing)

1. **Observed:** The trained weights read **89.97%** at epoch 70 (validation loss 0.780), within my predicted 89.5–90.8%. The epoch-72 probe of the averaged weights read **90.23%** (loss 0.734). That's 5 images below epoch 68's 90.86%, inside noise on a 788-image set, but below my predicted 90.5–91.3%. The averaged weights' lead over the trained ones has narrowed to roughly 0.3–0.9 points, from about 1–1.5 earlier. That's expected: as the learning rate falls (now about 18% of its peak), the trained weights jitter less. Their distance from the averaged weights fell to 0.50%. The averaged weights' validation loss is still creeping up (0.728 to 0.734). Training loss is flat at about 0.88. Time per update is 285.9 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Neither trigger fired. The trained weights haven't beaten the averaged ones at any comparison, and no probe has fallen below 89.8%. The averaged weights are still copied in at update 11400. The roughly 500-update window is still short against the 3,100 updates left, so it shouldn't trail the final low-rate phase much.

3. **Prediction:**
   - The trained weights at epoch 74 read about 89.5–90.8%.
   - The epoch-76 probe reads about 90.2–91.2%.
   - The final averaged weights at epoch 100 land at about 90.5–91.5%.

   **What would prove me wrong, and the response:** if the trained weights beat the neighbouring probe twice in the remaining comparisons, the averaged weights are trailing the low-rate phase. I'd then shorten the window before the end.

## Decision 33 (update 8586, scheduled; changed: nothing)

1. **Observed:** The trained weights read **89.72%** at epoch 74 (validation loss 0.774), within my predicted 89.5–90.8%. That's still below the last probe of the averaged weights (90.23% at epoch 72), so they haven't overtaken it. No probe ran in this window; the next is at epoch 76. Training loss is about 0.88, essentially flat. The averaged weights are now 0.42% away from the trained ones, down from 0.50%, as the learning rate (now about 15% of its peak) takes smaller steps. Time per update is 285.9 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Nothing new points to a change. The trained weights haven't beaten a neighbouring probe, the probes haven't dropped below 89.8%, and validation loss isn't climbing further. The averaged weights are still copied in at update 11400.

3. **Prediction:**
   - The epoch-76 probe reads about 90.2–91.2%.
   - The trained weights at epoch 78 read about 89.5–90.8%.
   - The final averaged weights at epoch 100 land at about 90.5–91.5%.

   **What would prove me wrong, and the response:** if the trained weights beat the neighbouring probe twice in the remaining comparisons, I'd shorten the averaging window before the end.

## Decision 34 (update 8906, scheduled; changed: nothing)

1. **Observed:** The epoch-76 probe of the averaged weights and the epoch-78 trained weights both read **89.97%**. That's a tie in accuracy, so the averaged weights' accuracy lead, about 1–1.5 points at epochs 36–52, has now gone. The averaged weights still have lower validation loss (0.742 against 0.759). The probe missed my predicted 90.2–91.2%, by 2 images. The trained weights were within my predicted 89.5–90.8%.

   The two are converging because the learning rate is down to about 12% of its peak. The averaged weights are now only 0.34% away from the trained ones, down from 0.42%. Training loss eased from about 0.88 to about 0.875. Time per update is 286.0 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. A tie doesn't meet my trigger for shortening the averaging window, which was the trained weights beating a neighbouring probe twice. The averaged weights still give lower validation loss. With so little gap left between them, copying the averaged weights in at update 11400 can cost very little and may still help slightly. With 22 epochs left and the learning rate low, a larger change would have little time to pay off.

3. **Prediction:**
   - The epoch-80 probe and epoch-82 trained weights both read about 89.5–90.8%.
   - The final averaged weights at epoch 100 land at about 90–91%, within about 0.3 points of what the trained weights alone would give.

   **What would prove me wrong, and the response:** if the trained weights beat the neighbouring probe in two of the remaining comparisons, I'd shorten the averaging window, or turn off the end-of-run copy.

## Decision 35 (update 9227, scheduled; changed: nothing)

1. **Observed:** The epoch-80 probe of the averaged weights read **90.23%** (validation loss 0.745), within my predicted 89.5–90.8%. That's 2 images above the trained weights' 89.97% at epoch 78, so the two are still essentially tied. Training loss eased from about 0.875 to about 0.873. The learning rate is at about 9% of its peak. As it falls, the averaged weights have closed to 0.28% of the trained weights' size, down from 0.34%. Time per update is 285.9 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Nothing triggered a change. The trained weights still haven't beaten a neighbouring probe, and the averaged weights match or slightly lead in both accuracy and loss. Copying the averaged weights in at update 11400 stays on. With about 19 epochs left at a low learning rate, changing the optimizer would have little time to pay off.

3. **Prediction:**
   - The trained weights at epoch 82 and the epoch-84 probe both read about 89.5–90.8%.
   - The final averaged weights at epoch 100 land at about 90–91%.

   **What would prove me wrong, and the response:** if the trained weights beat the neighbouring probe in two of the remaining comparisons, I'd shorten the averaging window or turn off the end-of-run copy.

## Decision 36 (update 9548, scheduled; changed: nothing)

1. **Observed:** The trained weights read **90.36%** at epoch 82 (validation loss 0.764), their best reading so far. That's 1 image above the epoch-80 probe of the averaged weights (90.23%, loss 0.745). It's the first time the trained weights have beaten a neighbouring probe, by a margin well inside noise on a 788-image set. The averaged weights still have the lower validation loss. Training loss eased from about 0.873 to about 0.871. The learning rate is at about 7% of its peak. The averaged weights have closed to 0.23% of the trained weights' size, down from 0.28%. Time per update is 285.9 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. My trigger for shortening the averaging window or turning off the end-of-run copy was the trained weights winning twice. A single 1-image win doesn't meet it. The two are now nearly identical, so copying the averaged weights in at update 11400 can cost almost nothing, while their lower validation loss suggests a slight edge. With 16 epochs left at a low learning rate, an optimizer change would have little time to pay off.

3. **Prediction:**
   - The epoch-84 probe reads about 89.8–90.8%.
   - The trained weights at epoch 86 read about 89.8–90.8%.
   - The final averaged weights at epoch 100 land at about 90–91%.

   **What would prove me wrong, and the response:** if the trained weights at epoch 86 beat the epoch-84 probe by 2 or more images, that's a second loss for the averaged weights. I'd then shorten the window before the end.

## Decision 37 (update 9868, scheduled; changed: updater.py)

1. **Observed:** The epoch-84 probe of the averaged weights read **90.10%** (validation loss 0.748). The trained weights then read **90.61%** at epoch 86 (loss 0.754), their best so far and 4 images above that probe. That's the second time the trained weights have beaten a neighbouring probe (the first was by 1 image at epoch 82). This meets the trigger I set last time.

   Each probe is 2 epochs earlier than the trained-weight reading it's compared with, and the model is still improving slowly. So part of the gap may simply be the averaged weights trailing recent progress. The averaged weights still have marginally lower loss. Training loss eased from about 0.871 to about 0.871. The learning rate is at about 5% of its peak. The averaged weights are now only 0.17% away from the trained ones. Time per update is 285.9 ms, against 284.1 ms for the starting program.

2. **Changed:** I installed version 4 of the updater, which shortens the averaging window from about 500 updates to about 200. That's 1.8 epochs instead of 4.4, so the averaged weights trail recent progress less. The averaged weights are still copied in at update 11400; a short average still smooths the last steps. Nothing else changed. I haven't run the new code; it takes over at the next update and keeps the existing average. With the two sets of weights this close, the effect either way should be a tenth of a point or less.

3. **Prediction:**
   - The epoch-88 probe reads about 90.2–91%.
   - The trained weights at epoch 90 read about 90–91%.
   - The final averaged weights at epoch 100 land at about 90.3–91.2%.

   **What would prove me wrong, and the response:** if the trained weights beat the shortened average by 3 or more images at the epoch-92/94 or epoch-96/98 comparisons, I'll turn off the end-of-run copy and let the final trained weights stand.

## Decision 38 (update 10189, scheduled; changed: nothing)

1. **Observed:** The epoch-88 probe of the averaged weights read **89.97%** (validation loss 0.752). That's below my predicted 90.2–91%, and 5 images below the trained weights' 90.61% at epoch 86. That would make a third loss for the averaged weights. But they are now only 0.07% away from the trained weights, down from 0.17% before the window was shortened. At that distance, the two give almost the same predictions. So a 5-image swing between epochs is noise from the epoch-to-epoch movement and the small validation set, not a real gap between them. Training loss eased from about 0.871 to about 0.870. The learning rate is at about 3% of its peak. Time per update is 285.6 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. Version 4's shorter window works as intended: the averaged weights now trail the trained ones by very little. At update 11400 the copied-in averaged weights will be essentially the final trained weights with a little smoothing. Turning the copy off or changing anything else would move the final result by much less than the validation noise.

3. **Prediction:**
   - The trained weights at epoch 90 and the epoch-92 probe both read about 89.8–90.8%.
   - The final result at epoch 100 lands at about 90–91% on validation.

   **What would prove me wrong, and the response:** if the trained weights beat the shortened average by 3 or more images at the epoch-92/94 or epoch-96/98 comparisons, I'll turn off the end-of-run copy. Given how close the two are, I don't expect that.

## Decision 39 (update 10509, scheduled; changed: nothing)

1. **Observed:** The trained weights read **89.85%** at epoch 90 (validation loss 0.763). The epoch-92 probe of the averaged weights read **89.97%** (loss 0.758). Both are within my predicted 89.8–90.8%, and the averaged weights are 1 image ahead with slightly lower loss. That was no trigger: I'd only drop the end-of-run copy if the trained weights won by 3 or more images. The averaged weights are now only 0.04% away from the trained weights. Training loss is flat at about 0.871. The learning rate is at about 2% of its peak. Validation has settled at about 90%: the readings since epoch 80 are 89.97–90.61%. Time per update is 285.8 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; `NOTES.md` now has this decision's numbers. With about 900 updates left at a very low learning rate, no optimizer change can move the result by more than validation noise. The averaged weights are still copied in at update 11400.

3. **Prediction:** the trained weights at epoch 94, the epoch-96 probe, and the final result at epoch 100 all read about 89.7–90.7% on validation. If anything clearly breaks that pattern, the one remaining decision can only toggle the end-of-run copy. Given how close the two sets of weights are, I don't expect to need it.

## Decision 40 (update 10830, scheduled; changed: nothing)

1. **Observed:** The trained weights read **90.10%** at epoch 94 (validation loss 0.759), within my predicted 89.7–90.7%. Since epoch 80, validation has stayed between 89.85% and 90.61%. Training loss is flat at about 0.87. The learning rate is under 1% of its peak, and the averaged weights are only 0.018% away from the trained ones, so the two are effectively the same. Time per update is 285.9 ms, against 284.1 ms for the starting program.

2. **Kept:** I didn't change `updater.py` or `analysis.py`; this is the last decision. With 570 updates left at nearly zero learning rate, no change could move accuracy by more than the validation noise. Version 4 stays in place.
   - One last probe of the averaged weights runs at update 10944 (epoch 96).
   - The averaged weights are copied into the model at update 11400; at this point that's essentially the final trained weights, lightly smoothed.

   **What changed over the run:**
   - **Optimizer:** AdamW with learning rates that shrink 0.85× per layer from the top (1e-4 at the top layer), a head learning rate 10× higher, and the original cosine schedule. This fixed the slow start: validation was 42.5% at epoch 2 and 84% by epoch 6.
   - **Averaged weights:** the probes showed they beat the trained weights by about 1–1.5 points while the learning rate was high. The advantage faded as the rate fell. I shortened the averaging window at update 9868 so the average wouldn't trail the final low-rate phase.

3. **Prediction:**
   - The final model, at update 11400, should score about 90% on validation, within roughly ±0.6, and similar on the test set.
   - The epoch-96 probe and the final epoch-100 reading should both fall in about 89.7–90.7%.

   No decisions remain, so I can't act on whatever those last readings show.
