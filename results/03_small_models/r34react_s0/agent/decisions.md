# Agent decisions: `r34react_s0`
The agent's reply at each decision point, as written. `changed` lists the programs it edited.

## Decision 1 (update 142, scheduled; changed: analysis.py, updater.py)

I replaced the starting optimizer with my own version that uses stronger regularization (version 1). I didn't run it before ending the turn; the run builds it and tries it out at the next update.

1. **Observed**: So far the run is still in warmup, so the numbers say little about the recipe. Training loss went from 5.31 to 5.24 over 142 updates, about the level of random guessing across 196 classes (5.28). Gradient norm fell from 1.52 to 1.35 and the weight norm stayed near 141. There are no validation results yet.

2. **Changed**:
   - **`updater.py`**: SGD with Nesterov momentum 0.9, rewritten with batched tensor operations. It picks up the starting optimizer's momentum where it left off.
     - Peak learning rate goes from 0.05 to **0.08**.
     - Weight decay goes from 5e-4 to **8e-4** on conv and fully connected weights, and stays at 0 on BatchNorm gains and biases. Learning rate × weight decay, which sets how strongly weights are pulled toward zero, is about 2.6× the starting recipe.
     - Warmup now ends at update 1140 instead of 570, then the rate decays on a cosine curve to 0 at update 22800. Both points are fixed update numbers, not offsets from now.
     - A clip on the total gradient norm at 5 acts as a safety net only. Today's gradient norm is about 1.4, so it shouldn't normally trigger.
     - The reason: with about 37 images per class and no pretraining, overfitting is the main risk, and stronger weight decay with a higher learning rate is the most likely way to gain several points.
   - **`analysis.py`**: now also reports weight norms per stage, the gradient norm before clipping, how often the clip fires, the current learning rate, validation results and time per update.
   - **`NOTES.md`**: records these beliefs, predictions and fallbacks.

3. **Prediction**:
   - **Stability**: no instability. The clip should fire on fewer than 2% of updates after warmup, and the loss should stay finite.
   - **Speed**: time per update should stay close to 58.6 ms.
   - **Accuracy**: training loss should fall more slowly in mid-run than the starting recipe would. Validation accuracy should be at least about 10% by epoch 20 (around update 2300).
   - **What would prove me wrong**: loss spikes or swings, validation accuracy below 5% by epoch 15, or the clip firing often. Then I would bring the learning rate back down to about 0.05.
   - **Later adjustments**: if training loss nears its floor with label smoothing (about 1.0) while validation accuracy lags far behind, I'll raise weight decay further. If the model is still clearly underfitting late in the run, I'll lower it.

## Decision 2 (update 285, scheduled; changed: nothing)

I kept both programs unchanged. The new optimizer (version 1) installed cleanly and the run is stable, but it is still warming up, so there's nothing yet to judge it on.

1. **Observed**:
   - **Loss**: training loss went from 5.25 to about 5.16 over the last 143 updates (the level of random guessing is 5.28). It's starting to fall, as expected while the learning rate is still ramping up.
   - **Gradients**: the gradient norm before clipping held steady at about 1.3 and the clip never fired.
   - **Learning rate**: now about 0.02, on its way to the 0.08 peak at update 1140.
   - **Weights**: weight norms are shrinking slowly in every stage. The total went from 140.6 to 139.4, which fits the stronger weight decay.
   - **Speed**: 59.0 ms per update against 58.6 ms for the starting program.
   - **Validation**: no results yet.

2. **Kept**: there's no evidence yet that would justify a change, and the first validation result comes at epoch 5 (around update 570). I added this decision's observations to `NOTES.md`.

3. **Prediction**:
   - **By update 570**: training loss below about 5.0 and validation accuracy above 1% (1/196 is 0.5%, i.e. chance level).
   - **Through the end of warmup**: the clip should stay at or near 0% as the learning rate climbs to 0.08.
   - **What would prove me wrong**: loss spikes or the clip firing often as the rate passes about 0.05. In that case I would lower the peak learning rate.

## Decision 3 (update 428, scheduled; changed: nothing)

I kept both programs unchanged again. The training loss has stalled around 5.15, but the learning rate is only at about a third of its peak, so it's too early to treat that as a problem.

1. **Observed**:
   - **Loss**: training loss has hovered around 5.10–5.17 over updates 285–427, with no clear downward trend in that window. Its spread from batch to batch is growing (standard deviation 0.04 → 0.085), which is often a sign that the model is starting to tell some classes apart.
   - **Gradients**: the gradient norm before clipping is about 1.26 and the clip never fired.
   - **Learning rate**: 0.03 by update 428, on its way to 0.08 at update 1140.
   - **Weights**: weight norms keep shrinking slowly in the conv layers (total 138.6 → 136.6), while the stem and classifier weights grow slightly.
   - **Speed**: 59.0 ms per update against 58.6 ms for the starting program.
   - **Validation**: no results yet.

2. **Kept**: models trained from scratch on fine-grained car images usually take a while before the loss starts to drop. Nothing points to instability or bad settings, so this doesn't justify a change yet.

3. **Prediction**:
   - **By about update 1000** (learning rate near its peak): training loss should break below 5.0 and keep falling.
   - **At the first validation, epoch 5 (around update 570)**: accuracy of roughly 1–2%, against 0.5% by chance.
   - **What would prove me wrong**: loss still above 5.0 around update 1000. I would then treat it as a slow-start problem and look at fixes such as a higher learning rate for the BatchNorm and classifier parameters. I've recorded this in `NOTES.md`.

## Decision 4 (update 570, scheduled; changed: nothing)

I kept both programs unchanged. Training loss is falling only slowly, but the run is stable and the learning rate is only halfway to its peak.

1. **Observed**:
   - **Loss**: training loss averaged about 5.11 over updates 428–569, down from about 5.15 in the previous stretch (5.28 is random guessing). It's falling, but only slowly.
   - **Gradients**: the gradient norm before clipping eased from 1.27 to 1.18, and the clip never fired.
   - **Learning rate**: 0.04 by update 550, halfway to the 0.08 peak at update 1140.
   - **Weights**: conv weight norms keep shrinking (total 135.4 → 132.7), while the stem and classifier weights grow slightly.
   - **Speed**: 59.0 ms per update against 58.6 ms for the starting program.
   - **Validation**: none yet. The first result, at epoch 5, should arrive just after this decision.

2. **Kept**: nothing points to instability or harm from the stronger regularization. A slow start is expected when training from scratch on fine-grained car images. The real check is whether the loss breaks down once the learning rate gets close to its peak. I added this decision's observations to `NOTES.md`.

3. **Prediction**:
   - **First validation (epoch 5)**: about 1–2% accuracy, against 0.5% by chance.
   - **By update 1000–1140**: training loss below 5.0.
   - **What would prove me wrong**: loss still above 5.0 around update 1100. I would then act on the slow start, for example with a higher learning rate for the BatchNorm and classifier parameters.

## Decision 5 (update 712, scheduled; changed: nothing)

I kept both programs unchanged. The first validation came in at 2.54%, above my 1–2% guess and about five times chance.

1. **Observed**:
   - **Validation**: at epoch 5 (update 570), accuracy was 2.54% with a validation loss of 5.11.
   - **Loss**: training loss averaged about 5.06 over updates 570–711, against 5.11 before. The latest block dipped to 5.00.
   - **Gradients**: the gradient norm before clipping kept easing (1.16 → 1.09), and the clip never fired.
   - **Learning rate**: 0.049 by update 700.
   - **Weights**: conv weight norms are shrinking at about 1.7 per 50 updates (total 131.1 → 127.8). That's the stronger weight decay at work, with nothing pointing to instability.
   - **Speed**: 59.0 ms per update against 58.6 ms for the starting program.

2. **Kept**: the first validation result says the model is learning. A slow start is normal when training from scratch on fine-grained car images. The settings haven't been tested yet at full strength, since the learning rate reaches its peak only at update 1140. I added this decision's observations to `NOTES.md`.

3. **Prediction**:
   - **Training loss**: stays below 5.0 consistently by about update 1000.
   - **Validation**: about 4–6% at epoch 10 (update 1140).
   - **Clip**: stays at or near 0% through the peak learning rate.
   - **What would prove me wrong**: loss spikes or the clip firing often near the peak would mean 0.08 is too high. A flat loss around 5.05 at epoch 10 would mean the slow start needs fixing.

## Decision 6 (update 855, scheduled; changed: nothing)

I kept both programs unchanged. Training loss has started to cross below 5.0, slightly ahead of my "by about update 1000" prediction, and the run is stable as the learning rate keeps rising.

1. **Observed**:
   - **Loss**: training loss went from about 5.03 in updates 712–797 to 4.96–5.01 in updates 798–854.
   - **Gradients**: the gradient norm before clipping is about 1.02 and the clip never fired.
   - **Learning rate**: 0.06 by update 850, still rising toward the 0.08 peak at update 1140.
   - **Weights**: total weight norm went from 126.0 to 122.2 over the window. The conv weights shrink by about 1.8 per 50 updates, while the classifier weights grow (9.26 → 9.71).
     - Because BatchNorm follows each conv, smaller conv weights make each step change them more, so this works like a higher learning rate for those layers.
   - **Speed**: 59.0 ms per update against 58.6 ms for the starting program.
   - **Validation**: no new results since 2.54% at epoch 5.

2. **Kept**: progress matches the expected slow start, and nothing points to instability. The settings will only be fully tested at the peak learning rate, so changing them now would be premature. I noted in `NOTES.md` to watch whether the weight norms level off rather than keep falling.

3. **Prediction**:
   - **Validation at epoch 10 (update 1140)**: about 4–6%.
   - **Training loss**: settles around or below 4.9 by update 1140.
   - **Clip**: stays at or near 0% through the peak learning rate.
   - **What would prove me wrong**: loss spikes or the clip firing often near the peak. Those would point to a learning rate that's too high.

## Decision 7 (update 998, scheduled; changed: nothing)

I kept both programs unchanged. Training loss is slowly getting lower and the run is stable at a learning rate of 0.067. The first real test comes at the next decision, when the learning rate peaks and the epoch-10 validation result arrives.

1. **Observed**:
   - **Loss**: training loss was about 5.02–5.05 in updates 855–911 and about 4.91–4.98 in updates 912–997. It is now consistently below 5.0, as I predicted.
   - **Gradients**: the gradient norm before clipping is about 0.99 and the clip never fired.
   - **Learning rate**: 0.067 by update 950, close to the 0.08 peak at update 1140.
   - **Weights**: total weight norm keeps falling (120.2 → 118.2) as the conv weights shrink. The classifier weights grow (9.91 → 10.22).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.
   - **Validation**: no new results since 2.54% at epoch 5.

2. **Kept**: the trend is steady and stable, and this is a normal pace for training from scratch on fine-grained car images. I want to see the epoch-10 validation result before considering any change. I added this decision's observations to `NOTES.md`.

3. **Prediction**:
   - **Validation at epoch 10 (update 1140)**: about 4–6%.
   - **Training loss**: settles around 4.9 by update 1140.
   - **Clip**: stays at or near 0% as the learning rate reaches 0.08.
   - **What would prove me wrong**: validation accuracy below 3.5% at epoch 10 would mean learning is too slow. I would then rethink the settings, for example lowering weight decay for now. Loss spikes near the peak would mean the learning rate is too high.

## Decision 8 (update 1140, scheduled; changed: analysis.py)

I kept the optimizer unchanged and added a safety alarm to `analysis.py`, because the next scheduled decision is 641 updates away (at update 1781). The learning rate has now reached its 0.08 peak with no instability. The epoch-10 validation result hasn't reached me yet, though.

1. **Observed**:
   - **Loss**: training loss averaged about 4.93 over updates 998–1139 (ranging 4.87–4.99), down from about 4.96 before.
   - **Gradients**: the gradient norm before clipping is about 1.0 and the clip never fired.
   - **Learning rate**: 0.077 at update 1100, reaching its 0.08 peak at update 1140.
   - **Weights**: total weight norm keeps falling (116.2 → 112.0). The layer3 conv weights have shrunk from 78 at update 150 to 56. Since BatchNorm follows each conv, that makes each step change those weights more, like a higher learning rate for them.
   - **Speed**: 59.0 ms per update against 58.6 ms for the starting program.
   - **Validation**: no result since 2.54% at epoch 5. The epoch-10 result should arrive just after this decision.

2. **Changed**:
   - **`updater.py`**: unchanged. Nothing argues against the current settings yet.
   - **`analysis.py`**: now asks for an extra decision if anything goes wrong in the long gap before update 1781. It does so if the loss becomes non-finite, rises 0.4 above the best 50-update average so far, or the clip fires on more than 30% of updates.
   - **`NOTES.md`**: records this decision and the predictions below.

3. **Prediction**:
   - **Validation**: about 4–6% at epoch 10 and about 8–12% at epoch 15 (update 1710), which I'll see at update 1781.
   - **Training loss**: about 4.6 by update 1781.
   - **Alarm**: should not fire.
   - **What would prove me wrong**: validation below 6% at epoch 15 would mean the model is learning too slowly. In that case I'll cut weight decay back to 5e-4.

## Decision 9 (update 1781, scheduled; changed: nothing)

I kept both programs unchanged, even though validation came in below my prediction and just under the 6% level where I said I would cut weight decay. The training loss has started falling fast, so I'm giving the current settings one more checkpoint.

1. **Observed**:
   - **Validation**: 3.93% at epoch 10 and 5.46% at epoch 15. I predicted 8–12% for epoch 15, so learning is slower than I expected.
   - **Loss**: training loss fell from 4.89 (updates 1140–1203) to 4.48 (updates 1717–1780). The drop is speeding up, now about 0.05–0.09 per 64 updates.
   - **Gradients**: the gradient norm before clipping rose from 1.0 to 1.45, as you'd expect once the model starts fitting. The clip never fired, and the alarm didn't trigger.
   - **Weights**: conv weight norms are still falling (layer3 52 → 38), so they haven't yet settled where weight decay and gradients balance out. The classifier weights keep growing (11.6 → 14.8).
   - **Speed**: 59.1 ms per update against 58.6 ms for the starting program.

2. **Kept**: I chose not to apply my own "cut weight decay below 6%" rule.
   - 5.46% is only just under the threshold.
   - The training loss is clearly taking off now.
   - In a 200-epoch run, how fast the model learns early matters less for the final accuracy than how well it is regularized.

   That leaves me with an unchecked judgement, so I set a firmer test in `NOTES.md`: if epoch-20 validation is below 8%, I cut weight decay to 5e-4.

3. **Prediction**:
   - **Validation**: at least 9% at epoch 20 (update 2280) and at least 14% at epoch 25.
   - **Training loss**: about 4.1 by update 2400.
   - **What would prove me wrong**: epoch-20 validation below 8%. Weight decay then comes down to 5e-4.

## Decision 10 (update 2422, scheduled; changed: nothing)

I kept both programs unchanged. Validation at epoch 20 came in at 9.39%, which meets the at-least-9% check from last time, so I'm not cutting weight decay.

1. **Observed**:
   - **Validation**: 9.39% at epoch 20, with validation loss 4.48.
   - **Loss**: training loss fell from 4.43 to 3.86 over the last 641 updates, about 0.06 per 64 updates. It is still falling steadily.
   - **Gradients**: the gradient norm before clipping rose from about 1.5 to 1.95. The clip (set at 5) never fired, and the alarm didn't trigger.
   - **Weights**: total weight norm is levelling off (90.4 → 84.8, and now falling by only about 0.3 per 50 updates). Weight decay and gradients seem close to balancing out. The layer4 weights are still shrinking a little and the classifier weights are still growing (15.4 → 18.4).
   - **Speed**: 59.1 ms per update against 58.6 ms for the starting program.

2. **Kept**: the model is now in the fast-learning phase. The settings passed the check I set, and nothing points to too much regularization or to instability. Weight decay stays at 8e-4 and the peak learning rate at 0.08. I added this decision's results and new checks to `NOTES.md`.

3. **Prediction**:
   - **Validation**: at least 14% at epoch 25 (update 2850) and at least 20% at epoch 30 (update 3420).
   - **Training loss**: about 3.4 by update 3064.
   - **What would prove me wrong**: validation stalling near 10% while training loss keeps falling, which would mean early overfitting.
     - I would then raise regularization, not lower it.
     - If instead the training loss stalls, the settings may be too strong and I would revisit weight decay.

## Decision 11 (update 3064, scheduled; changed: nothing)

I kept both programs unchanged. Validation jumped to 19.04% at epoch 25, well above the at-least-14% check, and training is speeding up.

1. **Observed**:
   - **Validation**: 19.04% at epoch 25 (update 2850), up from 9.39% at epoch 20. Validation loss fell to 3.65.
   - **Loss**: training loss fell from 3.80 to 3.10 over the last 642 updates, about 0.07 per 64 updates. That's slightly faster than before.
   - **Gradients**: the gradient norm before clipping rose from about 2.0 to 2.4. The clip (set at 5) never fired, and the alarm didn't trigger.
   - **Weights**: total weight norm has levelled off at about 84.3, so weight decay and gradients now balance out. Within that, layer4 is still shrinking slowly (25.2 → 22.0) and the classifier is still growing (19.0 → 21.3).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: progress is strong and stable, and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the learning rate follows the same schedule. I recorded the new checks in `NOTES.md`. That includes a rule for later: if training loss falls below 1.5 by about epoch 80 while validation accuracy plateaus, the model is overfitting and I'll raise weight decay.

3. **Prediction**:
   - **Validation**: at least 25% at epoch 30 (update 3420) and at least 30% at epoch 35.
   - **Training loss**: about 2.6 by update 3705.
   - **Clip**: stays at 0%.
   - **What would prove me wrong**: validation flattening below 25% while training loss keeps falling fast. That would mean the model is starting to overfit early.

## Decision 12 (update 3705, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached 34.9% at epoch 30, well above the at-least-25% check, and the model isn't overfitting yet.

1. **Observed**:
   - **Validation**: 34.9% at epoch 30 (update 3420), up from 19.04% at epoch 25. Validation loss fell from 3.65 to 2.77.
   - **Loss**: training loss fell from 2.98 to 2.44 over the last 641 updates, about 0.05 per 64 updates.
   - **Overfitting**: validation loss is about the same as the training loss of that period. Validation uses centre crops and no label smoothing, while training uses random crops and smoothing. So there's no sign of the model memorizing the training images yet.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.4 and the clip never fired. Total weight norm is creeping up slowly (84.8 → 86.3).
   - **Speed**: 59.3 ms per update against 58.6 ms for the starting program.

2. **Kept**: progress is strong and every check has passed since epoch 20. Weight decay stays at 8e-4 and the learning rate follows the same schedule. I added this decision's results and the next checks to `NOTES.md`.

3. **Prediction**:
   - **Validation**: at least 40% at epoch 35 (update 3990) and at least 45% at epoch 40 (update 4560).
   - **Training loss**: about 2.0 by update 4346.
   - **What would prove me wrong**: validation flattening while training loss keeps falling fast. That would mean early overfitting, and I would raise regularization.

## Decision 13 (update 4346, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached 42.51% at epoch 35, which passes the at-least-40% check, and the model still isn't overfitting.

1. **Observed**:
   - **Validation**: 42.51% at epoch 35 (update 3990), up from 34.9% at epoch 30. Validation loss fell from 2.77 to 2.32.
   - **Loss**: training loss fell from about 2.50 to about 2.15 over the last 641 updates. That's about 0.035 per 64 updates, slower than the 0.05 before. A slowdown is normal at this stage.
   - **Overfitting**: validation loss (2.32) is still at about the training loss. Training uses random crops and label smoothing while validation doesn't, so there's no gap that would point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.45 and the clip never fired. Total weight norm keeps creeping up (86.4 → 87.8).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: every check has passed since epoch 20, and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the learning rate follows the same schedule, now at 0.076 and decaying slowly. I added this decision's results and the next checks to `NOTES.md`.

3. **Prediction**:
   - **Validation**: at least 47% at epoch 40 (update 4560) and at least 51% at epoch 45 (update 5130).
   - **Training loss**: about 1.85 by update 4988.
   - **What would prove me wrong**: validation flattening while training loss keeps falling, which would mean the model is starting to overfit and I'd raise regularization. Training loss stalling well above 2.0 would instead suggest the settings are too strong.

## Decision 14 (update 4988, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached 51.02% at epoch 40, which passes the at-least-47% check, and the model still isn't overfitting.

1. **Observed**:
   - **Validation**: 51.02% at epoch 40 (update 4560), up from 42.51% at epoch 35. Validation loss fell from 2.32 to 2.01.
   - **Loss**: training loss fell from about 2.12 to about 1.95 over the last 642 updates, about 0.025 per 64 updates. It's still slowing, as you'd expect at this point.
   - **Overfitting**: validation loss (2.01) is still about level with the training loss. Training uses random crops and label smoothing while validation doesn't, so there's no gap that would point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.45 and the clip never fired. Total weight norm keeps creeping up (88.0 → 89.1).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: every check has passed since epoch 20, and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the learning rate follows the same schedule, now at 0.074 and decaying slowly. I added this decision's results and the next checks to `NOTES.md`.

3. **Prediction**:
   - **Validation**: at least 55% at epoch 45 (update 5130) and at least 58% at epoch 50 (update 5700).
   - **Training loss**: about 1.8 by update 5629.
   - **What would prove me wrong**: validation flattening while training loss keeps falling, which would mean the model is starting to overfit and I'd raise regularization. Training loss stalling around 1.95 would instead suggest the settings are too strong.

## Decision 15 (update 5629, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached 56.6% at epoch 45, which passes the at-least-55% check, and the model still isn't overfitting.

1. **Observed**:
   - **Validation**: 56.6% at epoch 45 (update 5130), up from 51.02% at epoch 40. Validation loss fell from 2.01 to 1.82.
   - **Loss**: training loss fell from about 1.92 to about 1.80 over the last 641 updates. That's about 0.02 per 64 updates, so the drop keeps slowing.
   - **Overfitting**: validation loss (1.82) is still about level with the training loss. Training uses random crops and label smoothing while validation doesn't, so there's no gap that would point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.5 and the clip never fired. Total weight norm keeps creeping up (89.3 → 90.1).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: every check has passed since epoch 20, and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the learning rate follows the same schedule, now at 0.072 and decaying slowly. I added this decision's results and the next checks to `NOTES.md`.

3. **Prediction**:
   - **Validation**: at least 59% at epoch 50 (update 5700) and at least 61% at epoch 55 (update 6270).
   - **Training loss**: about 1.72 by update 6270.
   - **What would prove me wrong**: validation flattening while training loss keeps falling, which would mean the model is starting to overfit and I'd raise regularization. Training loss stalling around 1.8 would instead suggest the settings are too strong.

## Decision 16 (update 6270, scheduled; changed: nothing)

I kept both programs unchanged. Validation came in at 55.96% at epoch 50, which missed my at-least-59% prediction. But that is flat against 56.6% at epoch 35, well within the noise of a 788-image validation set, and nothing points to overfitting.

1. **Observed**:
   - **Validation**: 55.96% at epoch 50, against 56.6% at epoch 45. Validation loss was also flat (1.82 → 1.81). On 788 images, one standard deviation of noise is roughly 1.8 points, so this is no real change.
   - **Loss**: training loss is still falling slowly, from about 1.82 to about 1.71 over the last 641 updates.
   - **Overfitting**: training loss (about 1.7) and validation loss (1.81) are still close. Training uses random crops and label smoothing while validation doesn't, so there's no gap that would point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 2.45 and the clip never fired. Total weight norm keeps creeping up slowly (90.2 → 90.8).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: my reading is that this is the limit of what the model can show while the learning rate is still high (0.0695), not overfitting.
   - With cosine decay, most of the gain in validation accuracy usually comes as the learning rate drops, and there are about 16,500 updates of decay left.
   - Training loss is still improving and there's no gap between training and validation loss, so raising weight decay now would be premature.
   - In `NOTES.md` I set a rule for the next decisions. If validation stays flat below 58% through epoch 65 while training loss keeps dropping, that would mean overfitting and I'd raise weight decay. If training loss also flattens, it's the high learning rate holding the model back, which is expected and fine.

3. **Prediction**:
   - **Validation**: at least 57% at epoch 55 (update 6270, which I'll see next time) and at least 59% at epoch 60.
   - **Training loss**: about 1.65 by update 6911.
   - **What would prove me wrong**: validation stuck below 58% at epoch 60 and 65 while training loss keeps falling. That would point to overfitting rather than learning-rate noise.

## Decision 17 (update 6911, scheduled; changed: nothing)

I kept both programs unchanged. Validation accuracy has been stuck around 56% for three checks, but validation loss reached a new best, so this still looks like the effect of the high learning rate rather than overfitting.

1. **Observed**:
   - **Validation**:
     - Epoch 55: dipped to 51.27%, with validation loss jumping to 2.27.
     - Epoch 60: back to 55.71%, with validation loss 1.75, the lowest so far.
     - Validation accuracy has now sat around 56% from epoch 45 to 60, while validation loss keeps trending down (1.82 → 1.81 → [2.27] → 1.75).
   - **Loss**: training loss is still falling, from about 1.65 to about 1.58 over the last 641 updates.
   - **Overfitting**: a modest gap is starting to show. Training loss (about 1.6) includes label smoothing, which adds to the loss. That means the plain cross-entropy on training batches is probably below the 1.75 validation loss.
   - **Gradients and weights**: the gradient norm before clipping is about 2.45 and the clip never fired. Total weight norm is flat at about 91.2.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**:
   - The epoch-55 dip looks like a one-off swing from the high learning rate, because epoch 60 fully recovered. I'm not reading it as a trend.
   - The learning rate is still 0.067. With cosine decay, most of the gain in validation accuracy usually comes once the rate drops well below its peak, roughly after epoch 100.
   - I'll raise weight decay only if validation loss starts rising while training loss keeps falling, which would be clear overfitting. That hasn't happened. I recorded this rule and the next checks in `NOTES.md`.

3. **Prediction**:
   - **Validation**: at least 56% at epoch 65 (update 7410) and at least 58% at epoch 70 (update 7980).
   - **Validation loss**: stays at or below about 1.8.
   - **Training loss**: about 1.5 by update 7552.
   - **What would prove me wrong**: validation loss rising above 1.9 at both epoch 65 and epoch 70 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 18 (update 7552, scheduled; changed: nothing)

I kept both programs unchanged. Validation broke out of its plateau, reaching 61.8% at epoch 65 with the lowest validation loss so far. That confirms last time's reading: the stall came from the high learning rate, not overfitting.

1. **Observed**:
   - **Validation**: 61.8% at epoch 65 (update 7410), up from about 56% over epochs 45–60. Validation loss fell to 1.597, its lowest so far. That's well above my at-least-56% prediction.
   - **Loss**: training loss went from about 1.58 to about 1.52 over the last 641 updates, so it's still falling slowly.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.45 and the clip never fired. Total weight norm is flat at about 91.6.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: validation accuracy and validation loss both improved, and training loss is still falling. Nothing points to overfitting or instability. Weight decay stays at 8e-4 and the learning rate follows the same cosine schedule, now at 0.064. I added this decision's results and the next checks to `NOTES.md`.

3. **Prediction**:
   - **Validation**: at least 61% at epoch 70 (update 7980) and at least 63% at epoch 75 (update 8550).
   - **Training loss**: about 1.47 by update 8194.
   - **What would prove me wrong**: validation loss rising back above about 1.8 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 19 (update 8194, scheduled; changed: nothing)

I kept both programs unchanged. Validation came in at 59.26% at epoch 70, which missed my at-least-61% prediction. That's a drop from 61.8% at epoch 65, but within the noise of a 788-image validation set, and training is still improving.

1. **Observed**:
   - **Validation**: 59.26% at epoch 70, with validation loss 1.663, against 61.8% and 1.597 at epoch 65. On 788 images, one standard deviation of noise is roughly 1.8 points, so the epoch-60–70 readings (55.7, 61.8, 59.3) average about 59%.
   - **Loss**: training loss went from about 1.53 to about 1.48 over the last 642 updates, so it's still falling slowly.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.45 and the clip never fired. Total weight norm is flat at about 91.9.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: one noisy reading below the last one isn't evidence of a problem. Validation loss (1.66) is still well below its earlier level of about 1.8, and training loss is still falling without instability. The learning rate is still high (0.061). I expect the clearer gains once it drops below about 0.05, around epoch 95. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages about 61–63% over epochs 75 and 80.
   - **Training loss**: about 1.43 by update 8835.
   - **What would prove me wrong**: validation loss rising above about 1.8 at both epoch 75 and epoch 80 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 20 (update 8835, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached 65.99% at epoch 75 with the lowest validation loss so far, above the 61–63% I predicted, and the run is stable.

1. **Observed**:
   - **Validation**: 65.99% at epoch 75 (update 8550), against 59.26% at epoch 70. Validation loss fell to 1.400, from 1.663.
   - **Loss**: training loss went from about 1.48 to about 1.43 over the last 641 updates, so it's still falling slowly.
   - **Overfitting**: validation loss (1.40) is about level with training loss (about 1.43), so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.5 and the clip never fired. Total weight norm is flat at about 92.0.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: validation accuracy and validation loss both reached new bests, and nothing points to overfitting or instability. Weight decay stays at 8e-4 and the learning rate follows the same cosine schedule, now at 0.058. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 65% over epochs 80 and 85.
   - **Training loss**: about 1.40 by update 9476.
   - **What would prove me wrong**: validation loss rising back above about 1.6 at both epoch 80 and epoch 85 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 21 (update 9476, scheduled; changed: nothing)

I kept both programs unchanged. Validation held at 65.86% at epoch 80, and validation loss reached a new best. Training loss has levelled off at about 1.41, but that looks like the limit of the current learning rate rather than a problem.

1. **Observed**:
   - **Validation**: 65.86% at epoch 80 (update 9120), against 65.99% at epoch 75. Validation loss fell slightly to 1.387, its lowest so far.
   - **Loss**: training loss has levelled off at about 1.41, ranging 1.39–1.43 over the last 641 updates, down from about 1.44 before.
   - **Overfitting**: validation loss (1.39) is still about level with training loss, so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.5 and the clip never fired. Total weight norm is flat at about 92.2.
   - **Speed**: 59.3 ms per update against 58.6 ms for the starting program.

2. **Kept**: the flat training loss is most likely the noise limit of a learning rate that's still 0.055. Cosine decay should lower that limit steadily from here. Validation accuracy is holding at its best level and validation loss is still improving, so nothing calls for more or less regularization. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 66% over epochs 85 and 90.
   - **Training loss**: starts falling again as the learning rate decays, to about 1.37 by update 10118.
   - **What would prove me wrong**: validation loss rising above about 1.6 while training loss keeps falling, which would point to overfitting and I'd raise weight decay. Training loss still flat at 1.41 by update 10118 would instead mean it isn't just the learning rate holding it back.

## Decision 22 (update 10118, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 67.89% at epoch 85, and training loss started falling again as the learning rate decays, as I predicted last time.

1. **Observed**:
   - **Validation**: 67.89% at epoch 85 (update 9690), against 65.86% at epoch 80. Validation loss fell to 1.288, its lowest so far.
   - **Loss**: training loss went from about 1.41 to about 1.34 over the last 642 updates.
   - **Overfitting**: validation loss (1.29) is still close to training loss, so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.4 and the clip never fired. Total weight norm is flat at about 92.1.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: the run is doing what I expected. Training loss and validation results are improving as the learning rate (now 0.051) comes down, and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next checks to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 68% over epochs 90 and 95.
   - **Training loss**: about 1.29 by update 10759.
   - **What would prove me wrong**: validation loss rising above about 1.5 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 23 (update 10759, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached 68.91% at epoch 90, a new best, and training loss keeps falling as the learning rate decays.

1. **Observed**:
   - **Validation**: 68.91% at epoch 90 (update 10260), up from 67.89% at epoch 85. Validation loss was 1.321, slightly above the 1.288 from epoch 85, which is within noise.
   - **Loss**: training loss went from about 1.35 to about 1.31 over the last 641 updates, so it's still falling slowly.
   - **Overfitting**: validation loss (1.32) is about level with training loss (1.31), so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.45 and the clip never fired. Total weight norm is flat at about 92.0.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: the run is steadily improving as the learning rate comes down (now 0.047), and nothing points to too much or too little regularization. The cosine decay to 0 over the remaining 12,000 updates should deliver most of the remaining gain, so I'm leaving weight decay at 8e-4 and the schedule unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 69% over epochs 95 and 100.
   - **Training loss**: about 1.27 by update 11400.
   - **What would prove me wrong**: validation loss rising above about 1.5 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 24 (update 11400, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 70.3% at epoch 95, and the run is now halfway through its 22,800 updates.

1. **Observed**:
   - **Validation**: 70.3% at epoch 95 (update 10830), up from 68.91% at epoch 90. Validation loss was 1.33, about the same as the 1.32 at epoch 90.
   - **Loss**: training loss went from about 1.31 to about 1.29 over the last 641 updates, so it's still falling slowly.
   - **Overfitting**: validation loss (1.33) is about level with training loss, so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.5 and the clip never fired. Total weight norm is flat at about 91.9.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: validation keeps climbing as the learning rate comes down (now 0.043), and nothing points to too much or too little regularization. The remaining 11,400 updates of cosine decay to 0 should carry most of the remaining gain, so weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 70% over epochs 100 and 105.
   - **Training loss**: about 1.25 by update 12041.
   - **What would prove me wrong**: validation loss rising above about 1.5 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 25 (update 12041, scheduled; changed: nothing)

I kept both programs unchanged. Validation averaged 72.1% over epochs 100 and 105, passing the at-least-70% check, and training loss keeps falling as the learning rate decays.

1. **Observed**:
   - **Validation**:
     - Epoch 100: 73.35%, validation loss 1.251 (the lowest so far).
     - Epoch 105: 70.81%, validation loss 1.291.
     - The two average 72.1%. The drop between them is within the noise of a 788-image validation set (roughly 1.8 points).
   - **Loss**: training loss went from about 1.29 to about 1.23 over the last 641 updates.
   - **Overfitting**: validation loss (1.25–1.29) is still about level with training loss, so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 2.4 and the clip never fired. Total weight norm is slowly easing (91.85 → 91.47).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: the run keeps improving as the learning rate comes down (now 0.040), and nothing points to too much or too little regularization. The remaining 10,800 updates of cosine decay to 0 should carry most of the remaining gain, so weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 72.5% over epochs 110 and 115.
   - **Training loss**: about 1.19 by update 12682.
   - **What would prove me wrong**: validation loss rising above about 1.45 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 26 (update 12682, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 75.63% at epoch 110, with the lowest validation loss so far.

1. **Observed**:
   - **Validation**: 75.63% at epoch 110 (update 12540), against 70.81% at epoch 105 and 73.35% at epoch 100. Validation loss fell to 1.187, its lowest so far.
   - **Loss**: training loss went from about 1.23 to about 1.215 over the last 641 updates, so it's nearly flat (1.20–1.23). That missed my prediction of about 1.19 by update 12682.
   - **Overfitting**: validation loss (1.19) is about level with training loss (about 1.21), so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.45 and the clip never fired. Total weight norm keeps easing slowly (91.41 → 91.14).
   - **Speed**: 59.3 ms per update against 58.6 ms for the starting program.

2. **Kept**: validation accuracy and validation loss are both at new bests, and nothing points to too much or too little regularization. The flat training loss looks like the noise limit at the current learning rate (0.036), and that limit should keep dropping as the cosine decay continues. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 74% over epochs 115 and 120.
   - **Training loss**: below about 1.19 by update 13324.
   - **What would prove me wrong**: validation loss rising above about 1.4 while training loss keeps falling. That would point to overfitting, and I'd raise weight decay.

## Decision 27 (update 13324, scheduled; changed: nothing)

I kept both programs unchanged. Validation at epoch 115 fell sharply, to 67.77% with a validation loss of 1.458, against 75.63% and 1.187 at epoch 110. That looks like a one-off swing rather than a trend. The same thing happened at epoch 55 and the next reading fully recovered.

1. **Observed**:
   - **Validation**: 67.77% at epoch 115, validation loss 1.458. That's a 7.9-point drop from epoch 110's best, more than random noise on 788 images would explain. The earlier case: epoch 55 dropped to 51.27% with loss 2.27, then epoch 60 returned to 55.71% with loss 1.75.
   - **Loss**: training loss kept falling, from about 1.20 to about 1.17 over the last 642 updates.
   - **Gradients and weights**: the gradient norm before clipping is about 2.4 and the clip never fired. Total weight norm keeps easing slowly (91.07 → 90.71). Nothing else looks unusual.
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**:
   - Overfitting would show up as validation loss creeping up over several readings while training loss falls. Here it's a single sudden jump, with nothing unusual in the gradients or weights.
   - My best explanation is that the weights, or the BatchNorm running statistics they produce, happened to be at a bad point on that one evaluation. The learning rate is still 0.033, so the model moves enough between updates for that to happen.
   - These swings should disappear as the learning rate decays to 0, and the final test is taken at the very end.
   - I set a clear rule in `NOTES.md`: if validation loss at epoch 120 is also above 1.35, I'll treat it as real and raise weight decay to 1e-3.

3. **Prediction**:
   - **Validation at epoch 120 (update 13680)**: recovers to at least 73%, with validation loss at or below about 1.25.
   - **Training loss**: about 1.14 by update 13965.
   - **What would prove me wrong**: validation loss above 1.35 at epoch 120 as well. That would point to a real move toward overfitting, and I would raise weight decay to 1e-3.

## Decision 28 (update 13965, scheduled; changed: nothing)

I kept both programs unchanged. Validation recovered to a new best of 77.03% at epoch 120, with the lowest validation loss so far. That confirms the epoch-115 drop was a one-off swing, not the start of overfitting.

1. **Observed**:
   - **Validation**: 77.03% at epoch 120 (update 13680), with validation loss 1.053. Both are the best so far, up from 67.77% and 1.458 at epoch 115.
     - This meets my prediction of at least 73% and loss at or below about 1.25.
     - It's well under the 1.35 loss level at which I would have raised weight decay.
   - **Loss**: training loss went from about 1.17 to about 1.15 over the last 641 updates, so it's still falling slowly.
   - **Overfitting**: validation loss (1.05) is now below training loss (about 1.15), so there's no sign of overfitting. Training loss includes label smoothing and validation loss doesn't, so validation coming in lower is consistent with that.
   - **Gradients and weights**: the gradient norm before clipping is flat at about 2.4 and the clip never fired. Total weight norm keeps easing slowly (90.68 → 90.34).
   - **Speed**: 59.4 ms per update against 58.6 ms for the starting program.

2. **Kept**: both validation numbers are at new bests, training loss is still falling, and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the cosine schedule is unchanged. The learning rate is now 0.029, with about 8,800 updates left to decay to 0. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 76% over epochs 125 and 130.
   - **Training loss**: about 1.12 by update 14606.
   - **What would prove me wrong**: validation loss rising above about 1.3 at two readings in a row while training loss keeps falling. That would point to overfitting, and I would raise weight decay to 1e-3.

## Decision 29 (update 14606, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 79.06% at epoch 125, and training loss keeps falling as the learning rate decays.

1. **Observed**:
   - **Validation**: 79.06% at epoch 125 (update 14250), with validation loss 1.014. Both are the best so far, up from 77.03% and 1.053 at epoch 120.
   - **Loss**: training loss went from about 1.14 to about 1.10 over the last 641 updates.
   - **Overfitting**: validation loss (1.01) is below training loss (about 1.10), so there's no sign of overfitting. Training loss includes label smoothing and validation loss doesn't, so the lower validation figure is expected.
   - **Gradients and weights**: the gradient norm before clipping is about 2.3 and the clip never fired. Total weight norm keeps easing slowly (90.28 → 89.85).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: validation keeps setting new bests as the learning rate comes down (now 0.025), and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the cosine schedule is unchanged, with about 8,200 updates left to decay to 0. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 78% over epochs 130 and 135.
   - **Training loss**: about 1.06 by update 15248.
   - **What would prove me wrong**: validation loss rising above about 1.25 at two readings in a row while training loss keeps falling. That would point to overfitting, and I would raise weight decay.

## Decision 30 (update 15248, scheduled; changed: nothing)

I kept both programs unchanged. Validation came in at 77.79% at epoch 130, slightly below epoch 125's best of 79.06%, but within the noise of a 788-image validation set. Training loss is still falling.

1. **Observed**:
   - **Validation**: 77.79% at epoch 130 (update 14820), with validation loss 1.073, against 79.06% and 1.014 at epoch 125. The 1.3-point dip is within noise (roughly 1.5 points at this accuracy level).
   - **Loss**: training loss went from about 1.10 to about 1.08 over the last 642 updates.
   - **Overfitting**: validation loss (1.07) is about level with training loss, so there's no sign of overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 2.3 and the clip never fired. Total weight norm keeps easing slowly (89.82 → 89.41).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: one reading slightly below the last one, within noise, is not a reason to change course. Training loss is still falling and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the cosine schedule is unchanged. The learning rate is now 0.022, with about 7,550 updates left to decay to 0. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 78.5% over epochs 135 and 140.
   - **Training loss**: about 1.05 by update 15889.
   - **What would prove me wrong**: validation loss rising above about 1.25 at two readings in a row while training loss keeps falling. That would point to overfitting, and I would raise weight decay.

## Decision 31 (update 15889, scheduled; changed: nothing)

I kept both programs unchanged. Validation has stayed between 77% and 79% for three readings while training loss kept falling. That's a mild sign the model is fitting the training images faster than it improves on new ones. But validation loss isn't rising, so it isn't overfitting yet.

1. **Observed**:
   - **Validation**: 77.28% at epoch 135 (update 15390), with validation loss 1.070.
     - Over epochs 125–135, validation accuracy went 79.06 → 77.79 → 77.28 and validation loss stayed between 1.01 and 1.07.
     - Each step is within the noise of a 788-image validation set.
   - **Loss**: training loss went from about 1.06 to about 1.04 over the last 641 updates (1.10 at epoch 125), as I predicted last time.
   - **Gradients and weights**: the gradient norm before clipping is about 2.2 and the clip never fired. Total weight norm keeps easing slowly (89.34 → 88.93).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**:
   - The flat validation is ambiguous: it could be noise, or the start of a gap between training and validation. Validation loss isn't rising, though, so there's no clear case for raising weight decay.
   - With about 6,900 updates left, a weight-decay change made now would have little time to pay off before the end.
   - The final stretch of a cosine schedule, as the learning rate drops from 0.019 to 0, usually adds several points on its own.

   Weight decay stays at 8e-4 and the schedule is unchanged. I recorded this reasoning and the next checks in `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 78.5% over epochs 140 and 145, and goes above 80% from epoch 150 on.
   - **Training loss**: about 1.01 by update 16530.
   - **What would prove me wrong**: validation loss rising above about 1.2 at two readings in a row while training loss keeps falling. That would point to real overfitting, and I'd raise weight decay despite the little time left.

## Decision 32 (update 16530, scheduled; changed: nothing)

I kept both programs unchanged. Validation broke out of the 77–79% range it had held since epoch 125, reaching 80.96% at epoch 140 with the lowest validation loss so far.

1. **Observed**:
   - **Validation**: 80.96% at epoch 140 (update 15960), with validation loss 0.927. Both are the best so far, against 77.28% and 1.070 at epoch 135.
   - **Loss**: training loss went from about 1.04 to about 1.02 over the last 641 updates, so it's still falling slowly. That's just short of my prediction of about 1.01.
   - **Overfitting**: validation loss (0.93) is well below training loss (about 1.02). Training loss includes label smoothing and validation loss doesn't, so a lower validation figure is expected. Last time's concern about the gap has eased.
   - **Gradients and weights**: the gradient norm before clipping is about 2.1 and the clip never fired. Total weight norm keeps easing slowly (88.85 → 88.47).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: the result supports last time's choice to leave the run alone and let the cosine decay keep working. The learning rate is now 0.0156, with about 6,270 updates left to decay to 0. Nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 80% over epochs 145 and 150.
   - **Training loss**: about 0.99 by update 17171.
   - **What would prove me wrong**: validation loss rising above about 1.1 at two readings in a row while training loss keeps falling. That would point to overfitting. With so little time left, I would only change something if the evidence were clear.

## Decision 33 (update 17171, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached a new best of 83.25% at epoch 150, and the two readings since the last decision average 81.8%, above the at-least-80% check.

1. **Observed**:
   - **Validation**:
     - Epoch 145: 80.33%, validation loss 0.946.
     - Epoch 150: 83.25%, validation loss 0.911. Both are the best so far.
   - **Loss**: training loss went from about 1.02 to about 1.00 over the last 641 updates, close to my prediction of 0.99.
   - **Overfitting**: validation loss (0.91) is still below training loss (about 1.00). Training loss includes label smoothing and validation loss doesn't, so this gap is expected and doesn't point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 2.1 and the clip never fired. Total weight norm keeps easing slowly (88.42 → 88.10).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: validation keeps setting new bests as the learning rate comes down (now 0.0127), and nothing points to too much or too little regularization. There are about 5,630 updates left before the rate reaches 0. A change this late would have little time to pay off, and the current recipe is still improving. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 82% over epochs 155 and 160.
   - **Training loss**: about 0.97 by update 17812.
   - **What would prove me wrong**: validation loss rising above about 1.05 at two readings in a row while training loss keeps falling. That would point to overfitting. With so little time left, I would only change something if the evidence were clear.

## Decision 34 (update 17812, scheduled; changed: nothing)

I kept both programs unchanged. Validation held at its best of 83.25% at epoch 155, and validation loss reached a new low of 0.838.

1. **Observed**:
   - **Validation**: 83.25% at epoch 155 (update 17670), the same as at epoch 150. Validation loss fell from 0.911 to 0.838, the lowest so far.
   - **Loss**: training loss went from about 1.00 to about 0.97 over the last 641 updates (0.965 in the latest block), close to my prediction of 0.97.
   - **Overfitting**: validation loss (0.84) is still below training loss. Training loss includes label smoothing and validation loss doesn't, so this gap is expected and doesn't point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 1.95 and the clip never fired. Total weight norm keeps easing slowly (88.04 → 87.75).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: the run is still improving as the learning rate comes down (now 0.010), with nothing that points to too much or too little regularization. About 5,000 updates remain before the rate reaches 0. A change this late would have little time to pay off, while the current recipe keeps setting new bests. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 83% over epochs 160 and 165.
   - **Training loss**: about 0.95 by update 18454.
   - **What would prove me wrong**: validation loss rising above about 0.95 at two readings in a row while training loss keeps falling. That would point to overfitting. With so little time left, I'd only act on clear evidence.

## Decision 35 (update 18454, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached new bests at epoch 160: 85.28% accuracy and a validation loss of 0.785.

1. **Observed**:
   - **Validation**: 85.28% at epoch 160 (update 18240), with validation loss 0.785. Both are the best so far, up from 83.25% and 0.838 at epoch 155.
   - **Loss**: training loss went from about 0.97 to about 0.96 over the last 642 updates (0.949 in the latest block), close to my prediction of about 0.95.
   - **Overfitting**: validation loss (0.78) is still below training loss. Training loss includes label smoothing and validation loss doesn't, so this gap is expected and doesn't point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 1.9 and the clip never fired. Total weight norm keeps easing slowly (87.70 → 87.46).
   - **Speed**: 59.3 ms per update against 58.6 ms for the starting program.

2. **Kept**: validation keeps setting new bests as the learning rate comes down (now 0.0077), and nothing points to too much or too little regularization. About 4,350 updates remain before the rate reaches 0. A change this late would have little time to pay off, while the current recipe is still improving. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 84.5% over epochs 165 and 170.
   - **Training loss**: about 0.94 by update 19095.
   - **What would prove me wrong**: validation loss rising above about 0.9 at two readings in a row while training loss keeps falling. That would point to overfitting. With so little time left, I'd only act on clear evidence.

## Decision 36 (update 19095, scheduled; changed: nothing)

I kept both programs unchanged. Validation came in at 84.14% at epoch 165, about one point below epoch 160's best of 85.28%. That's within the noise of a 788-image validation set, and training loss is still falling.

1. **Observed**:
   - **Validation**: 84.14% at epoch 165 (update 18810), with validation loss 0.830, against 85.28% and 0.785 at epoch 160. Both differences are within noise.
   - **Loss**: training loss went from about 0.96 to about 0.94 over the last 641 updates (0.935 in the latest block), slightly better than my prediction of 0.94.
   - **Overfitting**: validation loss (0.83) is still below training loss. Training loss includes label smoothing and validation loss doesn't, so this gap is expected and doesn't point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping eased from about 1.9 to about 1.7 and the clip never fired. Total weight norm keeps easing slowly (87.44 → 87.25).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: one reading slightly below the best, within noise, is no reason to change course this late. The learning rate is now 0.0058 and about 3,700 updates remain before it reaches 0. The recipe has kept improving and nothing points to too much or too little regularization. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 84.5% over epochs 170 and 175.
   - **Training loss**: about 0.92 by update 19736.
   - **What would prove me wrong**: validation loss above about 0.9 at two readings in a row while training loss keeps falling. That would point to overfitting. With so little time left, I'd only act on clear evidence.

## Decision 37 (update 19736, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached new bests at epoch 170: 86.04% accuracy and a validation loss of 0.761.

1. **Observed**:
   - **Validation**: 86.04% at epoch 170 (update 19380), up from 84.14% at epoch 165 and 85.28% at epoch 160. Validation loss fell to 0.761, the lowest so far.
   - **Loss**: training loss is flattening, at about 0.93 over the last 641 updates (0.927 in the latest block). That's slightly above my prediction of about 0.92.
   - **Overfitting**: validation loss (0.76) is still below training loss. Training loss includes label smoothing and validation loss doesn't, so this gap is expected and doesn't point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping eased to about 1.6 and the clip never fired. Total weight norm keeps easing slowly (87.22 → 87.08).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: the recipe is still improving and nothing points to too much or too little regularization. The learning rate is now 0.004, with about 3,060 updates left before it reaches 0. A change this late would have little time to pay off and could only risk the final result. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results and the next check to `NOTES.md`.

3. **Prediction**:
   - **Validation**: averages at least 85.5% over epochs 175 and 180.
   - **Training loss**: about 0.92 by update 20378.
   - **What would prove me wrong**: validation loss above about 0.9 at two readings in a row while training loss keeps falling. That would point to overfitting, but with so little time left I'd only act on clear evidence.

## Decision 38 (update 20378, scheduled; changed: nothing)

I kept both programs unchanged. Validation held at 85.79% at epoch 175, about level with epoch 170's best of 86.04%, and the run is settling as the learning rate approaches 0.

1. **Observed**:
   - **Validation**: 85.79% at epoch 175 (update 19950), with validation loss 0.766, against 86.04% and 0.761 at epoch 170. Both differences are within the noise of a 788-image validation set.
   - **Loss**: training loss went from about 0.925 to about 0.917 over the last 642 updates, and is now nearly flat. It's slightly above my prediction of about 0.92 by update 20378.
   - **Overfitting**: validation loss (0.77) is still below training loss. Training loss includes label smoothing and validation loss doesn't, so this gap is expected and doesn't point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 1.55 and the clip never fired. Total weight norm is nearly flat (87.05 → 86.96).
   - **Speed**: 59.2 ms per update against 58.6 ms for the starting program.

2. **Kept**: the model is converging as planned. The learning rate is now 0.0025 and about 2,400 updates remain before it reaches 0. A change this late would have little time to pay off and could only risk the final result. Nothing points to a problem. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results to `NOTES.md`.

3. **Prediction**:
   - **Validation**: about 85.5–87% at epochs 180 and 185.
   - **Training loss**: about 0.91 by update 21019.
   - **What would prove me wrong**: a sudden loss spike or validation loss above about 0.9. Neither is likely at such a small learning rate.

## Decision 39 (update 21019, scheduled; changed: nothing)

I kept both programs unchanged. Validation reached new bests at epoch 180: 86.29% accuracy and a validation loss of 0.742. The run is converging as the learning rate approaches 0.

1. **Observed**:
   - **Validation**: 86.29% at epoch 180 (update 20520), with validation loss 0.742. Both are the best so far, slightly above 85.79% and 0.766 at epoch 175. This is within my predicted 85.5–87%.
   - **Loss**: training loss is now almost flat at about 0.91 (0.906 in the latest block), in line with my prediction of about 0.91.
   - **Overfitting**: validation loss (0.74) is still below training loss. Training loss includes label smoothing and validation loss doesn't, so this gap is expected and doesn't point to overfitting.
   - **Gradients and weights**: the gradient norm before clipping is about 1.45 and the clip never fired. Total weight norm is essentially flat (86.94 → 86.88).
   - **Speed**: 59.1 ms per update against 58.6 ms for the starting program.

2. **Kept**: the learning rate is down to 0.0014, with about 1,780 updates left before it reaches 0. At that size, no change I could make now has time to help, and any change could only risk the final result. Nothing points to a problem. Weight decay stays at 8e-4 and the schedule is unchanged. I added this decision's results to `NOTES.md`.

3. **Prediction**:
   - **Validation**: about 85.5–87.5% at epochs 185, 190 and 195.
   - **Final test accuracy**: likely in the mid-80s, measured after update 22800.
   - **Training loss**: about 0.90–0.91 for the rest of the run.
   - **What would prove me wrong**: a loss spike or validation loss above about 0.85. Neither is likely at such a small learning rate.

## Decision 40 (update 21660, scheduled; changed: nothing)

I kept both programs unchanged for the rest of the run. At this last decision, validation is at its best accuracy so far: 86.55% at epoch 185. The learning rate is nearly 0, so nothing I could change now would help.

1. **Observed**:
   - **Validation**: 86.55% at epoch 185 (update 21090), with validation loss 0.743. Accuracy is slightly above epoch 180's 86.29%, and loss is about the same as its 0.742.
   - **Loss**: training loss has flattened at about 0.905.
   - **Gradients and weights**: the gradient norm before clipping is about 1.45 and the clip never fired. Total weight norm has stopped changing (about 86.85).
   - **Learning rate**: 0.00056, on its way to 0 at update 22800.
   - **Speed**: 59.3 ms per update against 58.6 ms for the starting program.

2. **Kept**: with about 1,140 updates left at a learning rate this small, the weights will barely move. A change now could only add risk, with no time to pay off. The recipe I put in at update 142 ran unchanged for the whole run:
   - **Optimizer**: SGD with Nesterov momentum.
   - **Learning rate**: peak 0.08.
   - **Weight decay**: 8e-4 on conv and fully connected weights.
   - **Schedule**: warmup to update 1140, then cosine decay to 0.
   - **Clip**: global gradient-norm clip at 5, as a safety net only (it never fired).

   I added the final entry to `NOTES.md`.

3. **Prediction**:
   - **Validation**: about 86–87% at epochs 190 and 195.
   - **Final test accuracy**: about 85–87% after update 22800.
   - **What would prove me wrong**: the validation set has only 788 images, so the test score could differ from these figures by a point or two. A larger gap would mean the validation set is a poor guide to the test set.
