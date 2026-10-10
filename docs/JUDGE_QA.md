# Questions a judge may ask, with answers

Short answers the team can give from memory, each with the place in the code or outputs that backs it up.

## The metric

**Why is 99.8% accuracy meaningless here?**
Only 0.17% of transactions are fraud. Answering "never fraud" every time is 99.83% accurate and catches zero frauds. Accuracy is counting the easy genuine transactions. We never use it.

**Why PR-AUC and not ROC-AUC?**
ROC-AUC rewards a model for ranking the 139,000 genuine transactions below the frauds, which is easy. Our logistic baseline and our champion both score 0.94 to 0.98 on ROC-AUC, yet differ by 13 real frauds in the top 200. Precision and recall look only at the alerts and the frauds, which is what the analysts live with. See the two curves side by side in the app's Model evidence tab.

**At 200 alerts a day, what do precision and recall become, in plain words?**
Of 200 alerts, 154 are real fraud, about three in four. Those 154 are 77% of the day's 201 frauds. On another day we would expect precision between 66% and 85% (`outputs/queue_statement.txt`).

**Where do the ranges come from?**
A bootstrap: we resample the day's transactions 1,000 times, rebuild the top 200 each time and take the middle 95% (`metrics.bootstrap_queue_ci`). One day holds only about 200 frauds, so a single number would hide real uncertainty.

## Leakage

**Where exactly did you apply SMOTE, and why?**
Inside an imbalanced-learn Pipeline (`models.py`, `xgb_smote_in_fold`). That Pipeline resamples during `fit` only, so in cross-validation it oversamples the four training folds and never the held-out fold. We also ran the wrong way once on purpose: SMOTE on everything, then split. That reports PR-AUC 0.9999. Done correctly it is 0.760 (`train.leakage_exhibit`). The first number is fiction: synthetic copies of test frauds are in training, and half the test set is synthetic.

**In the end you did not use SMOTE. Why?**
It did not help. Plain XGBoost had the best cross-validated PR-AUC (0.877 against 0.849 with SMOTE). For ranking the top 200, the model only needs the order right, and it learns that without rebalancing.

**What did you fit on training data only?**
Everything that is fitted: the scaler, SMOTE and the model. They are all steps of one Pipeline. The feature step is stateless. `tests/test_pipeline.py` checks that the scaler's mean equals the training mean and differs from the full-data mean.

**Why remove duplicates?**
The file has 1,081 exact duplicate rows. With a random split, 403 test rows are identical to a training row, 9 of them frauds, so the model is partly graded on rows it has memorised. We drop duplicates before any split.

**Is there any leak you could not remove?**
Yes, and we say so. The data provider computed the PCA features V1-V28 on the whole file, including what we use as test data. It is mild and outside our control.

## The split and the model

**Why split by day instead of a random stratified split?**
Because the question is "what will tomorrow's queue look like", and a model in production is always scored on a later day than it was trained on. A random split trains and tests on the same hours. We report both: 0.831 PR-AUC on the random split, 0.769 on the day replay. The gap is the cost of honesty.

**Why did PR-AUC fall from 0.877 in cross-validation to 0.769 on day 2?**
Cross-validation mixes rows from the same day, so it shares that day's fraud patterns. Day 2 has different ones. For example, 20 of our 46 false alarms are one repeating EUR 0.77 pattern that looks like day-1 card testing but is labelled genuine. This is the drift a client would see on the first morning, and it is why monitoring is first in our next steps.

**The weighted model caught 156 on day 2 and yours caught 154. Why not use it?**
We fixed the rule before looking at day 2: best cross-validated PR-AUC on day 1. Switching after seeing the test result would be choosing on the test set, and two frauds is well inside the noise.

**Why did you drop `Time`?**
It is seconds since the first transaction, a counter. Every day-2 value is larger than anything in training, so a tree would treat all of day 2 as "later than everything I know". We keep the useful part, hour of day, encoded as sine and cosine so 23:00 and 01:00 are close.

**What are V1 to V28?**
PCA components the issuer published instead of the confidential raw fields. We cannot name them, so our reasons say "V14 is lower than 99.9% of genuine transactions" rather than inventing a meaning. The issuer can map each component back.

## The output and the agent

**How are the reasons computed?**
SHAP values: for one transaction, how much each input moved the score up or down. XGBoost computes them exactly (`pred_contribs=True`, in `explain.py`). We take the three largest pushes towards fraud and describe each value against the genuine transactions in the training day.

**How did you choose the risk bands?**
They are policy thresholds, not tuned: CRITICAL at 0.99 because holding a payment and phoning a customer should need near-certainty, HIGH at 0.90. We then measured them. On day 2: 97%, 81% and 42% of alerts in the three bands were fraud.

**What does the agent do, and what stops it doing harm?**
It gathers the evidence for an alert, picks an action from a fixed rule table and drafts the note. It cannot execute anything: every draft is `PENDING_APPROVAL` until an analyst approves it, and approvals are logged with who and when (`store.py`).

**What stops the LLM making things up?**
Three things. It receives only a JSON evidence object. The prompt tells it to quote numbers exactly and never claim fraud is confirmed. And code checks its draft: every number must appear in the evidence, the transaction id must be present, and banned phrases cause rejection, in which case the template note is used (`agent.check_grounding`, tested in `tests/test_triage_agent.py`).

**Why a local LLM?**
Transaction data should not leave the bank. phi3.5 runs on the laptop through Ollama, free, with no API key.

## Business

**What is the business value?**
The team's 200 daily reviews go from catching about nothing (0 by largest amount, 0.3 at random) to 154 real frauds worth EUR 16,428 on the test day, with no extra staff.

**Should the client hire more analysts?**
Not on this evidence. Going from 200 to 300 reviews finds 5 more frauds, and even at 600 reviews a day about one fraud in six is still missed. Those frauds look genuine to this model, so the next gain comes from better signals such as card-level velocity features.

**Is the EUR 0.77 cluster really a false alarm?**
We do not know, and we say so. Twenty near-identical EUR 0.77 transactions were labelled genuine but look like card testing. Either one allow-list rule frees 10% of daily capacity, or the bank has unreported fraud. Both are worth a look.

**What would you do first with another week?**
Feed analyst decisions back as labels, add drift monitoring, and group look-alike alerts so one decision clears a pattern.
