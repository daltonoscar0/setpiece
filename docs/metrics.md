# metrics

Dataset `data/interim/corners.jsonl` — 2464 corners from 415 matches, 1. Bundesliga, FIFA World Cup, La Liga, Ligue 1, Major League Soccer, UEFA Euro, UEFA Women's Euro, Women's World Cup. Trained 2026-08-27.

**360 coverage.** 2468 of the 4467 corners walked came with a 360 freeze frame — 55.2% — and that is what the dataset is a sample of. Coverage is not uniform across the competitions and is not a property anyone can choose: it is whether StatsBomb published a frame. Among the competitions that ship any frames at all it ranges from 43.9% (Major League Soccer) to 76.0% (La Liga). African Cup of Nations ships **no** freeze frames despite being flagged for 360 in the catalogue, so none of its corners are here at all. A further 4 frames showed fewer than eight players and were dropped at load time.

| competition | corners with a frame | corners seen | coverage |
|---|---|---|---|
| African Cup of Nations | 0 | 460 | 0.0% |
| Major League Soccer | 25 | 57 | 43.9% |
| FIFA World Cup | 300 | 569 | 52.7% |
| Ligue 1 | 271 | 511 | 53.0% |
| UEFA Euro | 546 | 969 | 56.3% |
| Women's World Cup | 406 | 602 | 67.4% |
| UEFA Women's Euro | 425 | 622 | 68.3% |
| 1. Bundesliga | 245 | 348 | 70.4% |
| La Liga | 250 | 329 | 76.0% |

**Labels.** First touch: 1320 defender, 1033 attacker, 111 unknown. Shot within five events: base rate 0.268. Both windows are **event counts**, per the spec, and an event-count window walks straight through a dead ball: it will happily call the restart twelve events later the first touch on this corner. They therefore carry plausibility guards — a touch more than 10 s after the delivery, or a shot more than 15 s after it, is discarded. The guards bite. The first touch normally lands a median 1.61 s after the delivery (p90 2.53 s), but 110 rows sat beyond the guard, out to 238 s — and 102 of them were being scored as *the defence won first contact*. A discarded touch is dropped, never relabelled, which is what the 111-row `unknown` class is: corners this dataset declines to score rather than guesses at. 1 of those 111 carries no timing at all rather than a late one, so the guard could not clear it either.

Measured against `data/interim/corners_v2.jsonl`, the same corners built without the guards, so the guards' effect on the headline numbers is visible rather than inferred:

| label | before the guards | after |
|---|---|---|
| first touch: defender | 1422 | 1320 |
| first touch: attacker | 1042 | 1033 |
| first touch: unknown | 0 | 111 |
| attacker-first corners with a matched receiver | 923 | 918 |
| shot base rate | 0.275 | 0.268 |

17 corners lost their shot label. Every number in this file is measured against the guarded labels in the right-hand column.

Validation is **purged 5-fold cross-validation grouped by `match_id`**: every corner of a match sits in exactly one fold, so no model is ever scored on a match it trained on. The numbers below are pooled out-of-fold predictions — each corner scored exactly once — and both baselines are fitted on the same training folds and scored on the same held-out rows as the GNN.

Rows per fold: [493, 493, 493, 493, 492]. Matches per fold: [83, 83, 83, 83, 83].

**Every figure is a mean over 5 repeats of the whole protocol, written `mean ±sd`** with the sample standard deviation across those repeats. Seeds [0, 1000, 2000, 3000, 4000] each re-draw the fold assignment *and* the network initialisation, so the spread covers both of the things that change on a re-run. The two baselines are deterministic given a split, so their spread is pure split noise; the GNN's carries initialisation noise on top. A difference smaller than the spread beside it is not a result. Where two models are compared below, the gap is differenced *within* each repeat first, which cancels the shared split noise and is a tighter test than eyeballing two columns.

Run at `torch.set_num_threads(1)`. That is not decoration: torch's parallel reductions sum in thread-partition order, so the GNN column is only reproducible at a pinned thread count, and every checkpoint's card records the count it was trained at.

Freeze frames carry no velocities, so every model here is `assume_static`.

Features are computed in **mirrored coordinates** — reflected about y = GOAL_Y so the corner is always taken from the y < GOAL_Y side; every signed y feature is in those coordinates. A corner from the left and its mirror image from the right are the same coaching situation, so the reflection is a halving of the input space rather than a feature: the column count is unchanged. Every number below was measured after it, and a checkpoint trained before it has the same feature names and the opposite sign convention.

## receiver head

Scored on 2238 corners with a usable receiver label, of which 918 were won by an attacker. Top-1 and top-3 rank the candidate attackers against each other on the attacker-first subset, which is the only question the nearest-to-goal heuristic can answer. Overall accuracy is the argmax over the candidates *and* the "defence wins first contact" class, across every scored row.

The other 226 corners carry no usable receiver label: 115 are attacker-first corners whose receiver was not visible in the freeze frame, so there is no node to point at, and 111 are the `unknown` class — corners whose only candidate first touch fell outside the plausibility guard, so there is no label to point at either. Neither filter is neutral — both remove attacker-first rows far more often than defence-wins rows — so the scored subset is more defence-heavy than the dataset as a whole (0.590 against 0.536). All three models are scored on exactly these rows, so the columns compare cleanly against each other, but the overall-accuracy and defence-AUC figures are not population estimates.

| model | top-1 (attacker-first) | top-3 (attacker-first) | overall acc. incl. defence | defence-wins AUC |
|---|---|---|---|---|
| GNN | 0.432 ±0.012 | 0.729 ±0.015 | 0.625 ±0.006 | 0.631 ±0.007 |
| logistic regression | 0.432 ±0.001 | 0.734 ±0.004 | 0.625 ±0.002 | 0.636 ±0.005 |
| heuristic | 0.159 ±0.000 | 0.498 ±0.000 | 0.590 ±0.000 | 0.473 ±0.012 |

Chance level over the candidates is 0.150 top-1 and 0.441 top-3. Top-3 has a high floor by construction: a corner with three or fewer candidate attackers is a free hit for every model, so read the top-3 column against that floor and not against zero. Always predicting "defence wins" would score 0.590 overall.

## shot head

Binary `shot_within_5`, scored on all 2464 held-out corners. Base rate 0.268. The heuristic is the training-fold shot rate — a constant within a fold, so it carries no information about an individual corner. It lands a little *below* 0.5 rather than on it, and that is mechanical rather than noise: the training-fold rate is a strictly decreasing function of the held-out fold's own rate, so the score is mildly anti-correlated with the label by construction. Read it as the floor; the gap between it and 0.5 is not a signal.

| model | AUC |
|---|---|
| GNN | 0.584 ±0.014 |
| logistic regression | 0.586 ±0.009 |
| heuristic | 0.481 ±0.010 |

## reading

- GNN and logistic regression are level on top-1 on the attacker-first subset at 0.432, to the precision this table prints. Seed by seed, the GNN minus the logistic regression on top-1 on the attacker-first subset is 0.000 ±0.012 over 5 repeats — 0.0x its own spread, **inside the noise floor** and should not be leaned on.
- The logistic regression baseline **beats** the GNN on top-3 on the attacker-first subset: 0.734 against 0.729. Seed by seed, the logistic regression minus the GNN on top-3 on the attacker-first subset is 0.004 ±0.018 over 5 repeats — 0.2x its own spread, **inside the noise floor** and should not be leaned on.
- The logistic regression baseline **beats** the GNN on AUC for the defence-wins class: 0.636 against 0.631. Seed by seed, the logistic regression minus the GNN on AUC for the defence-wins class is 0.005 ±0.011 over 5 repeats — 0.4x its own spread, **inside the noise floor** and should not be leaned on.
- The logistic regression baseline **beats** the GNN on shot AUC: 0.586 against 0.584. Seed by seed, the logistic regression minus the GNN on shot AUC is 0.002 ±0.012 over 5 repeats — 0.1x its own spread, **inside the noise floor** and should not be leaned on.

Parameters: receiver 118,082, shot 97,601.

## what moved, and why

More than one thing changed under the table above, so a single before-and-after cannot say which of them did what. Each column below is the same protocol — same seeds, same folds, same hyperparameters — run over one configuration. A step between two columns is differenced *within* each seed, so the shared split draw cancels, and is read against its own spread exactly like the model comparisons above.

### `gnn`

| metric | before | mirrored | mirrored + guarded labels |
|---|---|---|---|
| receiver top-1 | 0.4254 ±0.0035 | 0.4377 ±0.0053 | 0.4322 ±0.0117 |
| receiver top-3 | 0.7263 ±0.0081 | 0.7356 ±0.0083 | 0.7294 ±0.0151 |
| receiver overall acc. | 0.6381 ±0.0042 | 0.6407 ±0.0029 | 0.6255 ±0.0063 |
| defence-wins AUC | 0.6263 ±0.0025 | 0.6285 ±0.0084 | 0.6308 ±0.0071 |
| shot AUC | 0.5768 ±0.0149 | 0.5824 ±0.0137 | 0.5844 ±0.0135 |

**before → mirrored.**

| metric | step | its own spread | ratio | reading |
|---|---|---|---|---|
| receiver top-1 | +0.0124 | ±0.0060 | 2.1x | outside the noise floor |
| receiver top-3 | +0.0093 | ±0.0077 | 1.2x | **inside the noise floor** |
| receiver overall acc. | +0.0026 | ±0.0051 | 0.5x | **inside the noise floor** |
| defence-wins AUC | +0.0022 | ±0.0077 | 0.3x | **inside the noise floor** |
| shot AUC | +0.0056 | ±0.0076 | 0.7x | **inside the noise floor** |

**mirrored → mirrored + guarded labels.**

The labels differ between these two columns, so they are not scored on the same rows. This step is the change in the reported number, not a model improvement — a metric can move here because the question changed.

| metric | step | its own spread | ratio | reading |
|---|---|---|---|---|
| receiver top-1 | -0.0055 | ±0.0164 | 0.3x | **inside the noise floor** |
| receiver top-3 | -0.0062 | ±0.0178 | 0.3x | **inside the noise floor** |
| receiver overall acc. | -0.0152 | ±0.0085 | 1.8x | **inside the noise floor** |
| defence-wins AUC | +0.0023 | ±0.0121 | 0.2x | **inside the noise floor** |
| shot AUC | +0.0020 | ±0.0040 | 0.5x | **inside the noise floor** |

### `gnn-logreg`, the gap itself, differenced inside each seed before anything else

| metric | before | mirrored | mirrored + guarded labels |
|---|---|---|---|
| receiver top-1 | 0.0078 ±0.0025 | 0.0078 ±0.0046 | 0.0004 ±0.0116 |
| receiver top-3 | 0.0113 ±0.0111 | 0.0056 ±0.0062 | -0.0041 ±0.0182 |
| receiver overall acc. | 0.0030 ±0.0058 | 0.0038 ±0.0016 | 0.0007 ±0.0063 |
| defence-wins AUC | -0.0075 ±0.0042 | -0.0073 ±0.0094 | -0.0048 ±0.0110 |
| shot AUC | -0.0089 ±0.0128 | 0.0015 ±0.0133 | -0.0015 ±0.0117 |

**before → mirrored.**

| metric | step | its own spread | ratio | reading |
|---|---|---|---|---|
| receiver top-1 | -0.0000 | ±0.0035 | 0.0x | **inside the noise floor** |
| receiver top-3 | -0.0056 | ±0.0070 | 0.8x | **inside the noise floor** |
| receiver overall acc. | +0.0009 | ±0.0046 | 0.2x | **inside the noise floor** |
| defence-wins AUC | +0.0002 | ±0.0076 | 0.0x | **inside the noise floor** |
| shot AUC | +0.0104 | ±0.0074 | 1.4x | **inside the noise floor** |

**mirrored → mirrored + guarded labels.**

The labels differ between these two columns, so they are not scored on the same rows. This step is the change in the reported number, not a model improvement — a metric can move here because the question changed.

| metric | step | its own spread | ratio | reading |
|---|---|---|---|---|
| receiver top-1 | -0.0074 | ±0.0160 | 0.5x | **inside the noise floor** |
| receiver top-3 | -0.0098 | ±0.0161 | 0.6x | **inside the noise floor** |
| receiver overall acc. | -0.0031 | ±0.0072 | 0.4x | **inside the noise floor** |
| defence-wins AUC | +0.0025 | ±0.0120 | 0.2x | **inside the noise floor** |
| shot AUC | -0.0031 | ±0.0044 | 0.7x | **inside the noise floor** |

## calibration

Two of the numbers this project emits are genuine binary probabilities and can be checked against observed frequencies: the shot head's sigmoid, and the receiver head's `P(defence wins first contact)`. The per-attacker receiver probabilities are not in that family — they are a distribution over a variable-length candidate set — so they are ranked, not calibrated. Expected calibration error (ECE) below is the count-weighted mean gap between predicted and observed frequency over 10 bins of the pooled out-of-fold predictions, reported under both binnings: equal-width bins are the textbook definition, and equal-count bins are the honest view for a head whose predictions all sit inside a narrow band, where equal-width bins drop every row into two buckets and report a flatteringly small number. Calibration is a monotone rescaling, so **every AUC above is identical before and after it**: this section changes what a number means, never how well it ranks.

| head | rows | base rate | mean predicted | predicted range | ECE raw | ECE calibrated | ECE raw (equal-count) | ECE calibrated (equal-count) |
|---|---|---|---|---|---|---|---|---|
| shot | 2464 | 0.268 | 0.261 | 0.039-0.510 | 0.018 | 0.018 | 0.027 | 0.027 |
| defence wins first contact | 2238 | 0.590 | 0.594 | 0.028-0.890 | 0.051 | 0.051 | 0.057 | 0.057 |

- The raw shot head spans 0.039-0.510 with a mean of 0.261 against a base rate of 0.268. Raw ECE 0.018, Brier 0.193; the best nested calibrator reaches 0.018, so no calibrator ships for it.
- The raw defence-wins head spans 0.028-0.890 with a mean of 0.594 against a base rate of 0.590. Raw ECE 0.051, Brier 0.228; the best nested calibrator reaches 0.051, so no calibrator ships for it.

**reliability of the raw shot head, 10 equal-count bins**

| predicted range | n | mean predicted | observed | gap |
|---|---|---|---|---|
| 0.000-0.146 | 247 | 0.119 | 0.138 | +0.018 |
| 0.146-0.187 | 246 | 0.167 | 0.228 | +0.060 |
| 0.187-0.219 | 246 | 0.205 | 0.199 | -0.006 |
| 0.219-0.243 | 247 | 0.231 | 0.251 | +0.020 |
| 0.243-0.265 | 246 | 0.254 | 0.297 | +0.043 |
| 0.265-0.282 | 246 | 0.273 | 0.272 | -0.001 |
| 0.282-0.302 | 247 | 0.293 | 0.255 | -0.038 |
| 0.302-0.331 | 246 | 0.317 | 0.309 | -0.008 |
| 0.331-0.364 | 246 | 0.346 | 0.374 | +0.028 |
| 0.364-1.000 | 247 | 0.400 | 0.356 | -0.044 |

**reliability of the raw defence wins first contact head, 10 equal-count bins**

| predicted range | n | mean predicted | observed | gap |
|---|---|---|---|---|
| 0.000-0.338 | 224 | 0.222 | 0.237 | +0.015 |
| 0.338-0.461 | 224 | 0.410 | 0.464 | +0.054 |
| 0.461-0.537 | 224 | 0.502 | 0.594 | +0.092 |
| 0.537-0.593 | 223 | 0.567 | 0.637 | +0.070 |
| 0.593-0.638 | 224 | 0.615 | 0.647 | +0.032 |
| 0.638-0.673 | 224 | 0.656 | 0.625 | -0.031 |
| 0.673-0.701 | 223 | 0.686 | 0.691 | +0.004 |
| 0.701-0.734 | 224 | 0.717 | 0.607 | -0.110 |
| 0.734-0.770 | 224 | 0.751 | 0.714 | -0.037 |
| 0.770-1.000 | 224 | 0.810 | 0.683 | -0.127 |

The calibrated column is measured **nested**: for each fold the calibrator was fitted on scores from an extra model that had seen neither that fold nor the fold the calibrator was fitted on, so no held-out row ever contributed to the map applied to it. The one approximation, stated rather than hidden: those extra models were trained on three fifths of the data instead of four fifths, and a model trained on less data is a little less confident, so this understates rather than overstates how much shrinking the shipped head needs.

One consequence of measuring it that way, since it looks like a contradiction otherwise: the *nested* column stitches together a different calibrator per fold, and two rows in different folds can swap order across that seam, so the nested column is not AUC-preserving. What ships is a single global map, and a single monotone map cannot reorder anything — which is why the AUC tables above stand unchanged.

**The shipped checkpoints are uncalibrated, and deliberately so:** no calibrator improved held-out ECE on either head, so neither checkpoint ships one. The probabilities they emit are ranking scores.

Calibrators are **not** applied automatically. `score`, `suggest`, `demo` and the app all print the raw sigmoid, so a probability from the CLI today is a ranking score with a documented correction available beside it, not a calibrated probability.

## receiver label threshold

`RECEIVER_MATCH_MAX_M = 8.0` in the frozen `setpiece/_io.py` decides which attacker-first corners get a receiver label at all: the first touch is matched to the nearest candidate attacker in the freeze frame and the row survives only if that candidate is within the threshold. It is a judgement call rather than a fitted number, so the labels here are re-derived at other thresholds from the `receiver_match_dist` the dataset stores for exactly this purpose, and the *same* out-of-fold predictions are re-scored against them. Nothing is retrained: this measures how sensitive the reported numbers are to the threshold, not how sensitive a refitted model would be. Match distances are short where they are usable at all — median 3.19 m, 75th percentile 5.30 m, 90th 8.47 m, 95th 14.55 m — so the threshold sits far out in the tail rather than through the middle of the distribution. Dropping the threshold entirely would add 115 attacker-first rows to the 918 that survive at 8 m (11.1% of every attacker-first corner with a match), and each of those is a corner whose *nearest visible attacker* was more than 8 m from where the ball was actually touched. That is not a receiver who happened to be mislocated; it is the closest thing on camera to an event whose real actor the frame does not show. Tightening to 4 m costs 280 attacker-first rows, leaving 638, and moves top-1 by GNN -0.001, logistic regression -0.008 and heuristic -0.001.

| threshold | corners with a usable label | attacker-first among them | share of matchable attacker-first rows kept |
|---|---|---|---|
| 4 m | 1958 | 638 | 0.618 |
| 6 m | 2155 | 835 | 0.808 |
| 8 m *(shipped)* | 2238 | 918 | 0.889 |
| 10 m | 2274 | 954 | 0.924 |
| 12 m | 2294 | 974 | 0.943 |
| unthresholded | 2353 | 1033 | 1.000 |

| threshold | GNN top-1 | logreg top-1 | heuristic top-1 | GNN top-3 | logreg top-3 | heuristic top-3 | GNN overall acc. |
|---|---|---|---|---|---|---|---|
| 4 m | 0.431 ±0.011 | 0.424 ±0.003 | 0.158 ±0.000 | 0.728 ±0.014 | 0.733 ±0.007 | 0.484 ±0.000 | 0.698 ±0.006 |
| 6 m | 0.434 ±0.011 | 0.433 ±0.002 | 0.154 ±0.000 | 0.733 ±0.015 | 0.739 ±0.006 | 0.493 ±0.000 | 0.646 ±0.006 |
| 8 m *(shipped)* | 0.432 ±0.012 | 0.432 ±0.001 | 0.159 ±0.000 | 0.729 ±0.015 | 0.734 ±0.004 | 0.498 ±0.000 | 0.625 ±0.006 |
| 10 m | 0.416-0.454 | 0.416-0.453 | 0.153-0.191 | 0.702-0.740 | 0.706-0.744 | 0.479-0.517 | 0.616-0.631 |
| 12 m | 0.407-0.465 | 0.407-0.464 | 0.150-0.207 | 0.687-0.745 | 0.691-0.749 | 0.469-0.527 | 0.610-0.635 |
| unthresholded | 0.384-0.495 | 0.384-0.495 | 0.141-0.253 | 0.648-0.760 | 0.652-0.763 | 0.442-0.554 | 0.595-0.644 |

- Every cell is a mean ±sd over the same 5 repeats as the main table, re-scoring stored out-of-fold predictions. The models are the ones trained under the shipped 8 m labels throughout; only the rows they are scored on change.
- Rows above the shipped threshold are given as a **range, not an estimate**. `setpiece/_io.py` keeps `receiver_match_dist` but drops `receiver_idx` once the match exceeds its own threshold, so for a corner matched at 11 m the dataset records that a match existed and how far it was, but not which attacker it was. The range is every added row wrong at the low end and every added row right at the high end. Loosening the threshold for real needs a re-fetch, not a re-read.
- The defence-wins AUC is not swept: loosening the threshold changes the negative class of that binary problem as well as its size, so a bracket on it would be vacuous rather than informative.
- Across 4 m, 6 m, 8 m no model's top-1 or top-3 moves by more than 0.013 from end to end, so the headline receiver comparison does not rest on the threshold. The ranking of the three models is identical at every threshold.
