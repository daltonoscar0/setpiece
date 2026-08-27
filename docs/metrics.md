# metrics

Dataset `data/interim/corners.jsonl` — 2464 corners from 415 matches, 1. Bundesliga, FIFA World Cup, La Liga, Ligue 1, Major League Soccer, UEFA Euro, UEFA Women's Euro, Women's World Cup. Trained 2026-08-26.

Validation is **purged 5-fold cross-validation grouped by `match_id`**: every corner of a match sits in exactly one fold, so no model is ever scored on a match it trained on. The numbers below are pooled out-of-fold predictions — each corner scored exactly once — and both baselines are fitted on the same training folds and scored on the same held-out rows as the GNN.

Rows per fold: [493, 493, 493, 493, 492]. Matches per fold: [83, 83, 83, 83, 83].

**Every figure is a mean over 5 repeats of the whole protocol, written `mean ±sd`** with the sample standard deviation across those repeats. Seeds [0, 1000, 2000, 3000, 4000] each re-draw the fold assignment *and* the network initialisation, so the spread covers both of the things that change on a re-run. The two baselines are deterministic given a split, so their spread is pure split noise; the GNN's carries initialisation noise on top. A difference smaller than the spread beside it is not a result. Where two models are compared below, the gap is differenced *within* each repeat first, which cancels the shared split noise and is a tighter test than eyeballing two columns.

Run at `torch.set_num_threads(1)`. That is not decoration: torch's parallel reductions sum in thread-partition order, so the GNN column is only reproducible at a pinned thread count, and every checkpoint's card records the count it was trained at.

Freeze frames carry no velocities, so every model here is `assume_static`.

## receiver head

Scored on 2345 corners with a usable receiver label, of which 923 were won by an attacker. Top-1 and top-3 rank the candidate attackers against each other on the attacker-first subset, which is the only question the nearest-to-goal heuristic can answer. Overall accuracy is the argmax over the candidates *and* the "defence wins first contact" class, across every scored row.

The other 119 corners carry no usable receiver label — almost all of them attacker-first corners whose receiver was not visible in the freeze frame, so there is no node to point at. That filter removes attacker-first rows far more often than defence-wins rows, which leaves the scored subset more defence-heavy than the dataset as a whole. All three models are scored on exactly these rows, so the columns compare cleanly against each other, but the overall-accuracy and defence-AUC figures are not population estimates.

| model | top-1 (attacker-first) | top-3 (attacker-first) | overall acc. incl. defence | defence-wins AUC |
|---|---|---|---|---|
| GNN | 0.425 ±0.003 | 0.726 ±0.008 | 0.638 ±0.004 | 0.626 ±0.003 |
| logistic regression | 0.418 ±0.004 | 0.715 ±0.004 | 0.635 ±0.002 | 0.634 ±0.004 |
| heuristic | 0.158 ±0.000 | 0.496 ±0.000 | 0.606 ±0.000 | 0.472 ±0.012 |

Chance level over the candidates is 0.149 top-1 and 0.441 top-3. Top-3 has a high floor by construction: a corner with three or fewer candidate attackers is a free hit for every model, so read the top-3 column against that floor and not against zero. Always predicting "defence wins" would score 0.606 overall.

## shot head

Binary `shot_within_5`, scored on all 2464 held-out corners. Base rate 0.275. The heuristic is the training-fold shot rate — a constant within a fold, so it carries no information about an individual corner. It lands a little *below* 0.5 rather than on it, and that is mechanical rather than noise: the training-fold rate is a strictly decreasing function of the held-out fold's own rate, so the score is mildly anti-correlated with the label by construction. Read it as the floor; the gap between it and 0.5 is not a signal.

| model | AUC |
|---|---|
| GNN | 0.577 ±0.015 |
| logistic regression | 0.586 ±0.007 |
| heuristic | 0.480 ±0.009 |

## reading

- The GNN wins on top-1 on the attacker-first subset (0.425), ahead of the logistic regression at 0.418. Seed by seed, the GNN minus the logistic regression on top-1 on the attacker-first subset is 0.008 ±0.002 over 5 repeats — 3.2x its own spread, clear of the noise floor.
- The GNN wins on top-3 on the attacker-first subset (0.726), ahead of the logistic regression at 0.715. Seed by seed, the GNN minus the logistic regression on top-3 on the attacker-first subset is 0.011 ±0.011 over 5 repeats — 1.0x its own spread, **inside the noise floor** and should not be leaned on.
- The logistic regression baseline **beats** the GNN on AUC for the defence-wins class: 0.634 against 0.626. Seed by seed, the logistic regression minus the GNN on AUC for the defence-wins class is 0.007 ±0.004 over 5 repeats — 1.8x its own spread, **inside the noise floor** and should not be leaned on.
- The logistic regression baseline **beats** the GNN on shot AUC: 0.586 against 0.577. Seed by seed, the logistic regression minus the GNN on shot AUC is 0.009 ±0.013 over 5 repeats — 0.7x its own spread, **inside the noise floor** and should not be leaned on.

Parameters: receiver 118,082, shot 97,601.

## calibration

Two of the numbers this project emits are genuine binary probabilities and can be checked against observed frequencies: the shot head's sigmoid, and the receiver head's `P(defence wins first contact)`. The per-attacker receiver probabilities are not in that family — they are a distribution over a variable-length candidate set — so they are ranked, not calibrated. Expected calibration error (ECE) below is the count-weighted mean gap between predicted and observed frequency over 10 bins of the pooled out-of-fold predictions, reported under both binnings: equal-width bins are the textbook definition, and equal-count bins are the honest view for a head whose predictions all sit inside a narrow band, where equal-width bins drop every row into two buckets and report a flatteringly small number. Calibration is a monotone rescaling, so **every AUC above is identical before and after it**: this section changes what a number means, never how well it ranks.

| head | rows | base rate | mean predicted | predicted range | ECE raw | ECE calibrated | ECE raw (equal-count) | ECE calibrated (equal-count) |
|---|---|---|---|---|---|---|---|---|
| shot | 2464 | 0.275 | 0.274 | 0.052-0.618 | 0.044 | 0.016 | 0.041 | 0.021 |
| defence wins first contact | 2345 | 0.606 | 0.608 | 0.054-0.920 | 0.045 | 0.045 | 0.048 | 0.048 |

- The raw shot head spans 0.052-0.618 with a mean of 0.274 against a base rate of 0.275. Raw ECE 0.044, Brier 0.197; the best nested calibrator reaches 0.016, so a `platt` map ships with it.
- The raw defence-wins head spans 0.054-0.920 with a mean of 0.608 against a base rate of 0.606. Raw ECE 0.045, Brier 0.224; the best nested calibrator reaches 0.045, so no calibrator ships for it.

**reliability of the raw shot head, 10 equal-count bins**

| predicted range | n | mean predicted | observed | gap |
|---|---|---|---|---|
| 0.000-0.151 | 247 | 0.123 | 0.158 | +0.035 |
| 0.151-0.193 | 246 | 0.173 | 0.167 | -0.006 |
| 0.193-0.224 | 246 | 0.210 | 0.260 | +0.050 |
| 0.224-0.251 | 247 | 0.238 | 0.271 | +0.034 |
| 0.251-0.274 | 246 | 0.263 | 0.313 | +0.050 |
| 0.274-0.299 | 246 | 0.286 | 0.325 | +0.039 |
| 0.299-0.326 | 247 | 0.312 | 0.291 | -0.021 |
| 0.326-0.353 | 246 | 0.339 | 0.301 | -0.038 |
| 0.353-0.390 | 246 | 0.370 | 0.329 | -0.041 |
| 0.390-1.000 | 247 | 0.427 | 0.332 | -0.095 |

**reliability of the raw defence wins first contact head, 10 equal-count bins**

| predicted range | n | mean predicted | observed | gap |
|---|---|---|---|---|
| 0.000-0.389 | 235 | 0.284 | 0.234 | -0.050 |
| 0.389-0.503 | 234 | 0.459 | 0.543 | +0.084 |
| 0.503-0.556 | 235 | 0.531 | 0.621 | +0.091 |
| 0.556-0.595 | 234 | 0.576 | 0.585 | +0.010 |
| 0.595-0.630 | 234 | 0.613 | 0.654 | +0.041 |
| 0.630-0.665 | 235 | 0.647 | 0.655 | +0.008 |
| 0.665-0.703 | 234 | 0.685 | 0.658 | -0.027 |
| 0.703-0.739 | 235 | 0.721 | 0.711 | -0.010 |
| 0.739-0.778 | 234 | 0.757 | 0.714 | -0.044 |
| 0.778-1.000 | 235 | 0.809 | 0.689 | -0.119 |

The calibrated column is measured **nested**: for each fold the calibrator was fitted on scores from an extra model that had seen neither that fold nor the fold the calibrator was fitted on, so no held-out row ever contributed to the map applied to it. The one approximation, stated rather than hidden: those extra models were trained on three fifths of the data instead of four fifths, and a model trained on less data is a little less confident, so this understates rather than overstates how much shrinking the shipped head needs.

One consequence of measuring it that way, since it looks like a contradiction otherwise: the *nested* column stitches together a different calibrator per fold, and two rows in different folds can swap order across that seam, so the nested column is not AUC-preserving. What ships is a single global map, and a single monotone map cannot reorder anything — which is why the AUC tables above stand unchanged.

**A calibrator ships inside the checkpoint it belongs to.** `setpiece/assets/shot.pt` stores a `platt` map for its shot probability, under `card["calibration"]`, refitted on the full pooled out-of-fold predictions once the nested measurement had chosen it. `setpiece.calibration.calibrator_from_card` rebuilds one.

The remaining head — defence-wins — ships **no** calibrator, because no candidate improved its held-out ECE. That is a measurement, not an omission.

Calibrators are **not** applied automatically. `score`, `suggest`, `demo` and the app all print the raw sigmoid, so a probability from the CLI today is a ranking score with a documented correction available beside it, not a calibrated probability.

## receiver label threshold

`RECEIVER_MATCH_MAX_M = 8.0` in the frozen `setpiece/_io.py` decides which attacker-first corners get a receiver label at all: the first touch is matched to the nearest candidate attacker in the freeze frame and the row survives only if that candidate is within the threshold. It was a judgement call and had never been checked, so the labels here are re-derived at other thresholds from the `receiver_match_dist` the dataset stores for exactly this purpose, and the *same* out-of-fold predictions are re-scored against them. Nothing is retrained: this measures how sensitive the reported numbers are to the threshold, not how sensitive a refitted model would be. Match distances are short where they are usable at all — median 3.19 m, 75th percentile 5.33 m, 90th 8.78 m, 95th 15.68 m — so the threshold sits far out in the tail rather than through the middle of the distribution. Dropping the threshold entirely would add 119 attacker-first rows to the 923 that survive at 8 m (11.4% of every attacker-first corner with a match), and each of those is a corner whose *nearest visible attacker* was more than 8 m from where the ball was actually touched. That is not a receiver who happened to be mislocated; it is the closest thing on camera to an event whose real actor the frame does not show. Tightening to 4 m costs 280 attacker-first rows, leaving 643, and moves top-1 by GNN -0.005, logistic regression -0.004 and heuristic -0.001.

| threshold | corners with a usable label | attacker-first among them | share of matchable attacker-first rows kept |
|---|---|---|---|
| 4 m | 2065 | 643 | 0.617 |
| 6 m | 2262 | 840 | 0.806 |
| 8 m *(shipped)* | 2345 | 923 | 0.886 |
| 10 m | 2381 | 959 | 0.920 |
| 12 m | 2401 | 979 | 0.940 |
| unthresholded | 2464 | 1042 | 1.000 |

| threshold | GNN top-1 | logreg top-1 | heuristic top-1 | GNN top-3 | logreg top-3 | heuristic top-3 | GNN overall acc. |
|---|---|---|---|---|---|---|---|
| 4 m | 0.420 ±0.005 | 0.413 ±0.005 | 0.157 ±0.000 | 0.719 ±0.013 | 0.720 ±0.005 | 0.482 ±0.000 | 0.711 ±0.007 |
| 6 m | 0.426 ±0.005 | 0.420 ±0.004 | 0.154 ±0.000 | 0.729 ±0.007 | 0.720 ±0.004 | 0.492 ±0.000 | 0.659 ±0.005 |
| 8 m *(shipped)* | 0.425 ±0.003 | 0.418 ±0.004 | 0.158 ±0.000 | 0.726 ±0.008 | 0.715 ±0.004 | 0.496 ±0.000 | 0.638 ±0.004 |
| 10 m | 0.409-0.447 | 0.402-0.439 | 0.152-0.190 | 0.699-0.737 | 0.688-0.726 | 0.478-0.515 | 0.628-0.644 |
| 12 m | 0.401-0.458 | 0.394-0.451 | 0.149-0.206 | 0.685-0.742 | 0.674-0.731 | 0.468-0.525 | 0.623-0.647 |
| unthresholded | 0.377-0.491 | 0.370-0.484 | 0.140-0.254 | 0.643-0.758 | 0.633-0.748 | 0.440-0.554 | 0.607-0.656 |

- Every cell is a mean ±sd over the same 5 repeats as the main table, re-scoring stored out-of-fold predictions. The models are the ones trained under the shipped 8 m labels throughout; only the rows they are scored on change.
- Rows above the shipped threshold are given as a **range, not an estimate**. `setpiece/_io.py` keeps `receiver_match_dist` but drops `receiver_idx` once the match exceeds its own threshold, so for a corner matched at 11 m the dataset records that a match existed and how far it was, but not which attacker it was. The range is every added row wrong at the low end and every added row right at the high end. Loosening the threshold for real needs a re-fetch, not a re-read.
- The defence-wins AUC is not swept: loosening the threshold changes the negative class of that binary problem as well as its size, so a bracket on it would be vacuous rather than informative.
- Across 4 m, 6 m, 8 m no model's top-1 or top-3 moves by more than 0.014 from end to end, so the headline receiver comparison does not rest on the threshold. The model ranking is not quite fixed, though: top-3 reorders somewhere in that range, by a margin smaller than the spread beside it — which is a statement about how close the GNN and the logistic regression are, not about the threshold.
