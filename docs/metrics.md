# metrics

Dataset `data/interim/corners.jsonl` — 2464 corners from 415 matches, 1. Bundesliga, FIFA World Cup, La Liga, Ligue 1, Major League Soccer, UEFA Euro, UEFA Women's Euro, Women's World Cup. Trained 2026-08-25.

Validation is **purged 5-fold cross-validation grouped by `match_id`**: every corner of a match sits in exactly one fold, so no model is ever scored on a match it trained on. The numbers below are pooled out-of-fold predictions — each corner scored exactly once — and both baselines are fitted on the same training folds and scored on the same held-out rows as the GNN.

Rows per fold: [493, 493, 493, 493, 492]. Matches per fold: [83, 83, 83, 83, 83].

Freeze frames carry no velocities, so every model here is `assume_static`.

## receiver head

Scored on 2345 corners with a usable receiver label, of which 923 were won by an attacker. Top-1 and top-3 rank the candidate attackers against each other on the attacker-first subset, which is the only question the nearest-to-goal heuristic can answer. Overall accuracy is the argmax over the candidates *and* the "defence wins first contact" class, across every scored row.

The other 119 corners carry no usable receiver label — almost all of them attacker-first corners whose receiver was not visible in the freeze frame, so there is no node to point at. That filter removes attacker-first rows far more often than defence-wins rows, which leaves the scored subset more defence-heavy than the dataset as a whole. All three models are scored on exactly these rows, so the columns compare cleanly against each other, but the overall-accuracy and defence-AUC figures are not population estimates.

| model | top-1 (attacker-first) | top-3 (attacker-first) | overall acc. incl. defence | defence-wins AUC |
|---|---|---|---|---|
| GNN | 0.427 | 0.724 | 0.643 | 0.623 |
| logistic regression | 0.417 | 0.713 | 0.636 | 0.639 |
| heuristic | 0.158 | 0.496 | 0.606 | 0.476 |

Chance level over the candidates is 0.149 top-1 and 0.441 top-3. Top-3 has a high floor by construction: a corner with three or fewer candidate attackers is a free hit for every model, so read the top-3 column against that floor and not against zero. Always predicting "defence wins" would score 0.606 overall.

## shot head

Binary `shot_within_5`, scored on all 2464 held-out corners. Base rate 0.275. The heuristic is the training-fold shot rate — a constant within a fold, so it carries no information about an individual corner. It lands a little *below* 0.5 rather than on it, and that is mechanical rather than noise: the training-fold rate is a strictly decreasing function of the held-out fold's own rate, so the score is mildly anti-correlated with the label by construction. Read it as the floor; the gap between it and 0.5 is not a signal.

| model | AUC |
|---|---|
| GNN | 0.566 |
| logistic regression | 0.593 |
| heuristic | 0.485 |

## reading

- The GNN wins on top-1 on the attacker-first subset (0.427).
- The logistic regression baseline **beats** the GNN on AUC for the defence-wins class: 0.639 against 0.623.
- The logistic regression baseline **beats** the GNN on shot AUC: 0.593 against 0.566.

Parameters: receiver 118,082, shot 113,857.
