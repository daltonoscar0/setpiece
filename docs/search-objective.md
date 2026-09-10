# the search objective

`setpiece suggest` searches for defender repositionings that lower a shot
probability. It reports a `delta`. This file asks the only question that makes
that number worth printing: **does the delta mean anything to anyone other than
the model that produced it?**

The short answer, measured on 400 real corners:

> **A searched move is better than a random legal move, but only just, and the
> number attached to it does not transfer.** Across model families the
> per-corner correlation between the searching model's delta and the second
> model's delta is **+0.10, with a bootstrap 95% interval that contains zero**.
> About **a fifth** of the claimed improvement survives re-scoring by the other
> family, about **a third** survives re-scoring by the same architecture trained
> at a different seed, and on **90% of corners at least one of twenty random
> legal move sets beats the suggestion** in the other family's eyes. The
> direction is weakly right on average. The magnitude, and the identity of the
> defenders named, are one model's opinion.

`README.md` should carry that caveat next to the `suggest` example. It does not
today.

## why this was worth measuring

`TODO.md` recorded the gap as "`setpiece suggest` optimises the weaker head",
because the logistic-regression baseline used to beat the GNN on shot AUC by
0.027. That framing is now dead: on the current 5-repeat protocol the shot AUCs
are GNN 0.584 ±0.014 and logistic regression 0.586 ±0.009, a gap of 0.002 which
is 0.1x its own spread (`docs/metrics.md`). Neither head is the weaker one.

What survives the reframing is the second half of the entry, and it was always
the more serious half: *no experiment has checked that the suggested moves
survive being re-scored by a different model.* A search does not evaluate a
model's ranking of real corners. It evaluates the model's response to
**counterfactual frames that do not exist in any dataset**, a defender moved
2.6 m north-east, a shape no match ever produced. Out-of-fold AUC says nothing
about that surface, because every row it is measured on is a real frame. Two
models can rank real corners identically and still disagree completely about
which way the probability moves when you nudge a centre-back, and a search
optimises exactly the thing AUC never looked at.

That is also the reason the default objective is **not** changed at the end of
this file: see "why a better ranker is not automatically a better search
target".

## protocol

**Corners.** 400 drawn without replacement from the 2464 in
`data/interim/corners.jsonl` with a fixed sample seed (20260827). All 2464 have
at least three movable outfield defenders, so no corner was excluded. The sample
spans 255 distinct matches and all 8 competitions; median 19 players per frame,
median 10 movable outfield defenders.

**Models.** Three shot models, all fitted on the same 2464 corners under the
same "refit on every corner after cross-validation" protocol the bundled
checkpoints use, at `torch.set_num_threads(1)`:

| id | what it is |
|---|---|
| `gnn_a` | the shipped `setpiece/assets/shot.pt`, `ShotModel`, `head="linear"`, 97,601 parameters, seed 0 |
| `gnn_b` | the same architecture, hyperparameters and rows, seed 4321 |
| `logreg` | the shipped shot baseline: logistic regression on the pooled flat node features, 400 Adam iterations, lr 0.05, weight decay 1e-3 |

They share their training data. That is deliberate, it isolates the second
opinion to a change of family or a change of seed, but it means everything
below is an **upper bound** on how much two genuinely independent models would
agree. A second opinion trained on different matches would agree less, not more.

For context, on the 400 sampled corners the three models do not even agree
closely on the *unmodified* frames:

| | mean | sd | Pearson vs `gnn_a` | Spearman vs `gnn_a` |
|---|---|---|---|---|
| `gnn_a` | 0.3149 | 0.0611 |, |, |
| `gnn_b` | 0.2145 | 0.0808 | 0.764 | 0.740 |
| `logreg` | 0.2683 | 0.0913 | 0.616 | 0.576 |

Two runs of the same recipe correlate at 0.76 on real corners. Across families
it is 0.62. Whatever agreement the deltas show has to live inside that.

**Search.** The shipped defaults, unchanged:
`suggest(frame, model, movers=3, max_move=3.0, min_sep=0.5, beam=8, seed=0, defend=True)`,
`max_evals=288`, keeper excluded, `max_candidates=6`, `deep_candidates=2`. Eight
suggestions come back per corner; the best one uses 3.00 movers on average when
searching `gnn_a` and 2.99 when searching `logreg`.

**Re-scoring.** The winning move set is replayed with `apply_moves` and the
resulting frame is scored by every model. Every delta below is
`p(moved) - p(base)` under the model named, on the identical shape. Negative is
a defensive improvement.

**The null.** For each corner, 20 random legal move sets under the *same* hard
constraints and the same mover count as the searched best: three distinct
movable defenders (keeper excluded), each given one displacement drawn uniformly
from the identical 16-member candidate set `candidate_displacements(3.0, 0)`,
rejected unless it keeps every pair at least 0.5 m apart and stays on the pitch.
8000 draws in all. A robustness variant restricts the draw to the six defenders
nearest the goal, the pool the search itself can reach, and is reported below.

## 1. search against the GNN, re-score with something else

| | second model = `gnn_b` (same architecture, new seed) | second model = `logreg` (different family) |
|---|---|---|
| Pearson r of the two deltas | **+0.395** [+0.251, +0.517] | **+0.098** [-0.046, +0.241] |
| Spearman rho | +0.294 | -0.003 |
| sign agreement | 0.905 (p = 2e-67) | 0.623 (p = 1.1e-06) |
| median delta, searching model | -0.05920 | -0.05920 |
| median delta, second model | -0.01760 [-0.01950, -0.01586] | -0.01214 [-0.01539, -0.00629] |
| median delta of random legal moves, second model | -0.00075 | +0.00131 |
| second model calls it an improvement on | 0.905 of corners | 0.623 of corners |
| ... random legal moves manage | 0.580 | 0.463 |
| searched beats that corner's median random draw | 0.897 (p = 1.5e-64) | 0.645 (p = 7.1e-09) |

Bracketed figures are 4000-sample bootstrap 95% intervals. The p-values are
exact two-sided binomial sign tests over the 400 corners.

The cross-family Pearson interval **contains zero**. Rank correlation is
literally nothing: -0.003. A corner where the GNN found a large improvement is
not a corner where the logistic regression sees a large improvement.

## 2. the reverse: search against the logistic baseline

| | second model = `gnn_a` | second model = `gnn_b` |
|---|---|---|
| Pearson r of the two deltas | **+0.097** [-0.056, +0.245] | **+0.224** [+0.105, +0.343] |
| Spearman rho | +0.021 | +0.163 |
| sign agreement | 0.730 (p = 1.2e-20) | 0.795 (p = 6.8e-34) |
| median delta, searching model | -0.08756 | -0.08756 |
| median delta, second model | -0.00861 [-0.01100, -0.00686] | -0.00712 [-0.00825, -0.00574] |
| median delta of random legal moves, second model | -0.00378 | -0.00075 |
| second model calls it an improvement on | 0.730 of corners | 0.795 of corners |
| ... random legal moves manage | 0.755 | 0.580 |
| searched beats that corner's median random draw | 0.623 (p = 1.1e-06) | 0.745 (p = 2.3e-23) |

Symmetrical, and just as weak. Note the row that looks wrong and is not: under
`gnn_a`, random legal moves are called an improvement on 0.755 of corners while
the logistic regression's own best suggestion manages 0.730. Sign agreement is
worthless as a statistic here precisely because the GNN thinks almost any
perturbation helps, see the next section.

Neither direction is better than the other. Whichever head does the searching,
the other family recovers roughly a seventh of what it thinks was available on
that frame:

- `gnn_a`'s suggestion scores -0.01214 under `logreg`, against the -0.08756
  `logreg` reaches when it searches for itself: **14% captured**.
- `logreg`'s suggestion scores -0.00861 under `gnn_a`, against `gnn_a`'s own
  -0.05920: **15% captured**.

And they largely do not name the same players. Mean Jaccard overlap of the moved
defender sets is **0.419**; the two searches choose an identical set of three
defenders on **7.2%** of corners, though on 100% of corners they share at least
one, which is mostly the candidate restriction talking, since both draw from
the same six defenders nearest the goal.

## 3. the null, which is the part that matters

**Random legal moves are not neutral, and that is the first finding.** Over 8000
draws:

| model | mean delta | median | fraction negative | sd |
|---|---|---|---|---|
| `gnn_a` | -0.00448 | -0.00380 | 0.599 | 0.02018 |
| `gnn_b` | -0.00083 | -0.00082 | 0.534 | 0.01556 |
| `logreg` | +0.00229 | +0.00162 | 0.475 | 0.03632 |

The displacement set is symmetric about zero, so a locally linear model would
give a mean of about zero. `gnn_a` instead calls **60% of arbitrary legal
perturbations a defensive improvement**. Some of any suggestion's headline delta
is therefore not "this is the right adjustment" but "moving defenders at all
looks good to this network". That alone is a reason not to read a `suggest`
delta as a coaching result.

Against that null, the searched moves do win. Expressed as the suggestion's
percentile among its own corner's 20 random draws, 0.500 would mean
indistinguishable from random, 0.000 means it beat every draw:

| searched against | judged by `gnn_a` | judged by `gnn_b` | judged by `logreg` |
|---|---|---|---|
| `gnn_a` | **0.005** | 0.159 | **0.391** |
| `logreg` | **0.401** | 0.321 | **0.001** |

Read the diagonal first: under its own objective the search is emphatic, beating
99.5% of random draws. Read the off-diagonal second. **Under the logistic
regression, a random legal move set beats the GNN's best suggestion 39.1% of the
time**, against the 50% it would manage if the search carried no cross-family
information whatever. Put the other way round: on **90.5%** of corners at least
one of twenty random draws beats the GNN's suggestion in the logistic
regression's eyes. Under its own model that happens on 7.7%.

The paired comparisons are still significant, so the honest verdict is *not*
"the search does nothing":

| arm | searched beats that corner's median random draw |
|---|---|
| search `gnn_a`, judge `gnn_b` | 0.897 (p = 1.5e-64) |
| search `gnn_a`, judge `logreg` | 0.645 (p = 7.1e-09) |
| search `logreg`, judge `gnn_a` | 0.623 (p = 1.1e-06) |
| search `logreg`, judge `gnn_b` | 0.745 (p = 2.3e-23) |
| search `gnn_a`, judge `gnn_a` (control) | 1.000 (400/400) |
| search `logreg`, judge `logreg` (control) | 1.000 (400/400) |

**Robustness of the null.** Drawing the random defenders from all ten movable
ones gives the search a possible unfair edge, because the search only ever
considers the six nearest the goal. Repeating the null restricted to those six -
the search's own pool, changes nothing that matters:

| judged by | matched-pool random median | all-defender random median | searched (`gnn_a`) beats matched-pool median |
|---|---|---|---|
| `gnn_a` | -0.00490 | -0.00378 |, |
| `gnn_b` | -0.00053 | -0.00075 | 0.920 (p = 1.7e-73) |
| `logreg` | -0.00026 | +0.00131 | 0.660 (p = 1.5e-10) |

and for the logistic search, 0.608 judged by `gnn_a` (p = 2e-05) and 0.780
judged by `gnn_b` (p = 1.8e-30). The tighter null is if anything slightly kinder
to the search.

## how much of the claimed improvement survives

| what the searching model claims | what `gnn_b` sees | what `logreg` sees |
|---|---|---|
| `gnn_a`: -0.05920 | -0.01760 (**30%**) | -0.01214 (**21%**) |
| `logreg`: -0.08756 | -0.00712 (**8%**) |, |

Net of the random-move baseline the surviving excess is -0.0168 (28% of the
claim) for `gnn_b` and -0.0135 (23%) for `logreg`.

Scale matters as much as sign. The GNN's median searched delta is 0.97 of its
*own* spread across real corners, a full standard deviation, which is why the
CLI prints an impressive-looking number, but only 0.22 of `gnn_b`'s spread and
0.13 of `logreg`'s. A suggestion that moves the shipped head by a whole standard
deviation moves an equally good model by a tenth of one.

The same thing again as within-corner rank agreement, over all 36 move sets
scored per corner (8 GNN suggestions + 8 logreg suggestions + 20 random draws),
averaged over the 400 corners:

| pair | mean Spearman | median | fraction positive |
|---|---|---|---|
| `gnn_a` / `gnn_b` | **+0.626** | +0.715 | 0.950 |
| `gnn_b` / `logreg` | +0.221 | +0.210 | 0.767 |
| `gnn_a` / `logreg` | **+0.039** | +0.044 | 0.557 |

Two seeds of the same architecture rank a corner's candidate shapes at +0.63.
Two families rank them at +0.04, which is to say **not at all**. The search is
ranking counterfactual shapes on a surface the other family cannot see.

## an ensemble objective was tried, and is not being adopted

The obvious repair is to search against the mean of the two families and hope
the moves generalise. Measured, on the same 400 corners: objective
`0.5 * (gnn_a + logreg)`, judged by `gnn_b`, which is in neither.

| objective | median delta under the held-out `gnn_b` | fraction negative |
|---|---|---|
| `0.5 * (gnn_a + logreg)` | -0.01534 | 0.915 |
| `gnn_a` alone | **-0.01760** | 0.905 |
| `logreg` alone | -0.00712 | 0.795 |
| random legal | -0.00075 | 0.580 |

The ensemble **loses** to `gnn_a` alone under the held-out judge, on 232 of 400
corners (p = 0.0016). What it does buy is agreement with the logistic
regression: -0.07660 on the logreg's scale against the -0.01214 the GNN-alone
search manages, at a cost of 38% of the GNN's own claimed delta
(-0.03674 against -0.05920).

That is a trade between two members, not an improvement, and the only judge
outside the objective prefers the single GNN. It also doubles the cost of every
evaluation, which the pitch editor's one-second round trip cannot absorb. So
the ensemble stays out.

## why a better ranker is not automatically a better search target

At the time of writing there is no "better ranker" to promote, the shot AUCs
are 0.584 ±0.014 and 0.586 ±0.009, differing by 0.1x their spread. But the
argument matters independently of which way that gap points, so it is recorded
here rather than left implicit:

1. **AUC is measured on real frames; a search lives on counterfactual ones.**
   Every held-out row in `docs/metrics.md` is a shape some team actually
   produced. `suggest` spends 288 of its 289 evaluations on shapes no team ever
   produced. A model can be perfectly ordered on the first set and arbitrary on
   the second, and nothing in the cross-validation would notice.
2. **The measurements above say the two are close to unrelated.** The two
   families rank real corners at Spearman 0.576 and rank a corner's counterfactual
   shapes at Spearman 0.039. Whichever of them ranks real corners better, that
   fact predicts almost nothing about which of them ranks *moves* better.
3. **Nothing here can adjudicate move quality, because there is no label for it.**
   The dataset records what happened, not what would have happened had a
   defender stood two metres to his left. Both searches beat random under a
   foreign model, and there is no third measurement that separates them. Swapping
   the objective on the strength of a 0.002 AUC difference would be dressing an
   arbitrary choice as a finding.
4. **The logistic baseline is the more volatile search target, not the safer
   one.** Its response to a random legal perturbation has sd 0.036 against the
   GNN's 0.020, and its median searched delta is -0.088 against -0.059. It is
   easier to push around, which is not the same as being right about which push
   helps.

The default objective therefore stays as it is: the bundled GNN shot head.
`suggest` already accepts any `frame -> float` callable, so a user who wants to
search the logistic baseline, an ensemble or their own model can pass one.

## verdict

- **The search works, in the narrow sense.** It beats a random legal move under
  every foreign model tested, on every arm, with p-values between 1e-06 and
  1e-73, and the finding holds when the null is restricted to the search's own
  candidate pool. It is not a random number generator.
- **Cross-model agreement is weak, and a delta is one model's opinion.** The
  cross-family correlation is +0.10 with a 95% interval containing zero, the
  cross-family rank correlation over counterfactual shapes is +0.04, and only
  21% of the claimed improvement is visible to the other family. Even a fresh
  seed of the identical architecture keeps only 30%.
- **The specific advice does not transfer either.** The two families agree on
  which three defenders to move on 7% of corners.
- **Part of the delta is a perturbation artefact.** The shipped head calls 60%
  of arbitrary legal defender moves an improvement, with a median of -0.0038.
  Subtract that before reading any single suggestion.

Practical reading of a `suggest` output: treat the *direction* as a weak signal
and the *number* as uninterpretable. A delta of -0.06 from the CLI is worth
roughly -0.012 to -0.018 to an equally defensible model, against about -0.001 to
-0.004 for moving three defenders at random. `README.md` and `docs/usage.md`
should say so where the delta is first shown.

## reproducing this

Everything above comes from the bundled `setpiece/assets/shot.pt` plus two
models fitted with the public helpers in `setpiece.train`:

```python
rows = train.load_rows("data/interim/corners.jsonl")
logreg = train._fit_shot_logreg(rows, iters=400, lr=0.05, weight_decay=1e-3, seed=0)
gnn_b  = train._train_shot(rows, epochs=20, hidden=64, layers=3, dropout=0.1,
                           lr=3e-3, weight_decay=1e-4, batch_size=32, seed=4321)
```

with `torch.set_num_threads(1)`, then `suggest()` at the defaults and
`apply_moves()` to replay each winning move set for re-scoring. The random null
uses `candidate_displacements(3.0, 0)` and the same admissibility rule as
`setpiece.suggest._admissible`; `tests/test_suggest.py` carries a checked
implementation of the draw and asserts it produces only legal move sets, and
asserts the control result, that the search beats every random draw under its
own objective.

Cost, for planning: fitting `gnn_b` took 57 s, the 400-corner main run 159 s
(0.40 s per corner for two searches plus 108 re-scorings), the ensemble arm
146 s and the matched-pool null 21 s. All single-threaded on a 10-core machine
whose 1-minute load average sat between 4.0 and 6.1 throughout, several other
jobs were running, so treat these as upper bounds rather than benchmarks.
