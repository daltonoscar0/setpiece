# setpiece

A set-piece coach's tool for corner kicks: it scores who is most likely to win
the first contact on the delivery and how likely a shot is in the next five
events, then searches for small defensive repositionings that lower that shot
probability. The models are graph neural networks over StatsBomb 360 freeze
frames from the public open-data release.

- `setpiece score` — per-player probability of first contact, plus P(shot).
- `setpiece suggest` — ranked defender adjustments, each at most 3 m.
- `setpiece app` — a Streamlit pitch editor: drag a player, re-score on drop.

Everything runs on CPU. Scoring one corner with both heads takes about 6 ms
(measured over the 18 corners in `data/interim/smoke.jsonl`, single-threaded, on
a busy ten-core machine).

---

## Install

The project is managed with [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/daltonoscar0/setpiece
cd setpiece
uv sync                 # core: torch, torch-geometric, numpy
uv sync --extra app     # adds streamlit + plotly, needed only for `setpiece app`
```

Python 3.11 or 3.12. Trained checkpoints are committed under
`setpiece/assets/`, so scoring works straight after `uv sync` — no download and
no training needed.

## Quickstart

### 1. `demo` — three real Euro 2024 corners, offline

```
uv run setpiece demo
```

It touches no network. Scoring and searching the three corners takes about
0.5 s; the rest is importing torch and loading the checkpoints, which puts a
warm run at roughly 5 s end to end.

One caveat worth knowing: the very first run in a freshly created venv pays a
one-off cost to page torch in from disk, and that run has been measured between
46 s and 60 s on a clean clone here, dropping to about 5 s once warm. That is a
property of importing torch, not of the model — inference itself is a few
milliseconds per corner. Abridged output:

```
setpiece demo: three real Euro 2024 corners, scored offline
======================================================================

corner e5c0796a  (UEFA Euro, 2024, match 3930159, 71')
20 players: 9 attacking, 11 defending

 #  role         x     y  P(first contact)
--  -------  -----  ----  ----------------  ----------------
14  attack   100.5  14.5             0.211  ###.............
 8  attack    97.4  36.5             0.081  #...............
13  attack    99.3  31.7             0.076  #...............
 4  attack    95.9  41.6             0.059  #...............
 7  attack    97.4  32.3             0.053  #...............
 5  attack    96.7  32.9             0.045  #...............
 2  attack    85.2  37.5             0.007  ................
 0  attack    81.9  13.1             0.002  ................
 -  defence                          0.467  #######.........

P(shot within 5 events)  0.238  ####............
actual: first contact attacker (attacker #8), shot within 5: True
positions only; freeze frames carry no velocities (--assume-static)

corner e5c0796a  (UEFA Euro, 2024, match 3930159, 71')
moving up to 2 defenders to lower the shot probability
search width: 6 defenders for the first move, 2 for later moves, nearest the goal first; keeper excluded; budget 64 evaluations (--max-candidates / --deep-candidates / --move-keeper / --max-evals)
base P(shot) 0.238

#  P(shot)   delta  moves
-  -------  ------  ------------------------
1    0.224  -0.013  #15 -> 99.1,30.5 (3.0m)
2    0.226  -0.011  #17 -> 100.4,28.3 (3.0m)
3    0.226  -0.011  #15 -> 98.5,32.7 (3.0m)

static repositioning at the moment of the delivery; no player is assumed to cover ground (--assume-static)
----------------------------------------------------------------------
...
scored and searched 3 corners in 0.5s (plus 4.9s to import torch and load the checkpoints), no network access
```

`--format json` and `--format csv` are available on `demo`, `score` and
`suggest`.

### 2. `score` — one corner of your own

Write the shape to a CSV (columns and units are in
[Input formats](#input-formats) below) and score it:

```
uv run setpiece score corner.csv
```

### 3. `suggest` — what to change

```
uv run setpiece suggest corner.csv --movers 2 --top-k 3
```

Flags: `--defend` (the default, minimise P(shot)), `--attack` (maximise it, to
show an attacking coach where the shape breaks), `--movers N`, `--max-move M`
(metres), `--beam N`, `--top-k N`, `--seed N`, `--format`, `--model PATH`, and
the search-width group `--max-candidates N|all`, `--deep-candidates N|all`,
`--move-keeper`, `--max-evals N`. The search is deterministic: same corner, same
seed, same output.

**What a `suggest` delta means.** `docs/search-objective.md` measures this on
400 real corners, and the answer is narrow. A searched move beats a random legal
move under every second model tested (p between 1e-06 and 1e-73), so the
direction is a real signal — but the *number* does not transfer. The
cross-family correlation between the searching model's delta and a logistic
regression's delta on the same moved frame is +0.10 with a bootstrap 95%
interval containing zero, only 21% of the claimed improvement is visible to that
other family (30% to a fresh seed of the same architecture), the two searches
pick the same three defenders on 7% of corners, and the shipped head calls 60%
of *arbitrary* legal defender moves an improvement, median -0.0038. So a
`-0.06` on the CLI is worth roughly `-0.012` to `-0.018` to an equally
defensible model. Read the direction, not the magnitude, and never read it as a
measured change in a real shot probability — nothing in this dataset labels what
would have happened had a defender stood two metres to his left.

### 4. `app` — the pitch editor

```
uv sync --extra app
uv run setpiece app            # add --headless to stay out of the browser
```

Serves on <http://localhost:8501> (`--port` to change it). Pick one of the
bundled corners or upload a JSON/CSV in the sidebar, drag players on the pitch,
watch the probabilities move, press **suggest defender moves** to overlay the
proposed adjustments, **apply these moves** to commit them, and **download PNG**
to export the annotated pitch. The model cards are in an expander at the bottom.

## Rebuilding the models from scratch

`data/` is gitignored, so a fresh clone has the checkpoints but not the dataset.
To rebuild both:

```
uv run setpiece fetch          # walk StatsBomb open data, extract corners
uv run setpiece train          # 5-fold purged CV over 5 seeds, then refit
```

`fetch` walks every open competition-season flagged for 360 data, downloads the
events and freeze frames one match at a time (rate limited to one request per
second, with retries), extracts the corners that carry a frame, and writes JSON
lines to `data/interim/corners.jsonl`. Raw files are discarded after extraction
unless you pass `--keep-raw`. Progress is checkpointed in
`data/interim/corners.state.json` after every tenth match, so an interrupted
build resumes instead of re-downloading; the same ledger counts corners seen
against corners kept per season, which is where the 360 coverage figures below
come from. A match that fails is retried at most three times and then given up
on, and its `failures` row is updated in place rather than appended to.
`--competition NAME` (repeatable) and `--max-matches N` narrow it:

```
$ uv run setpiece fetch --competition "UEFA Euro" --max-matches 1 --out data/interim/sample.jsonl
building dataset: 1 matches to fetch
1/1 matches, 3 corners
3 corners from 1 matches -> data/interim/sample.jsonl
```

The full build is several hundred matches at one request per second, so budget
an hour or so of mostly waiting.

`train` cross-validates both heads and both baselines, repeats the whole
protocol over five seeds and reports `mean ±sd`, prints the metrics table,
writes it to `docs/metrics.md`, and saves `receiver.pt`, `shot.pt` and
`report.json` to `--out-dir` (default `runs/latest`). Useful flags: `--dataset`,
`--out-dir`, `--folds`, `--seeds`, `--epochs`, `--hidden`, `--layers`,
`--dropout`, `--lr`, `--weight-decay`, `--batch-size`, `--seed`, `--limit N`,
`--quiet`, `--no-calibrate`, `--no-threshold-sweep`, and `--no-metrics-file` to
print the table without touching `docs/metrics.md`.

Plain `train` writes to `--out-dir` and does **not** touch the bundled
checkpoints in `setpiece/assets/`. Score against a fresh run with
`--model runs/latest`; pass `--install` when you want the refitted checkpoints
copied over the bundled ones. `--install` will happily overwrite good
checkpoints with a two-fold smoke run, so pair it with a real `--dataset`.

## Input formats

Two things are accepted wherever a command takes `<corner>`:

**StatsBomb 360 JSON.** A `three-sixty` entry (an object with `event_uuid` and
`freeze_frame`), a bare `freeze_frame` list, or a one-element list of either.
`teammate`, `actor` and `keeper` map to attacker, taker and goalkeeper, and the
120x80 StatsBomb coordinates are rescaled to metres at the loader boundary.
`setpiece`'s own serialisation (`{"corner_id": ..., "players": [...]}`) is
accepted too, as is a whole `.jsonl` or `.ndjson` dataset — every corner in the
file is scored.

**Plain CSV.** Columns `team,x,y,is_taker` with an optional `is_keeper`:

| column | meaning |
|---|---|
| `team` | `attack` / `defend` (also `att`/`def`, `a`/`d`, `1`/`0`) |
| `x`, `y` | metres on a 105x68 pitch |
| `is_taker` | `1` for the player taking the corner, else `0` |
| `is_keeper` | `1` for a goalkeeper, else `0` (optional) |

**Coordinates are metres on a 105x68 pitch, the attacking team always attacking
towards +x.** The goal being attacked is centred at `(105, 34)`, its posts at
`y = 30.34` and `y = 37.66`, and the corner flags are at `(105, 0)` and
`(105, 68)`.

Flag exactly one attacker as the taker. If nothing is flagged, the delivery
origin falls back to whichever corner flag is nearest to the shape — a
geometric inference, never an invented player — but the unflagged taker is then
treated as an ordinary attacker and will be scored as a candidate receiver,
which is wrong and obvious in the output.

### A worked example

Save this as `corner.csv` — a delivery from the flag at `(105, 68)`, six
attackers arriving in front of goal plus the taker, ten defenders including the
keeper on his line:

```csv
team,x,y,is_taker,is_keeper
attack,104.8,67.6,1,0
attack,99.5,36.5,0,0
attack,98.2,33.0,0,0
attack,97.0,38.5,0,0
attack,94.5,34.5,0,0
attack,92.0,29.0,0,0
attack,88.0,34.0,0,0
defend,103.6,34.0,0,1
defend,100.0,30.9,0,0
defend,99.8,37.2,0,0
defend,98.5,35.0,0,0
defend,97.6,31.5,0,0
defend,96.0,38.0,0,0
defend,94.0,32.0,0,0
defend,93.5,36.5,0,0
defend,90.0,34.0,0,0
defend,86.0,40.0,0,0
```

```
$ uv run setpiece score corner.csv
corner corner
17 players: 7 attacking, 10 defending

#  role        x     y  P(first contact)
-  -------  ----  ----  ----------------  ----------------
3  attack   97.0  38.5             0.132  ##..............
1  attack   99.5  36.5             0.119  ##..............
2  attack   98.2  33.0             0.082  #...............
4  attack   94.5  34.5             0.081  #...............
5  attack   92.0  29.0             0.040  #...............
6  attack   88.0  34.0             0.002  ................
-  defence                         0.545  #########.......

P(shot within 5 events)  0.281  ####............
positions only; freeze frames carry no velocities (--assume-static)
```

The taker is absent from the table because the taker is never a candidate to
receive their own corner; the probabilities over the listed attackers and the
`defence` row sum to 1. Then:

```
$ uv run setpiece suggest corner.csv --movers 2 --top-k 3
corner corner
moving up to 2 defenders to lower the shot probability
search width: 6 defenders for the first move, 2 for later moves, nearest the goal first; keeper excluded; budget 224 evaluations (--max-candidates / --deep-candidates / --move-keeper / --max-evals)
base P(shot) 0.281

#  P(shot)   delta  moves
-  -------  ------  -----------------------------------------------
1    0.234  -0.047  #13 -> 91.6,30.1 (3.0m); #8 -> 97.6,29.1 (3.0m)
2    0.234  -0.047  #13 -> 91.6,30.1 (3.0m); #9 -> 98.0,39.6 (3.0m)
3    0.236  -0.046  #13 -> 93.6,29.0 (3.0m); #9 -> 98.0,39.6 (3.0m)

static repositioning at the moment of the delivery; no player is assumed to cover ground (--assume-static)
```

That `-0.047` is one model's opinion about a shape no match ever produced. See
"What a `suggest` delta means" above before quoting it to anybody.

`--format csv` gives one row per player (plus a `defence_first_contact` row per
corner) and `--format json` gives the full structure, including the input
coordinates, the search width actually used, and an `assume_static: true` flag.

## What this model knows, and what it does not

Read this before showing a number to a coach.

**Freeze frames are static. There are no velocities anywhere, and none are
invented.** A StatsBomb 360 frame is a single snapshot at the moment the corner
is struck. Nothing in this codebase estimates, interpolates or fabricates a
velocity, so the models cannot know who is already moving, and the suggestion
search has no idea whether a defender could actually cover the ground it
proposes. Every output is annotated `--assume-static` for that reason. A
suggested move is "stand two metres deeper at the moment of the delivery", not
"make this run".

**Frames are anonymous and incomplete.** There are no player identities, only
positions and three flags (teammate, actor, keeper), so the model cannot know
that a particular centre-back is 1.95 m tall or that a particular attacker wins
everything at the near post. Frames are also variable size: only the players
inside the broadcast camera's field of view are captured. Across the 2464
corners in this dataset that is 8 to 22 visible players, median 19, and 2 to 11
attackers, median 8. Someone is often missing. Frames showing fewer than eight
players are dropped at load time; over the full walk that discarded 4 corners.

**Just over half of open-data corners have a frame, and this is now measured
rather than guessed.** The fetch ledger counts corners seen against corners
kept: **2468 of the 4467 corners walked came with a 360 freeze frame, 55.2%**.
Coverage is not uniform and is not something anyone here can choose — it is
whether StatsBomb published a frame. Among competitions that ship any frames at
all it runs from **43.9% (Major League Soccer) to 76.0% (La Liga)**. African Cup
of Nations 2023 is flagged `match_available_360` in the catalogue and ships
**no** freeze frames at all — 0% of its 460 corners — so none of it is in the
dataset. The per-competition table is in `docs/metrics.md`. Corners without a
frame are skipped entirely and the models never see them.

**The receiver label is noisy, because it is recovered positionally.** Freeze
frames carry no identities, so there is no direct way to say "player 7 received
it". The loader takes the first genuine touch after the corner and matches its
location to the *nearest attacking node in the frame*. That is an inference, not
a fact: if two attackers are a metre apart the wrong one may be credited, and if
the real receiver was outside the camera view the nearest visible attacker would
be credited instead. Matches further than 8 m from the touch are treated as
ambiguous and **dropped** from receiver training and evaluation rather than
guessed at. Of 2464 corners, 918 carry a clean receiver label. That 8 m is a
judgement call, not a fitted number, and `docs/metrics.md` sweeps it: across
4 m, 6 m and 8 m no model's top-1 or top-3 moves by more than 0.013 end to end
and the ranking of the three models is identical at every threshold, so the
headline receiver comparison does not rest on it.

**Both label windows are event counts, and both carry a plausibility guard.**
`first_touch` is the first genuine touch in a 12-event lookahead and
`shot_within_5` a 5-event window, per the spec — but an event-count window walks
straight through a dead ball and will happily call the restart twelve events
later "the first touch on this corner". So a touch more than 10 s after the
delivery (`FIRST_TOUCH_MAX_S`) or a shot more than 15 s after it (`SHOT_MAX_S`)
is discarded. The guards bite: the first touch normally lands a median 1.61 s
after the delivery (p90 2.53 s), but 110 rows sat beyond the guard, out to
238 s, and 102 of those were being scored as *the defence won first contact*. A
discarded touch is dropped, never relabelled, which is what the 111-corner
`unknown` class is — corners this dataset declines to score rather than guesses
at. The event-count definition itself survives the check: replacing the 5-event
shot window with a 5-second one flips 2.3% of the shot labels, so the two rules
**agree on 97.7% of corners**.

**The shot label is coarse.** `shot_within_5` is true when the attacking team
shoots within the five events following the corner, and within 15 s. It says
nothing about shot quality, and it inherits whatever the event feed decided
counts as an event. The base rate is 0.268.

**The probabilities are ranking scores, not calibrated frequencies.** Two of the
outputs are genuine binary probabilities and were checked against observed
frequencies: the shot head's raw expected calibration error is 0.018 and the
defence-wins head's 0.051. No calibrator beat the raw head on held-out ECE for
either, so **neither checkpoint ships one**, and `score`, `suggest`, `demo` and
the app all print the raw sigmoid. `setpiece/calibration.py` has the machinery
and `docs/metrics.md` the reliability tables if you want to fit your own.

**The suggestion search is a search over one model's opinion.** It optimises the
GNN shot head over counterfactual frames — shapes no match ever produced — and
cross-validated AUC says nothing about that surface, because every row AUC is
measured on is a shape some team really made. `docs/search-objective.md`
measures what survives a second opinion; the summary is in "What a `suggest`
delta means" above.

## The receiver target

The receiver head is a softmax over the candidate attackers **plus one extra
class: "defence wins first contact"**. Candidates are the attackers in the frame
excluding the taker.

That extra class is not a modelling convenience, it is the most common outcome
in the data. Over 2464 corners the defence makes the first contact 1320 times
against the attack's 1033, with 111 corners unlabelled by the plausibility
guards. A model that could only rank attackers would be answering the wrong
question for the majority of corners, and a coach asking "who is going to get on
the end of this?" needs "probably nobody, it gets cleared" to be an available
answer. It is also why the metrics below report top-1 and top-3 on the
attacker-first subset separately from overall accuracy: the nearest-to-goal
heuristic can only rank attackers, so that subset is the only place all three
models can be compared like for like.

Rows where the attack won the first contact but the positional match was
ambiguous (`receiver_match_dist > 8 m`) are dropped, not relabelled. So is the
attacking team's own goalkeeper taking the first touch — that is scored as
`"defender"`, deliberately, since an attacking keeper is not receiving his own
corner.

## Results

From `docs/metrics.md`, which is written by `setpiece train` and holds measured
numbers, not targets.

Dataset: `data/interim/corners.jsonl`, 2464 corners from 415 matches across
eight competitions (1. Bundesliga, FIFA World Cup, La Liga, Ligue 1, Major
League Soccer, UEFA Euro, UEFA Women's Euro, Women's World Cup), seasons 2020
to 2025.

Split scheme: **purged 5-fold cross-validation grouped by `match_id`**. Every
corner of a match sits in exactly one fold, so no model is scored on a match it
trained on. The numbers are pooled out-of-fold predictions — each corner scored
exactly once — and both baselines are fitted on the same training folds and
scored on the same held-out rows as the GNN. Folds hold 83 matches each; rows
per fold are [493, 493, 493, 493, 492].

**Every figure is a mean over 5 repeats of the whole protocol, written
`mean ±sd`.** Seeds [0, 1000, 2000, 3000, 4000] each re-draw the fold assignment
*and* the network initialisation, so the spread covers both of the things that
change on a re-run. Everything is run at `torch.set_num_threads(1)`: torch's
parallel reductions sum in thread-partition order, so the GNN column is only
reproducible at a pinned thread count, and each checkpoint's card records the
count it was trained at as `training.torch_threads`.

**Read the spreads before the means.** Where two models are compared, the gap is
differenced *within* each repeat first, which cancels the shared split noise and
is a tighter test than eyeballing two columns.

### Receiver head

Scored on 2238 corners with a usable receiver label, of which 918 were won by an
attacker. The other 226 are corners nobody can score: 115 attacker-first corners
whose receiver was not visible in the frame, and the 111 `unknown` rows the
plausibility guards removed. Neither filter is neutral — both drop attacker-first
rows far more often than defence-wins rows — so the scored subset is more
defence-heavy than the dataset as a whole (0.590 against 0.536). All three
models are scored on exactly these rows, so the columns compare cleanly against
each other, but the overall-accuracy and defence-AUC figures are **not**
population estimates.

| model | top-1 (attacker-first) | top-3 (attacker-first) | overall acc. incl. defence | defence-wins AUC |
|---|---|---|---|---|
| GNN | 0.432 ±0.012 | 0.729 ±0.015 | 0.625 ±0.006 | 0.631 ±0.007 |
| logistic regression | 0.432 ±0.001 | 0.734 ±0.004 | 0.625 ±0.002 | 0.636 ±0.005 |
| nearest-to-goal heuristic | 0.159 ±0.000 | 0.498 ±0.000 | 0.590 ±0.000 | 0.473 ±0.012 |

Chance over the candidates is 0.150 top-1 and 0.441 top-3 (the top-3 floor is
high because a corner with three or fewer candidate attackers is a free hit for
everyone); always predicting "defence wins" scores 0.590 overall.

### Shot head

Binary `shot_within_5`, scored on all 2464 held-out corners, base rate 0.268.
The heuristic is the training-fold shot rate — a constant within a fold, so it
carries no information about an individual corner and is here as the floor. Its
0.481 sits below 0.5 for a mechanical reason rather than by chance: the
training-fold rate is a strictly decreasing function of the held-out fold's own
rate, so a constant-prior score is mildly anti-correlated with the label by
construction. The gap between 0.481 and 0.5 is not a signal.

| model | AUC |
|---|---|
| GNN | 0.584 ±0.014 |
| logistic regression | 0.586 ±0.009 |
| shot-rate prior | 0.481 ±0.010 |

### What is actually established

**Both learned models rank receivers far better than the geometric floor, and
that is the whole result.** The GNN's top-1 of 0.432 against the heuristic's
0.159 and chance's 0.150 is a wide, unambiguous margin, and the same holds on
top-3 (0.729 against 0.498, on a 0.441 chance floor). The nearest-to-goal
heuristic is a **floor, not a competitor**: it is barely distinguishable from
guessing, and quoting "GNN 0.729 against a baseline's 0.498" as if the second
number were a rival model would be misleading.

**Every GNN-versus-logistic-regression gap in both tables is inside the noise
floor.** Differenced seed by seed over the 5 repeats:

| comparison | gap | its own spread | ratio | reading |
|---|---|---|---|---|
| receiver top-1, GNN − logreg | 0.000 | ±0.012 | 0.0x | inside the noise floor |
| receiver top-3, logreg − GNN | 0.004 | ±0.018 | 0.2x | inside the noise floor |
| defence-wins AUC, logreg − GNN | 0.005 | ±0.011 | 0.4x | inside the noise floor |
| shot AUC, logreg − GNN | 0.002 | ±0.012 | 0.1x | inside the noise floor |

So: **the GNN and the logistic regression are tied, on all four numbers.**
Neither beats the other on this dataset, and no sentence in this repository
should say otherwise. Earlier versions of this README claimed the logistic
baseline "beats" the GNN on shot AUC on the strength of a single run; with five
repeats that gap is 0.1x its own spread, which is not a result. The message
passing is not buying anything measurable over a pooled linear model here.

The shot head is weak in absolute terms besides — 0.584 AUC on a 0.268 base
rate — and `setpiece suggest` optimises it anyway, because that is the model
wired into the search. `docs/search-objective.md` explains why swapping to a
0.002-AUC-different objective would be dressing an arbitrary choice as a
finding, and measures what a delta is worth regardless.

The honest summary: this tool ranks receivers well against a geometric floor,
predicts shots poorly, and has not shown that its graph network earns its keep
over a linear baseline. See `TODO.md`.

Model sizes: receiver 118,082 parameters, shot 97,601.

## Data, licence and attribution

The code in this repository is MIT licensed, as declared in `pyproject.toml`
and in `LICENSE`.

The data is not. Everything the models are trained on comes from
[StatsBomb Open Data](https://github.com/statsbomb/open-data), which StatsBomb
publish free of charge under the **StatsBomb Public Data User Agreement**
(`LICENSE.pdf` in that repository). Read that document before you publish
anything derived from the data; do not rely on this paragraph as a summary of
it. The condition it is best known for, and the one that applies to this
repository, is attribution: work made public using the data must credit
StatsBomb as the source and display the StatsBomb logo. Other conditions,
including limits on commercial use, are in the agreement itself.

`setpiece fetch` downloads that data at runtime from StatsBomb's public
repository and caches it under `data/`, which is gitignored. Three Euro 2024
freeze frames are committed in `setpiece/assets/demo/` so that `setpiece demo`
works offline; they are StatsBomb open data too, and the same agreement covers
them and the trained checkpoints in `setpiece/assets/`.

Data provided by StatsBomb.

## Screenshots

Both images below are produced by the app's own PNG export (`render_png`), so
they are exactly what the export button writes — same drawing, same header. They
were exported before the current checkpoints were fitted, so the probabilities
printed in them are that model's, not today's; the drawing is what is being
shown.

A scored corner. Each attacker carries its probability of winning first
contact; the header shows P(shot within 5 events) and P(defence wins first
contact), with the static-frame caveat:

![A scored corner](assets/pitch-editor.png)

The same corner after `suggest`, with the three proposed defender moves drawn
as arrows:

![Suggested defender moves](assets/suggest-overlay.png)

The CLI output is shown inline throughout this README and in `docs/usage.md`
rather than as an image, so it stays copy-pasteable and greppable.

## Documentation

- `docs/usage.md` — an end-to-end walkthrough with real commands and output.
- `docs/architecture.md` — the module contract.
- `docs/metrics.md` — the measured numbers, regenerated by `setpiece train`.
- `docs/search-objective.md` — what a `suggest` delta is worth, measured.
- `TODO.md` — known gaps, shortcuts and things that are not finished.
