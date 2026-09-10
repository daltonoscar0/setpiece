# usage

An end-to-end walkthrough. Every command below was run from the repository root
on a laptop CPU, and every block of output is the real output of the command
above it, abridged only where marked `...`.

Contents:

1. [Setup](#1-setup)
2. [The offline demo](#2-the-offline-demo)
3. [Scoring a StatsBomb 360 corner](#3-scoring-a-statsbomb-360-corner)
4. [Scoring a corner of your own](#4-scoring-a-corner-of-your-own)
5. [Machine-readable output](#5-machine-readable-output)
6. [Suggesting defensive adjustments](#6-suggesting-defensive-adjustments)
7. [Widening the search](#7-widening-the-search)
8. [The pitch editor](#8-the-pitch-editor)
9. [Rebuilding the dataset](#9-rebuilding-the-dataset)
10. [Retraining](#10-retraining)
11. [Calling it from Python](#11-calling-it-from-python)
12. [Running the tests](#12-running-the-tests)
13. [Exit codes and errors](#13-exit-codes-and-errors)

Throughout: coordinates are **metres on a 105x68 pitch, attacking towards +x**,
the goal being attacked is at `(105, 34)`, and every model is `assume_static` -
freeze frames carry no velocities and none are invented.

---

## 1. Setup

```
git clone https://github.com/daltonoscar0/setpiece
cd setpiece
uv sync
```

That is enough to score corners: the trained checkpoints are committed in
`setpiece/assets/`. The pitch editor needs the extra:

```
uv sync --extra app
```

Check the install:

```
$ uv run setpiece --help
usage: setpiece [-h] [--version] [-v] COMMAND ...

Corner-kick receiver and shot scoring, with defender adjustment search.

positional arguments:
  COMMAND
    score        score one corner: receiver, defence, shot
    suggest      rank small defensive adjustments
    demo         score the three bundled Euro 2024 corners, offline
    app          launch the Streamlit pitch editor
    fetch        build the corner dataset from StatsBomb open data
    train        train the receiver and shot models (wraps setpiece.train)

options:
  -h, --help     show this help message and exit
  --version      show program's version number and exit
  -v, --verbose  log progress to stderr
```

## 2. The offline demo

`demo` scores and searches the three Euro 2024 corners bundled in
`setpiece/assets/demo/`. It never touches the network.

```
$ uv run setpiece demo
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

Reading the first block: the model puts 21.1% on the attacker at
`(100.5, 14.5)` winning the first contact and 46.7% on the defence winning it.
It was in fact the attacker at index 8, which the model ranked second. The
`actual:` line only appears when the corner carries labels, which the bundled
demo corners do.

A warm `demo` takes about five seconds wall clock, nearly all of it importing
torch. The first run in a freshly created venv pays a one-off cost to page torch
in from disk and has been measured between 46 s and 60 s on a clean clone.

## 3. Scoring a StatsBomb 360 corner

`score` accepts a StatsBomb `three-sixty` entry, a bare `freeze_frame` list,
setpiece's own JSON, or a whole `.jsonl` dataset. The bundled demo files are
StatsBomb 360 exports, so:

```
$ uv run setpiece score setpiece/assets/demo/euro2024_3930163_82min.json
corner 36ececb0  (UEFA Euro, 2024, match 3930163, 82')
18 players: 7 attacking, 11 defending

 #  role         x     y  P(first contact)
--  -------  -----  ----  ----------------  ----------------
11  attack   100.3  30.5             0.105  ##..............
 8  attack    98.2  32.2             0.095  ##..............
13  attack   101.4  33.7             0.054  #...............
 6  attack    95.9  36.4             0.044  #...............
 4  attack    93.8  35.7             0.032  #...............
 1  attack    84.2  25.8             0.001  ................
 -  defence                          0.669  ###########.....

P(shot within 5 events)  0.244  ####............
actual: first contact defender (defender), shot within 5: True
positions only; freeze frames carry no velocities (--assume-static)
```

Eighteen players are visible, not twenty-two: a 360 frame only contains the
players inside the broadcast camera's field of view. The taker is not listed in
the table because the taker is never a candidate to receive their own corner.
The listed attackers and the `defence` row form one distribution and sum to 1.

Pointing `score` at a `.jsonl` dataset scores every corner in it, which is why
`--format csv` exists:

```
$ uv run setpiece score data/interim/smoke.jsonl --format csv | head -3
corner_id,idx,team,x,y,is_taker,is_keeper,p_receive,p_shot
3422371f-2ae4-4959-9fde-5e5ddb8aeaa5,0,attack,80.91,27.57,0,0,0.007438,0.258953
3422371f-2ae4-4959-9fde-5e5ddb8aeaa5,1,defend,91.66,42.32,0,0,0.000000,0.258953
```

## 4. Scoring a corner of your own

The CSV format is `team,x,y,is_taker` with an optional `is_keeper`. `team` is
`attack` or `defend` (`att`/`def`, `a`/`d` and `1`/`0` also work). Save this as
`corner.csv`:

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

The first row is the taker, standing at the corner flag at `(105, 68)`. The
eighth row is the goalkeeper on his line.

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

No `actual:` line this time, because a hand-written corner carries no labels.
The heading is `corner corner` because the corner id defaults to the file stem.

Two things to get right:

- **Flag the taker.** If no row has `is_taker=1`, the delivery origin falls back
  to the corner flag nearest the shape, and the unflagged taker is scored as an
  ordinary candidate receiver. On this corner that mistake hands the taker
  standing on the flag a probability of 0.249, the highest in the table, which
  is nonsense and visible at a glance.
- **Get the direction right.** The attacking team always attacks towards +x. A
  shape mirrored into the other half will be scored as if everyone were 60 m
  from goal.

## 5. Machine-readable output

`--format json` on `score` gives the full structure: the input coordinates, the
per-player probabilities, the defence class, P(shot), and an `assume_static`
flag.

```
$ uv run setpiece score corner.csv --format json
{
  "corner_id": "corner",
  "competition": null,
  "season": null,
  "match_id": null,
  "minute": null,
  "n_players": 17,
  "n_attackers": 7,
  "n_defenders": 10,
  "players": [
    {
      "idx": 0,
      "team": "attack",
      "x": 104.8,
      "y": 67.6,
      "is_taker": true,
      "is_keeper": false,
      "p_receive": 0.0
    },
    ...
  ],
  "p_defence_first_contact": 0.5449554920196533,
  "p_shot": 0.28122878074645996,
  "assume_static": true
}
```

`--format csv` is one row per player plus a `defence_first_contact` row per
corner, with `p_shot` repeated on every row so the file joins cleanly:

```
$ uv run setpiece score corner.csv --format csv
corner_id,idx,team,x,y,is_taker,is_keeper,p_receive,p_shot
corner,0,attack,104.80,67.60,1,0,0.000000,0.281229
corner,1,attack,99.50,36.50,0,0,0.118569,0.281229
corner,2,attack,98.20,33.00,0,0,0.081793,0.281229
corner,3,attack,97.00,38.50,0,0,0.132168,0.281229
corner,4,attack,94.50,34.50,0,0,0.080900,0.281229
corner,5,attack,92.00,29.00,0,0,0.039592,0.281229
corner,6,attack,88.00,34.00,0,0,0.002024,0.281229
corner,7,defend,103.60,34.00,0,1,0.000000,0.281229
corner,8,defend,100.00,30.90,0,0,0.000000,0.281229
corner,9,defend,99.80,37.20,0,0,0.000000,0.281229
corner,10,defend,98.50,35.00,0,0,0.000000,0.281229
corner,11,defend,97.60,31.50,0,0,0.000000,0.281229
corner,12,defend,96.00,38.00,0,0,0.000000,0.281229
corner,13,defend,94.00,32.00,0,0,0.000000,0.281229
corner,14,defend,93.50,36.50,0,0,0.000000,0.281229
corner,15,defend,90.00,34.00,0,0,0.000000,0.281229
corner,16,defend,86.00,40.00,0,0,0.000000,0.281229
corner,-1,defence_first_contact,,,0,0,0.544955,0.281229
```

Defenders get `p_receive` 0.000000: the receiver head is a distribution over
attacking candidates and the defence class only.

## 6. Suggesting defensive adjustments

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

`delta` is `new_prob - base_prob`, so negative is an improvement, and
suggestions are ranked best first. The player numbers are the row indices from
the scored table, which are the CSV rows in order starting at 0, so `#13` is
`defend,94.0,32.0`, the fourteenth data row.

**What that `-0.047` is worth.** `docs/search-objective.md` measures exactly
this on 400 real corners, and the answer is narrow: a searched move beats a
random legal move under every second model tested, so the *direction* is real,
but only about a fifth of the claimed improvement is visible to a logistic
regression on the same moved frame (about a third to a fresh seed of the same
architecture), the cross-family correlation between the two deltas is +0.10 with
a bootstrap interval containing zero, and the shipped head calls 60% of
*arbitrary* legal defender moves an improvement at a median of -0.0038. Read the
direction as a weak signal and the number as one model's opinion about a shape
no match ever produced.

The search enforces its constraints internally, only defenders move, at most
`--movers` of them, each displacement at most `--max-move` metres, no two
players end up closer than 0.5 m to each other (attackers and unmoved defenders
included), and nobody leaves the pitch. It is deterministic: the same corner,
flags and `--seed` produce byte-identical output, including the ordering.

Flip the objective with `--attack` to see where the shape is already brittle:

```
$ uv run setpiece suggest corner.csv --attack --movers 1 --top-k 3
corner corner
moving up to 1 defenders to raise the shot probability
search width: 6 defenders for the first move, 2 for later moves, nearest the goal first; keeper excluded; budget 96 evaluations (--max-candidates / --deep-candidates / --move-keeper / --max-evals)
base P(shot) 0.281

#  P(shot)   delta  moves
-  -------  ------  -----------------------
1    0.304  +0.022  #13 -> 96.4,33.9 (3.0m)
2    0.297  +0.016  #12 -> 95.6,35.0 (3.0m)
3    0.297  +0.016  #11 -> 98.0,34.5 (3.0m)
```

`--format json` gives every move as `dx`, `dy`, `length`, `from` and `to`, and
records the search width under `params`:

```
$ uv run setpiece suggest corner.csv --movers 1 --top-k 2 --format json
{
  "corner_id": "corner",
  ...
  "objective": "minimise_shot",
  "params": {
    "movers": 1,
    "max_move": 3.0,
    "seed": 0,
    "beam": 8,
    "top_k": 2,
    "max_candidates": 6,
    "deep_candidates": 2,
    "move_keeper": false,
    "max_evals": 96
  },
  "base_p_shot": 0.28122878074645996,
  "suggestions": [
    {
      "rank": 1,
      "moves": [
        {
          "player_idx": 9,
          "dx": -1.847,
          "dy": 2.364,
          "length": 3.0,
          "from": [
            99.8,
            37.2
          ],
          "to": [
            97.95,
            39.56
          ]
        }
      ],
      "base_prob": 0.28122878074645996,
      "new_prob": 0.25592541694641113,
      "delta": -0.025303363800048828
    },
    ...
  ],
  "assume_static": true
}
```

Every proposed move is a **static repositioning at the moment the corner is
struck**, not a run. The model has no velocities and therefore no way of knowing
whether a defender could get there.

## 7. Widening the search

By default only the six defenders nearest the attacked goal are candidates for
the first move, and only the nearest two for moves after that. That bounds the
branching factor so a search fits the pitch editor's round trip. Four flags
expose it:

| flag | default | what it does |
|---|---|---|
| `--max-candidates N\|all` | 6 | defenders, nearest the goal first, tried for the first move |
| `--deep-candidates N\|all` | 2 | the same, for the second and later moves |
| `--move-keeper` | off | lets the goalkeeper into the movable pool |
| `--max-evals N` | sized from the layout | ceiling on shot-model evaluations |

The pretty output prints the width actually used on every run, the
`search width:` line above, and the JSON records it under `params`, so a
suggestion always says how hard it looked.

```
$ uv run setpiece suggest corner.csv --movers 1 --top-k 2 --max-candidates all --move-keeper
corner corner
moving up to 1 defenders to lower the shot probability
search width: all defenders for the first move, 2 for later moves, nearest the goal first; keeper movable; budget 160 evaluations (--max-candidates / --deep-candidates / --move-keeper / --max-evals)
base P(shot) 0.281

#  P(shot)   delta  moves
-  -------  ------  -----------------------
1    0.252  -0.029  #14 -> 91.7,38.9 (3.0m)
2    0.256  -0.025  #9 -> 98.0,39.6 (3.0m)

static repositioning at the moment of the delivery; no player is assumed to cover ground (--assume-static)
```

Note the budget: `setpiece suggest` sizes `--max-evals` from the widths you
asked for, so widening the flags really does widen the search rather than
silently truncating it. Set `--max-evals` yourself to buy latency back, the
search degrades gracefully, expanding states in beam-rank order and stopping
when the budget runs out. The library is stricter about this than the CLI:
calling `suggest(..., max_candidates=None)` while leaving `max_evals` at its
default of 288 raises a warning naming the number of evaluations the widened
layout actually needs.

What the default costs, measured over 100 corners sampled from
`data/interim/corners.jsonl` (seed 20260827, single-threaded): the shipped 6/2
width finds a median best delta of -0.061 against the -0.080 an every-defender
search finds, so the default gives up a median 0.015 of shot probability, and
the wide search charges 3.5x the wall time for it. Given how little of any delta
transfers to a second model (section 6), that is not an obviously good trade,
which is why 6/2 is the default.

## 8. The pitch editor

```
$ uv sync --extra app
$ uv run setpiece app --headless --port 8599
launching: /Users/.../.venv/bin/python3 -m streamlit run /Users/.../setpiece/app.py --server.port 8599 --server.headless true
...
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8599
```

That `launching:` line comes from `setpiece`, the rest from Streamlit; the two
streams are buffered separately, so the order they appear in a redirected log
is not the order above. Without `--port` it serves on 8501, and without
`--headless` it opens a browser window for you.

The app is launched as a fresh `streamlit run` subprocess, so nothing the CLI
sets in process reaches it. `app.py` therefore pins `torch.set_num_threads(1)`
in its own `main()`, which keeps the editor off the machine's other cores. An
`OMP_NUM_THREADS` or `MKL_NUM_THREADS` you set explicitly wins over the pin.

In the sidebar: pick a **bundled demo** corner or **upload JSON / CSV**; set the
zoom with *pixels per metre* and *metres of pitch shown*; point *checkpoint file
or directory* at another model, or leave it blank for the bundled one; and set
the *defenders moved*, *max move (m)* and *seed* used by the search.

On the page: drag any player on the canvas and the corner is re-scored on drop,
with the round-trip time reported underneath. **suggest defender moves** runs
the same search as the CLI and draws the proposed moves as arrows; **apply these
moves** commits them so you can carry on from the new shape. **download PNG**
exports the annotated pitch, and the model cards for both heads are in the
expander at the bottom.

## 9. Rebuilding the dataset

`data/` is gitignored, so a fresh clone has no dataset. `fetch` builds one:

```
$ uv run setpiece fetch --competition "UEFA Euro" --max-matches 1 --out data/interim/sample.jsonl
building dataset: 1 matches to fetch
1/1 matches, 3 corners
3 corners from 1 matches -> data/interim/sample.jsonl
```

With no flags it walks every open competition-season flagged for 360 data and
writes `data/interim/corners.jsonl`. What it does per match: download the events
and the freeze frames, keep the corners that carry a frame and have at least 8
visible players, attach the labels, and throw the raw files away (`--keep-raw`
keeps them; they run about 7 MB a match). Requests are rate limited to one per
second per host with exponential backoff on failure.

Progress is checkpointed to `<out>.state.json` after every tenth match, so
interrupting the build and running it again resumes rather than re-downloading.
The state file also records what went wrong, how many corners it saw against how
many it kept, and how many times each failing match has been tried:

```
$ uv run python -c "import json; s=json.load(open('data/interim/corners.state.json')); print(len(s['done_matches']), s['n_corners'], s['no_360'], s['failures'][0]['match_id'], s['attempts'])"
476 2464 {'African Cup of Nations 2023': 51} 3845506 {'3845506': 1}
```

That is the full build behind the shipped models: 476 matches walked, 2464
corners extracted, 51 matches in a competition that is flagged for 360 data but
ships none, and one match whose upstream file is malformed JSON. That match has
one recorded failure; `build_dataset` gives up on a match after `max_attempts`
(default 3) and stops re-downloading it, updating its `failures` row in place
rather than appending a duplicate on every pass. `max_attempts` is a keyword
argument to `setpiece.fetch.build_dataset`, **not** a CLI flag, from
`setpiece fetch` the cap is fixed at 3. See `TODO.md`.

The `coverage` entry in the same file is where the 360 coverage numbers in
`docs/metrics.md` come from, `corners_seen`, `corners_with_frame` and
`dropped_too_few_players` per season, which is how "55.2% of open-data corners
have a frame" became a measurement rather than an estimate.

The full build is several hundred matches at one request per second, so expect
it to take about an hour, most of it waiting politely.

## 10. Retraining

```
uv run setpiece train
```

That runs 5-fold cross-validation purged by `match_id`, every corner of a match
lands in one fold, evaluates the GNN heads against both baselines on identical
held-out rows, **repeats the whole protocol over five seeds and reports
`mean ±sd`**, measures calibration and the receiver-label threshold sweep, refits
on all the data, prints the metrics table, writes it to `docs/metrics.md`, and
saves `receiver.pt`, `shot.pt` and `report.json` under `runs/latest`.

Five repeats of the full protocol plus the calibration and sweep sections is a
long run. A smoke run on the 18-corner subset, with the repeats and the extra
sections turned off, takes about eleven seconds end to end:

```
$ uv run setpiece train --dataset data/interim/smoke.jsonl --out-dir runs/smoke \
    --no-metrics-file --folds 2 --epochs 2 --seeds 1 --no-calibrate --no-threshold-sweep
...
| model | top-1 (attacker-first) | top-3 (attacker-first) | overall acc. incl. defence | defence-wins AUC |
|---|---|---|---|---|
| GNN | 0.000 | 0.286 | 0.588 | 0.486 |
| logistic regression | 0.000 | 0.429 | 0.176 | 0.571 |
| heuristic | 0.143 | 0.429 | 0.588 | 0.414 |
...
Parameters: receiver 118,082, shot 97,601.
```

Those numbers are meaningless, 18 corners, seven of them attacker-first, and
are shown only to demonstrate the command. `--no-metrics-file` is what keeps a
run like that from overwriting the real `docs/metrics.md`; use it for anything
you do not intend to publish.

Other flags: `--folds`, `--seeds`, `--epochs`, `--hidden`, `--layers`,
`--dropout`, `--lr`, `--weight-decay`, `--batch-size`, `--seed`,
`--baseline-iters`, `--limit N`, `--quiet`, `--no-calibrate`,
`--no-threshold-sweep`, `--cache-dir` (reuse per-seed out-of-fold predictions so
repeats can be precomputed in parallel and the report regenerated without
retraining), `--label-baseline` and `--ablation`.
`uv run setpiece train --help` lists them with defaults.

Plain `train` does **not** overwrite the bundled checkpoints in
`setpiece/assets/`. To try a fresh run without committing to it:

```
uv run setpiece score corner.csv --model runs/latest
```

`--model` takes a directory holding `receiver.pt` and `shot.pt`, or a single
checkpoint file. When you do want the refitted checkpoints to become the
defaults, pass `--install`:

```
uv run setpiece train --install
```

which copies `receiver.pt` and `shot.pt` from `--out-dir` over the bundled ones.
It has a sharp edge: it will happily install a two-fold smoke run over good
checkpoints, so pair it with a real `--dataset`.

## 11. Calling it from Python

```python
from setpiece._io import load_corner
from setpiece.models import load_bundled, receiver_probabilities, shot_probability

frame = load_corner("corner.csv")
receiver, card = load_bundled("receiver")
shot, _ = load_bundled("shot")

probs, p_defence = receiver_probabilities(receiver, frame)
print("trained on", card["data"]["n_corners"], "corners")
print("best candidate:", int(probs.argmax()), round(float(probs.max()), 3))
print("P(defence wins first contact):", round(p_defence, 3))
print("P(shot within 5):", round(shot_probability(shot, frame), 3))
```

```
trained on 2464 corners
best candidate: 3 0.132
P(defence wins first contact): 0.545
P(shot within 5): 0.281
```

`probs` is a numpy array with one entry per player, zero for everyone who is not
a candidate (defenders and the taker). `card` is the model card: the
competitions and seasons trained on, the 360 coverage ledger, the split scheme,
the measured metrics with their five-seed spreads under `metrics_spread`, the
calibrator (`{"kind": "identity"}` on both heads, nothing improved held-out
ECE, so neither ships one), the feature list, the parameter count, the thread
count it was trained at, the date, and `assume_static: True`. The features list
carries an `excluded` entry naming `meta.end_xy` and `meta.pass_outcome`, which
describe what actually happened to the delivery and are never used as inputs;
the same goes for the diagnostic timings `meta.first_touch_dt`, `meta.shot_dt`
and `meta.shot_n_events`, which exist so the label windows can be audited.

Scoring both heads costs about 6 ms per corner single-threaded (measured over
the 18 corners in `data/interim/smoke.jsonl` on a busy ten-core machine), so a
few thousand corners is a matter of seconds.
`setpiece.suggest.suggest(frame, shot_model, ...)` is the search itself,
returning `Suggestion` objects with `moves`, `base_prob`, `new_prob` and
`delta`. It accepts any `frame -> float` callable as the objective, so searching
your own model, the logistic baseline or an ensemble needs no change here.
`setpiece.suggest.required_evals()` computes the evaluation budget a given
layout wants, which is how `DEFAULT_MAX_EVALS = 288` was derived.

## 12. Running the tests

```
uv run python -m pytest
```

430 passed, 1 skipped. Six tests carry the `slow` marker, the training
reproducibility check, the smoke-dataset training run, the nested-calibration
check, the inference benchmark, and the two `app.py` thread-pinning tests that
each spawn a second interpreter, so an edit loop can skip them:

```
uv run python -m pytest -m "not slow"
```

which runs 424 of the 431 collected tests. Do not expect it to be dramatically
faster: several seconds of every run are a fixed floor spent importing torch and
torch_geometric during collection, which no marker selection can avoid, and the
heaviest single file (`tests/test_suggest_stress.py`) is unmarked. `TODO.md`
carries the per-file breakdown and says what is still not marked.

Lint with:

```
uv run ruff check setpiece tests
```

## 13. Exit codes and errors

| code | meaning |
|---|---|
| 0 | success |
| 1 | runtime failure: unreadable corner file, fetch error |
| 2 | argparse usage error, or no subcommand given |
| 3 | a required checkpoint is missing or unloadable |
| 4 | an optional dependency is missing (`streamlit` for `app`) |

```
$ uv run setpiece score missing.csv; echo "EXIT=$?"
error: no such corner file: missing.csv
EXIT=1

$ uv run setpiece score corner.csv --model nope.pt; echo "EXIT=$?"
error: --model path does not exist: nope.pt
EXIT=3
```

If the bundled checkpoints are absent, every scoring command exits 3 with the
commands to run rather than scoring the corner with an untrained network:

```
no trained 'receiver' model available (looked for .../setpiece/assets/receiver.pt).
Train one and install it as the bundled checkpoint:
    uv run setpiece train --install
(plain 'setpiece train' only writes to --out-dir, default runs/latest)
or point at an existing checkpoint:
    uv run setpiece <command> --model path/to/receiver.pt
(--model also accepts a directory holding receiver.pt and shot.pt).
```
