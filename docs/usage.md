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
7. [The pitch editor](#7-the-pitch-editor)
8. [Rebuilding the dataset](#8-rebuilding-the-dataset)
9. [Retraining](#9-retraining)
10. [Calling it from Python](#10-calling-it-from-python)
11. [Exit codes and errors](#11-exit-codes-and-errors)

Throughout: coordinates are **metres on a 105x68 pitch, attacking towards +x**,
the goal being attacked is at `(105, 34)`, and every model is `assume_static` —
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
14  attack   100.5  14.5             0.179  ###.............
 8  attack    97.4  36.5             0.100  ##..............
 4  attack    95.9  41.6             0.077  #...............
13  attack    99.3  31.7             0.046  #...............
 7  attack    97.4  32.3             0.035  #...............
 5  attack    96.7  32.9             0.033  #...............
 2  attack    85.2  37.5             0.019  ................
 0  attack    81.9  13.1             0.001  ................
 -  defence                          0.510  ########........

P(shot within 5 events)  0.208  ###.............
actual: first contact attacker (attacker #8), shot within 5: True
positions only; freeze frames carry no velocities (--assume-static)

corner e5c0796a  (UEFA Euro, 2024, match 3930159, 71')
moving up to 2 defenders to lower the shot probability
base P(shot) 0.208

#  P(shot)   delta  moves
-  -------  ------  ------------------------
1    0.194  -0.014  #15 -> 103.3,30.0 (3.0m)
2    0.203  -0.006  #15 -> 99.6,34.7 (3.0m)
3    0.204  -0.004  #15 -> 98.5,32.7 (3.0m)

static repositioning at the moment of the delivery; no player is assumed to cover ground (--assume-static)
----------------------------------------------------------------------
...
scored and searched 3 corners in 1.3s (plus 7.1s to import torch and load the checkpoints), no network access
```

Reading the first block: the model puts 17.9% on the attacker at `(100.5, 14.5)`
winning the first contact and 51.0% on the defence winning it. It was in fact
the attacker at index 8, which the model ranked second. The `actual:` line only
appears when the corner carries labels, which the bundled demo corners do.

`demo` takes about eleven seconds wall clock, most of it importing torch.

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
11  attack   100.3  30.5             0.083  #...............
 8  attack    98.2  32.2             0.069  #...............
 6  attack    95.9  36.4             0.068  #...............
13  attack   101.4  33.7             0.049  #...............
 4  attack    93.8  35.7             0.043  #...............
 1  attack    84.2  25.8             0.001  ................
 -  defence                          0.687  ###########.....

P(shot within 5 events)  0.287  #####...........
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
3422371f-2ae4-4959-9fde-5e5ddb8aeaa5,0,attack,80.91,27.57,0,0,0.009912,0.295533
3422371f-2ae4-4959-9fde-5e5ddb8aeaa5,1,defend,91.66,42.32,0,0,0.000000,0.295533
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
3  attack   97.0  38.5             0.124  ##..............
1  attack   99.5  36.5             0.096  ##..............
2  attack   98.2  33.0             0.072  #...............
4  attack   94.5  34.5             0.067  #...............
5  attack   92.0  29.0             0.027  ................
6  attack   88.0  34.0             0.003  ................
-  defence                         0.612  ##########......

P(shot within 5 events)  0.332  #####...........
positions only; freeze frames carry no velocities (--assume-static)
```

No `actual:` line this time, because a hand-written corner carries no labels.
The heading is `corner corner` because the corner id defaults to the file stem.

Two things to get right:

- **Flag the taker.** If no row has `is_taker=1`, the delivery origin falls back
  to the corner flag nearest the shape, and the unflagged taker is scored as an
  ordinary candidate receiver. On this corner that mistake hands the taker
  standing on the flag a probability of 0.383, which is nonsense and visible at
  a glance.
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
  "p_defence_first_contact": 0.6118548967471266,
  "p_shot": 0.332246333360672,
  "assume_static": true
}
```

`--format csv` is one row per player plus a `defence_first_contact` row per
corner, with `p_shot` repeated on every row so the file joins cleanly:

```
$ uv run setpiece score corner.csv --format csv
corner_id,idx,team,x,y,is_taker,is_keeper,p_receive,p_shot
corner,0,attack,104.80,67.60,1,0,0.000000,0.332246
corner,1,attack,99.50,36.50,0,0,0.095516,0.332246
corner,2,attack,98.20,33.00,0,0,0.071644,0.332246
corner,3,attack,97.00,38.50,0,0,0.124232,0.332246
corner,4,attack,94.50,34.50,0,0,0.067135,0.332246
corner,5,attack,92.00,29.00,0,0,0.026684,0.332246
corner,6,attack,88.00,34.00,0,0,0.002933,0.332246
corner,7,defend,103.60,34.00,0,1,0.000000,0.332246
corner,8,defend,100.00,30.90,0,0,0.000000,0.332246
corner,9,defend,99.80,37.20,0,0,0.000000,0.332246
corner,10,defend,98.50,35.00,0,0,0.000000,0.332246
corner,11,defend,97.60,31.50,0,0,0.000000,0.332246
corner,12,defend,96.00,38.00,0,0,0.000000,0.332246
corner,13,defend,94.00,32.00,0,0,0.000000,0.332246
corner,14,defend,93.50,36.50,0,0,0.000000,0.332246
corner,15,defend,90.00,34.00,0,0,0.000000,0.332246
corner,16,defend,86.00,40.00,0,0,0.000000,0.332246
corner,-1,defence_first_contact,,,0,0,0.611855,0.332246
```

Defenders get `p_receive` 0.000000: the receiver head is a distribution over
attacking candidates and the defence class only.

## 6. Suggesting defensive adjustments

```
$ uv run setpiece suggest corner.csv --movers 2 --top-k 3
corner corner
moving up to 2 defenders to lower the shot probability
base P(shot) 0.332

#  P(shot)   delta  moves
-  -------  ------  ------------------------------------------------
1    0.268  -0.064  #13 -> 93.6,29.0 (3.0m); #8 -> 101.8,28.5 (3.0m)
2    0.270  -0.062  #13 -> 93.6,29.0 (3.0m); #8 -> 99.6,27.9 (3.0m)
3    0.270  -0.062  #13 -> 91.6,30.1 (3.0m); #8 -> 101.8,28.5 (3.0m)
```

`delta` is `new_prob - base_prob`, so negative is an improvement, and
suggestions are ranked best first. The player numbers are the row indices from
the scored table, which are the CSV rows in order starting at 0, so `#13` is
`defend,94.0,32.0` — the fourteenth data row.

The search enforces its constraints internally — only defenders move, at most
`--movers` of them, each displacement at most `--max-move` metres, no two
players end up closer than 0.5 m to each other (attackers and unmoved defenders
included), and nobody leaves the pitch. It is deterministic: the same corner,
flags and `--seed` produce byte-identical output, including the ordering.

Flip the objective with `--attack` to see where the shape is already brittle:

```
$ uv run setpiece suggest corner.csv --attack --movers 1 --top-k 3
corner corner
moving up to 1 defenders to raise the shot probability
base P(shot) 0.332

#  P(shot)   delta  moves
-  -------  ------  ------------------------
1    0.334  +0.001  #10 -> 101.5,34.6 (3.0m)
2    0.334  +0.001  #13 -> 94.7,33.3 (1.5m)
3    0.333  +0.001  #13 -> 93.6,33.5 (1.5m)
```

Those deltas are `+0.001`. On this shape no single defender can make things
meaningfully worse, which is worth reporting as such rather than dressing up.

`--format json` gives every move as `dx`, `dy`, `length`, `from` and `to`:

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
    "top_k": 2
  },
  "base_p_shot": 0.332246333360672,
  "suggestions": [
    {
      "rank": 1,
      "moves": [
        {
          "player_idx": 13,
          "dx": -0.366,
          "dy": -2.978,
          "length": 3.0,
          "from": [
            94.0,
            32.0
          ],
          "to": [
            93.63,
            29.02
          ]
        }
      ],
      "base_prob": 0.332246333360672,
      "new_prob": 0.2983100116252899,
      "delta": -0.03393632173538208
    },
    ...
  ],
  "assume_static": true
}
```

Every proposed move is a **static repositioning at the moment the corner is
struck**, not a run. The model has no velocities and therefore no way of knowing
whether a defender could get there.

## 7. The pitch editor

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

## 8. Rebuilding the dataset

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
The state file also records what went wrong:

```
$ uv run python -c "import json; s=json.load(open('data/interim/corners.state.json')); print(len(s['done_matches']), s['n_corners'], s['no_360'], s['failures'][0]['match_id'])"
476 2464 {'African Cup of Nations 2023': 51} 3845506
```

That is the full build behind the shipped models: 476 matches walked, 2464
corners extracted, 51 matches in a competition that is flagged for 360 data but
ships none, and one match whose upstream file is malformed JSON. See `TODO.md`.

The full build is several hundred matches at one request per second, so expect
it to take about an hour, most of it waiting politely.

## 9. Retraining

```
uv run setpiece train
```

That runs 5-fold cross-validation purged by `match_id` — every corner of a match
lands in one fold — evaluates the GNN heads against both baselines on identical
held-out rows, refits on all the data, prints the metrics table, writes it to
`docs/metrics.md`, and saves `receiver.pt`, `shot.pt` and `report.json` under
`runs/latest`.

A smoke run on the 18-corner subset takes about eleven seconds end to end:

```
$ uv run setpiece train --dataset data/interim/smoke.jsonl --out-dir runs/smoke --no-metrics-file --folds 2 --epochs 2
...
| model | top-1 (attacker-first) | top-3 (attacker-first) | overall acc. incl. defence | defence-wins AUC |
|---|---|---|---|---|
| GNN | 0.143 | 0.286 | 0.588 | 0.557 |
| logistic regression | 0.143 | 0.429 | 0.235 | 0.657 |
| heuristic | 0.143 | 0.429 | 0.588 | 0.414 |
...
Parameters: receiver 118,082, shot 113,857.
```

Those numbers are meaningless — 18 corners, seven of them attacker-first — and
are shown only to demonstrate the command. `--no-metrics-file` is what keeps a
run like that from overwriting the real `docs/metrics.md`; use it for anything
you do not intend to publish.

Other flags: `--folds`, `--epochs`, `--hidden`, `--layers`, `--dropout`, `--lr`,
`--weight-decay`, `--batch-size`, `--seed`, `--baseline-iters`, `--limit N`,
`--quiet`. `uv run setpiece train --help` lists them with defaults.

Training does **not** overwrite the bundled checkpoints in `setpiece/assets/`.
To try a fresh run without committing to it:

```
uv run setpiece score corner.csv --model runs/latest
```

`--model` takes a directory holding `receiver.pt` and `shot.pt`, or a single
checkpoint file. Copy the two files into `setpiece/assets/` when you want them
to become the defaults.

## 10. Calling it from Python

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
best candidate: 3 0.124
P(defence wins first contact): 0.612
P(shot within 5): 0.332
```

`probs` is a numpy array with one entry per player, zero for everyone who is not
a candidate (defenders and the taker). `card` is the model card: the
competitions and seasons trained on, the split scheme, the measured metrics, the
feature list, the parameter count, the date, and `assume_static: True`. The
features list carries an `excluded` entry naming `meta.end_xy` and
`meta.pass_outcome`, which describe what actually happened to the delivery and
are never used as inputs.

Scoring both heads costs about 10 ms per corner single-threaded (measured over
the 18 corners in `data/interim/smoke.jsonl`), so a few thousand corners is a
matter of seconds. `setpiece.suggest.suggest(frame, shot_model, ...)` is the
search itself, returning `Suggestion` objects with `moves`, `base_prob`,
`new_prob` and `delta`.

## 11. Exit codes and errors

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
command to run rather than scoring the corner with an untrained network:

```
no trained 'receiver' model available (looked for .../setpiece/assets/receiver.pt).
Train one, which writes the bundled checkpoints:
    uv run setpiece train
or point at an existing checkpoint:
    uv run setpiece <command> --model path/to/receiver.pt
(--model also accepts a directory holding receiver.pt and shot.pt).
```

Note that the second line of that message is wrong as things stand: `train`
writes to `--out-dir` (default `runs/latest`), not to `setpiece/assets/`. It is
recorded in `TODO.md`.
