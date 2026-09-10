# TODO

Known gaps, deliberate shortcuts and things that are not finished. Referenced
from `setpiece/_io.py` and from `README.md`.

Struck-through entries were closed; the note after each says how. Everything
else is still open.

A note on the wall-clock figures below. This machine is a ten-core laptop that
routinely carries several other jobs, and its one-minute load average swung
between 6 and 31 while these were taken. Load is recorded beside every timing,
and every one of them is an upper bound rather than a benchmark. Several numbers
that used to be in this file were pure contention artefacts; where that is the
case the entry says so and gives the re-measured figure.

## data layer

- **`sportio` is not installed on this machine, so there is no shared data
  layer to call.** `setpiece/_io.py` carries a minimal, corner-only substitute:
  the StatsBomb catalogue walk, the match download, corner extraction from
  events plus 360 freeze frames, and the JSONL serialisation. It covers exactly
  what this project needs and nothing more. If `sportio` becomes available,
  the fetch half of `_io.py` should be replaced by it; the label recovery
  (`first_touch`, `receiver_idx`, `receiver_match_dist`, `shot_within_5`) is
  corner-specific and would stay here either way.
- The fetch path is `urllib` plus a hand-rolled one-request-per-second throttle
  and exponential backoff, with no `Retry-After` handling and no resumable
  download for the ~7 MB freeze-frame files.
- ~~Five declared dependencies are never imported.~~ Dropped. `pyproject.toml`
  now requires `numpy`, `torch` and `torch-geometric` and nothing else;
  `requests`, `pandas`, `pyarrow`, `pydantic` and `tqdm` are gone, with a comment
  in the file recording that the loader uses `urllib` and `json` from the
  standard library and that `pandas`/`pyarrow` still arrive transitively with
  `streamlit` under the `app` extra.

## dataset gaps

Recorded in `data/interim/corners.state.json`; the current dataset is 2464
corners from 415 matches across 8 competitions, out of 476 matches walked.

- **African Cup of Nations 2023 ships no freeze frames.** The competition is
  flagged `match_available_360` in StatsBomb's `competitions.json`, but no
  `three-sixty/<match_id>.json` file exists for it: 51 processed matches
  yielded zero freeze frames, and a three-match sample returned 404 for every
  three-sixty URL. The 51 matches are therefore absent from the dataset, and so
  are all 460 of their corners, 0% coverage, the only competition at zero. The
  catalogue flag cannot be trusted on its own; `fetch_match_corners` reports
  the real answer through `report["has_360"]`.
- **Match 3845506 could not be parsed.** Its three-sixty file is malformed JSON
  upstream (`Expecting ',' delimiter: line 92794 column 3`). One match, skipped,
  recorded under `failures`. ~~Every subsequent fetch retries it, re-downloads
  about 7 MB, and appends a duplicate `failures` entry.~~ Fixed:
  `build_dataset` keeps a per-match `attempts` counter in the state file, stops
  retrying a match after `max_attempts` (3), logs "giving up on it", returns the
  abandoned ids under `gave_up`, and updates the existing `failures` row in place
  instead of appending a new one. The ledger currently shows
  `attempts == {'3845506': 1}`, so the match has one recorded failure and two
  retries left. The upstream file is still broken; nothing here can fix that.
- ~~The overall 360 coverage of open-data corners has never been measured.~~
  Measured, during the fetch, per season, into the state file. `corners_seen`
  against `corners_with_frame` over the whole walk: **2468 of 4467 corners came
  with a 360 freeze frame, 55.2%**. It is not uniform, 0.0% for African Cup of
  Nations, then 43.9% (Major League Soccer), 52.7% (FIFA World Cup), 53.0%
  (Ligue 1), 56.3% (UEFA Euro), 67.4% (Women's World Cup), 68.3% (UEFA Women's
  Euro), 70.4% (1. Bundesliga), 76.0% (La Liga). The per-competition table is in
  `docs/metrics.md` and in both model cards under `data.coverage`. The old
  "roughly half" in `corners_from_match` is superseded, and README no longer
  says the fraction is unmeasured.
- ~~Corners are dropped when the frame shows fewer than 8 players; that
  threshold is not tuned and the drops are not counted anywhere.~~ Counted. The
  fetch ledger carries `dropped_too_few_players` per season and the whole walk
  discarded **4 corners** out of 2468 with a frame. Tuning a threshold that
  removes 0.16% of the data would not move any number in `docs/metrics.md`, so
  the threshold stands as a sanity check rather than a hyperparameter.

## labels

- Receiver labels are positional matches, not identities. 918 of the 2464
  corners have a clean receiver label; the rest are either defence-first
  contact (a real class, 1320 corners), an attacker who was off-camera (115
  corners, dropped from the receiver head), or a corner the plausibility guards
  declined to label at all (111 `unknown`). Two attackers a metre apart can be
  confused for each other and nothing detects that.
- ~~`RECEIVER_MATCH_MAX_M = 8.0` is a judgement call and its effect on the
  receiver metrics has not been checked.~~ Swept, in `docs/metrics.md`, by
  re-deriving the labels from the stored `receiver_match_dist` and re-scoring the
  same out-of-fold predictions. It is still a judgement call, but no longer an
  unexamined one: across 4 m, 6 m and 8 m **no model's top-1 or top-3 moves by
  more than 0.013 end to end**, and the ranking of the three models is identical
  at every threshold. Match distances are median 3.19 m, p75 5.30 m, p90 8.47 m,
  so 8 m sits out in the tail rather than through the middle of the
  distribution. Thresholds above 8 m are reported as a **bracket, not an
  estimate**, because `_io.py` drops `receiver_idx` once the match exceeds its own
  threshold; loosening it for real needs a re-fetch.
- ~~Both label windows are event counts, not seconds, so they stretch and shrink
  with how densely the match was tagged. A time-based window would be better
  defined.~~ Checked, and the event-count definition stands, but it now carries
  a plausibility guard, because an event-count window walks straight through a
  dead ball and will call a restart twelve events later "the first touch on this
  corner". `FIRST_TOUCH_MAX_S = 10.0` and `SHOT_MAX_S = 15.0` in `_io.py`
  discard, never relabel, a touch or shot outside the window. Measured: the first
  touch normally lands a median 1.61 s after the delivery (p90 2.53 s), but 110
  rows sat beyond the guard out to 238 s and **102 of them were being scored as
  "the defence won first contact"**. The guards moved the labels from
  1422/1042/0 defender/attacker/unknown to **1320/1033/111**, the receiver rows
  from 923 to 918, and the shot base rate from 0.275 to 0.268. On the
  time-versus-event question itself: replacing the 5-event shot window with a
  5-second one flips 2.3% of the shot labels, so the two rules **agree on 97.7%
  of the 2464 corners** and there is no case for changing the definition. That
  comparison is only available in one direction, a shot more than twelve events
  away but within five seconds is not recoverable from the stored dataset, so it
  is a bound on the disagreement, not a full audit.
- The attacking team's own goalkeeper taking the first touch is scored as
  `"defender"`. That is deliberate, an attacking keeper is not receiving his
  own corner, but it does put a handful of odd rows in the defence class.

## modelling

- ~~The logistic-regression baseline beats the GNN on the shot head and on the
  defence-wins AUC; the GNN only wins the receiver ranking.~~ Dead as stated -
  those were single-run gaps quoted to three decimals. On the 5-repeat protocol
  in `docs/metrics.md`, differenced within each seed so the shared split noise
  cancels, **every GNN-versus-logistic gap is inside its own noise floor**:
  receiver top-1 0.000 ±0.012 (0.0x), receiver top-3 0.004 ±0.018 (0.2x),
  defence-wins AUC 0.005 ±0.011 (0.4x), shot AUC 0.002 ±0.012 (0.1x). Nothing
  beats anything.
- **The replacement gap is worse, not better: the graph network has not been
  shown to earn its keep.** A pooled logistic regression on the same flat node
  features matches it on all four numbers with a fifth to a tenth of the spread
  and a fraction of the training cost. The message passing is not measurably
  buying anything, and the shot head is weak in absolute terms besides, 0.584
  ±0.014 AUC on a 0.268 base rate. Things worth trying, none of them tried:
  delivery-side and near/far-post structure as explicit features, a shared
  encoder trained jointly on both heads, class weighting, and simply more data
  than eight competitions. Until one of them separates the two models, the
  honest claim in `README.md` is "ranks receivers well against a geometric
  floor", not "the GNN works".
- **The nearest-to-goal heuristic is barely distinguishable from chance, and the
  metrics table should not be read as "GNN 0.729 against a real baseline".**
  Chance over the candidates is 0.150 top-1 and 0.441 top-3; the heuristic
  scores 0.159 and 0.498. It is a floor, not a competitor. The top-3 floor is
  high because a corner with three or fewer candidate attackers is a free hit,
  so read that column against 0.441 and not against zero. The floor is printed
  next to the column in `docs/metrics.md` and in `README.md` so nobody has to
  work it out.
- **The shot-prior baseline's 0.481 AUC is not evidence of anything.** It is a
  constant within each fold, and the training-fold rate is a strictly decreasing
  function of the held-out fold's own rate, so the score is anti-correlated with
  the label by construction and lands below 0.5 systematically rather than by
  noise. A genuinely constant score scores exactly 0.500 on the same labels.
- **The receiver table is scored on 2238 of 2464 corners and that subset is not
  the population.** 115 dropped rows are attacker-first corners whose receiver
  was off camera and 111 are the guarded `unknown` class. Both filters remove
  attacker-first rows far more often than defence-wins rows, so the scored subset
  runs 0.590 defence-wins against the dataset's 0.536. All three models are
  scored on identical rows, so the columns compare cleanly; the overall-accuracy
  and defence-AUC values are not population estimates and should not be quoted
  as such.
- ~~GNN training is only reproducible at a fixed thread count, so the GNN column
  moves in the third decimal between runs, and nobody has run the obvious fix of
  repeated seeds with a reported spread.~~ Run. `setpiece train --seeds N`
  (default 5) repeats the entire protocol over seeds [0, 1000, 2000, 3000, 4000],
  each re-drawing the fold assignment *and* the network initialisation, and every
  figure in `docs/metrics.md` is now `mean ±sd` across those repeats. Model
  comparisons are differenced *within* each repeat before averaging, which
  cancels the shared split draw. The underlying cause is unchanged and still
  documented, torch's parallel reductions sum in thread-partition order, so
  training at `OMP_NUM_THREADS=4` and at `1` gives different weights
  (state-dict digests `561b3c60…` and `d0e77a07…`) while two runs at the same
  count are bit-identical, but it is now inside the reported spread instead of
  being an unquantified caveat. Everything ships trained at
  `torch.set_num_threads(1)`, recorded in each card as `training.torch_threads`.
- ~~`setpiece suggest` optimises the weaker head, and no experiment has checked
  that the suggested moves survive being re-scored by a different model.~~ Both
  halves closed, in `docs/search-objective.md`, on 400 corners against a
  same-architecture-different-seed model and a logistic regression. There is no
  longer a "weaker head", the shot AUCs differ by 0.1x their spread. On
  transfer, the answer is uncomfortable and is now stated in `README.md` next to
  the `suggest` example: the search beats a random legal move under every foreign
  model tested (p between 1e-06 and 1e-73), but cross-family delta correlation is
  +0.10 with a bootstrap interval containing zero, cross-family rank correlation
  over counterfactual shapes is +0.04, only 21% of the claimed improvement is
  visible to the other family (30% to a fresh seed), the two searches name the
  same three defenders on 7% of corners, and the shipped head calls 60% of
  *arbitrary* legal defender moves an improvement at a median of -0.0038. An
  ensemble objective was measured and rejected: it loses to the single GNN under
  the only judge outside the objective, on 232 of 400 corners.
- **There is still no label for move quality, and there cannot be one from this
  data.** The dataset records what happened, not what would have happened had a
  defender stood two metres to his left. Every statement about the search is
  therefore a statement about model agreement, not about football. Settling it
  needs tracking data or an intervention, neither of which open data provides.
- **Freeze frames carry no velocities.** Every model here is `assume_static`.
  Nothing in the codebase fabricates a velocity, and the suggestion search has
  no way to know whether a defender could actually cover the ground it proposes.
  Any future "can he get there in time?" filter needs tracking data, which open
  data does not provide.
- ~~The keeper is excluded from the search by default and the CLI exposes no
  flag to change it.~~ `setpiece suggest --move-keeper` lets him into the pool;
  the library argument `move_keeper` is unchanged and still defaults to `False`,
  which remains the sensible default for a coaching tool rather than a
  constraint. Covered by
  `tests/test_suggest_stress.py::test_the_keeper_joins_the_pool_only_when_asked_across_the_sweep`.
- Real 360 frames sometimes place two players closer than `min_sep` (0.5 m). The
  search never creates such a pair but cannot repair one between two players it
  does not move. That is stated in `suggest()`'s own docstring and asserted in
  `tests/test_suggest_stress.py::test_a_pre_existing_overlap_is_never_made_worse`,
  in the only form that is actually checkable: every tight pair left after a
  suggestion was tight before it, and involves no player the search moved.
- ~~Only the six defenders nearest the goal are ever candidates for the first
  move and only the nearest two at deeper levels; it is a search restriction the
  CLI does not surface and no flag can lift.~~ Surfaced and liftable.
  `setpiece suggest` grew `--max-candidates N|all`, `--deep-candidates N|all`,
  `--move-keeper` and `--max-evals N`; the pretty output now prints the width
  actually used on every run and the JSON records it under `params`; and the CLI
  sizes `max_evals` from the layout with `required_evals()` so widening the flags
  really widens the search rather than silently truncating it. `suggest()` warns
  when the branching was widened past the defaults and the budget cannot cover
  it. What the default costs is now measured rather than assumed: over 100
  corners sampled from `data/interim/corners.jsonl` (seed 20260827,
  single-threaded, one-minute load 13-31), the shipped 6/2 width reaches a median
  best delta of -0.061 against the -0.080 an every-defender search finds, so the
  **median cost of the default is 0.015 of shot probability** (mean 0.018, worst
  0.064) and the narrow search ties or wins on only 2% of corners, for which the
  wide search charges **3.5x the wall time** (208.0 s against 60.1 s for the same
  100 corners). Given what `docs/search-objective.md` says a delta is worth to a
  second model, buying 0.015 of one model's opinion for 3.5x the latency is not
  obviously the right trade, which is why 6/2 is still the default.
- ~~No calibration work has been done on either head.~~ Done, in
  `setpiece/calibration.py` and the calibration section of `docs/metrics.md`.
  Both binary outputs were checked against observed frequencies over 10 bins,
  under equal-width and equal-count binning: the shot head's raw ECE is 0.018
  (Brier 0.193) and the defence-wins head's 0.051 (Brier 0.228), and the
  calibrated column is measured **nested**, with the calibrator fitted on scores
  from a model that saw neither the held-out fold nor the fitting fold. The
  result is that **no calibrator improved held-out ECE on either head, so neither
  checkpoint ships one** and both cards record `{"kind": "identity"}`. Nothing is
  applied automatically: `score`, `suggest`, `demo` and the app all print the raw
  sigmoid, and `README.md` now says the probabilities are ranking scores rather
  than frequencies. The residual is that the nested measurement trains its extra
  models on three fifths of the data instead of four, which understates rather
  than overstates how much shrinking the shipped head would need.

## tooling and docs

- ~~`setpiece train` does not update the bundled checkpoints.~~ Fixed: `train`
  takes `--install`, which copies the two refitted checkpoints over the bundled
  ones in `setpiece/assets/`. Plain `train` still writes only to `--out-dir`
  (default `runs/latest`), and the `_train_hint` message in `setpiece/cli.py`
  now says so. Note the sharp edge: `--install` will happily overwrite good
  checkpoints with a smoke run, so pair it with a real `--dataset`.
- ~~There is no `LICENSE` file.~~ Added, MIT, with an explicit note that the
  StatsBomb Public Data User Agreement, not MIT, governs the downloaded data,
  the bundled demo corners and the trained checkpoints.
- ~~Screenshots are not in the tree.~~ `assets/pitch-editor.png` and
  `assets/suggest-overlay.png` are committed and referenced from `README.md`.
  Both come from the app's own `render_png` export path, so they are the real
  drawing rather than a mock-up. Two things still missing. First, a screenshot of
  the live Streamlit UI *as a page* (browser chrome, sidebar, the canvas
  mid-drag), that needs a browser and was not capturable in the environment this
  was built in. Second, **both images were exported before the current
  checkpoints were fitted**, so the probabilities printed in them (21.4% base,
  -0.039 delta on the suggest overlay) are the old model's and no longer match
  what `setpiece demo` prints; `README.md` says so, but re-exporting them is the
  real fix. `assets/score-cli.png` was dropped on purpose; CLI output is shown
  inline in `README.md` and `docs/usage.md` so it stays copy-pasteable.
- ~~`setpiece demo` meets the ten second budget in process but takes about 11 s
  wall clock through `uv run`.~~ Re-measured, and the old figure was contention.
  Warm, at a one-minute load average of 9-10, three consecutive runs came in at
  **4.0 s, 5.7 s and 21.2 s**, the third while the load spiked past 14, so
  about 5 s is the honest warm number and anything above that is other jobs. In
  process the demo reports **0.5 s** of scoring and searching plus **4.9 s** to
  import torch and load the checkpoints. The one real cost remains the first run
  in a freshly created venv, which pages torch in from disk and has been measured
  between **46 s and 60 s** on a clean clone. `README.md` states the same pair of
  numbers: about 5 s warm, 46-60 s on a genuinely fresh venv. The contract's
  "under 10 s" holds once the interpreter is warm and does not on a cold one.
- ~~Nothing pins torch's thread count outside the CLI, and the Streamlit app is
  outside it.~~ Fixed in `app.py`'s own `main()`, which is the only place that
  can work: `setpiece app` launches `streamlit run setpiece/app.py` as a fresh
  subprocess, so nothing `cli.py` sets in process reaches it. `app.py` now calls
  `torch.set_num_threads(1)` unless the user set `OMP_NUM_THREADS` or
  `MKL_NUM_THREADS` explicitly, in which case theirs wins, and
  `tests/test_app.py` asserts both branches by spawning a real second
  interpreter. **The old justification for it was wrong, though, and is corrected
  here.** "A default `suggest()` takes 0.6 s pinned to one thread and 2.5-2.9 s
  at torch's default four" was a contention artefact. Re-measured today on the
  22-player frame from the dataset with the bundled `shot.pt`, five repeats each
  at a one-minute load average of 16-21:

  | thread setting | min | median |
  |---|---|---|
  | torch default (4 on this machine) | 0.62 s | 0.69 s |
  | `OMP_NUM_THREADS=1` | 0.43 s | 0.45 s |
  | `OMP_NUM_THREADS=10` | 0.38 s | 0.45 s |

  So the pin is worth roughly 0.2 s at the default, not 2 s, and the 2.5-2.9 s
  regime did not reproduce at any thread count I could set. The pin stays, it is
  free, it provably takes effect, and it keeps the app off the other cores, but
  the under-one-second round trip was never as close to breaking as this file
  claimed.
- ~~The test suite is slower to start than it is to run, and the old per-file
  figures here were measurement error.~~ Corrected, and corrected again. "Roughly
  46 s for `test_features`" was recorded while three sibling checkouts were
  running their own suites on the same ten-core machine; `test_features` is
  **0.05 s** of execution on a quiet machine and measured **0.12 s** today at a
  one-minute load average of 16-22. Nothing in that file was ever slow. Measured
  today with `pytest -q --durations=0 --durations-min=0`, summing call plus setup
  plus teardown per file, at load 16-22:

  | file | execution |
  |---|---|
  | `test_suggest_stress.py` | 13.72 s |
  | `test_app.py` | 5.04 s |
  | `test_cli.py` | 4.70 s |
  | `test_train.py` | 4.17 s |
  | `test_suggest.py` | 1.65 s |
  | `test_models.py` | 0.71 s |
  | `test_features.py` | 0.12 s |
  | `test_calibration.py` | 0.02 s |
  | total | 30.13 s |

  The suite is **430 passed, 1 skipped**. Wall clock for the whole run is about
  22-32 s on a quiet machine and came in at 37-53 s today under load 15-25. Of
  that, roughly **4-6 s is a fixed floor no marker can remove**: `pytest
  --collect-only` costs that much in CPU time (6.2-7.0 s of `user` today, against
  8.8-17.6 s of wall at load 23-30), almost all of it importing torch and
  torch_geometric. Any `-m` selection pays it first.

  ~~Slow tests are unmarked.~~ Six tests now carry `slow`:
  `test_training_is_reproducible`, `test_smoke_dataset_trains`,
  `test_nested_calibration_never_fits_a_calibrator_on_the_rows_it_scores`,
  `test_inference_is_under_100ms_per_corner`, and the two `app.py` thread-pinning
  tests that each spawn a second interpreter. `pytest -m "not slow"` runs 424 of
  the 431 collected tests and is documented in `docs/usage.md`.

  What is still unmarked is where the time actually is. `test_suggest_stress.py`
  is 13.72 s, 46% of all execution, and carries no marks at all. The worst cases
  today were `test_hard_constraints_hold_with_the_candidate_restriction_lifted`
  (2.70 s and 1.56 s for two of its parameter sets),
  `test_identical_across_separate_processes` (1.77 s, a second interpreter
  importing torch) and
  `test_the_keeper_joins_the_pool_only_when_asked_across_the_sweep` (1.61 s).
  Marking those would take several more seconds out of `-m "not slow"`, which is
  the point at which the fast subset becomes an edit loop rather than a rounding
  error.
- The `network` marker declared in `pyproject.toml` has no user and should not
  acquire one. Every test that goes near the fetch path deliberately *blocks* the
  network, `tests/test_cli.py::no_network` replaces `_io.fetch_json`,
  `socket.socket` and `socket.create_connection` with things that raise, so a
  test that genuinely needed the network would be a bug rather than a marker
  case. The marker should be dropped, but `pyproject.toml` is frozen.
- **The scripts under `analysis/` are outside the package and outside the test
  suite.** `precompute_oof.py`, `threshold_sweep.py`, `ablation.py` and
  `label_audit.py` produced numbers that `docs/metrics.md` now reports, but they
  are not importable as `setpiece.*`, nothing under `tests/` exercises them, and
  `ruff` only sees them if you name the directory. They are one-off measurement
  scripts and are honest about it in their docstrings, but a number in a shipped
  document should not depend on an untested file.
- `data/` and `runs/` are gitignored, so a fresh clone can score corners with
  the bundled checkpoints but cannot reproduce the metrics without a full
  `setpiece fetch`, several hundred matches at one request per second, about an
  hour of mostly waiting. There is no smaller published dataset artefact.
- `docs/search-objective.md` still says "`README.md` should carry that caveat
  next to the `suggest` example. It does not today." It does now, under "What a
  `suggest` delta means", so that sentence is stale. The file is otherwise
  current.

## contract items in `docs/architecture.md`

Checked against the code as it stands. Met unless listed here.

- ~~`suggest()` takes three keyword arguments the contract does not list.~~
  Synced. The contract now carries the full signature, including `max_evals`,
  `move_keeper`, `top_k`, `max_candidates` and `deep_candidates`, and explains
  what the two width bounds do.
- ~~The `suggest`, `demo` and `app` subcommands grew flags beyond the documented
  ones.~~ Synced. The contract's CLI block now lists `--attack`, `--beam`,
  `--top-k`, `--format`, `--max-candidates`, `--deep-candidates`, `--model`,
  `--port`, `--headless` and `--install`.
- ~~`build_graph` returns `CornerData`, a `Data` subclass, where the contract
  says `torch_geometric.data.Data`.~~ Stated explicitly in the contract, along
  with why `__inc__("y_receiver")` is pinned to 0 and why
  `global_receiver_index` exists.
- ~~The contract's "typically 12-22 visible players" understates the spread.~~
  Corrected to "8 to 22 players, median 19", measured over the 2464-corner
  dataset.
- **New: the contract lists a `setpiece fetch --max-attempts 3` flag that does
  not exist.** `max_attempts` is a keyword argument to
  `setpiece.fetch.build_dataset` and defaults to 3, but `_cmd_fetch` does not
  pass it and `argparse` never defines the flag, so from the CLI the retry cap is
  fixed at 3 with no way to change it. Either add the flag or drop it from the
  contract; `docs/usage.md` currently documents the honest version, which is that
  it is a library argument.
