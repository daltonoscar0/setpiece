# TODO

Known gaps, deliberate shortcuts and things that are not finished. Referenced
from `setpiece/_io.py` and from `README.md`.

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
- **Five declared dependencies are never imported.** `pyproject.toml` requires
  `requests`, `pandas`, `pyarrow`, `pydantic` and `tqdm`; nothing under
  `setpiece/` imports any of them (`grep -rn "import <pkg>" setpiece` returns
  nothing for all five). They add install weight for no benefit and should be
  dropped, but `pyproject.toml` is frozen, so this is a report rather than a
  fix.

## dataset gaps

Recorded in `data/interim/corners.state.json`; the current dataset is 2464
corners from 415 matches across 8 competitions, out of 476 matches walked.

- **African Cup of Nations 2023 ships no freeze frames.** The competition is
  flagged `match_available_360` in StatsBomb's `competitions.json`, but no
  `three-sixty/<match_id>.json` file exists for it: 51 processed matches
  yielded zero freeze frames, and a three-match sample returned 404 for every
  three-sixty URL. The 51 matches are therefore absent from the dataset. The
  catalogue flag cannot be trusted on its own; `fetch_match_corners` reports
  the real answer through `report["has_360"]`.
- **Match 3845506 could not be parsed.** Its three-sixty file is malformed JSON
  upstream (`Expecting ',' delimiter: line 92794 column 3`). One match, skipped,
  recorded under `failures`. A failed match is never added to `done_matches`, so
  every subsequent `setpiece fetch` retries it, re-downloads about 7 MB and fails
  again, and appends a duplicate entry to `failures`. Failures should be counted
  and given up on after a few attempts, and `failures` deduplicated.
- **The overall 360 coverage of open-data corners has never been measured.**
  `corners_from_match` says "roughly half of open-data corners have no 360
  frame", which is the right order of magnitude but is not a number anyone
  computed here. A spot check of four matches that do carry 360 data found 22 of
  27 corners with a frame; the catalogue-wide figure across all competitions,
  including those with no 360 at all, is lower and unknown. Either measure it
  during the fetch (count corners seen against corners kept, per season, into
  the state file) or stop quoting a fraction.
- Corners are dropped when the frame shows fewer than 8 players. That threshold
  is not tuned and the drops are not counted anywhere.

## labels

- Receiver labels are positional matches, not identities. 923 of the 2464
  corners have a clean receiver label; the rest are either defence-first
  contact (a real class, 1422 corners) or an attacker who was off-camera, whose
  row is dropped from the receiver head. Two attackers a metre apart can be
  confused for each other and nothing detects that.
- `RECEIVER_MATCH_MAX_M = 8.0` is a judgement call, not a fitted threshold. The
  sensitivity of the receiver metrics to it has not been checked.
- `first_touch` is derived from a 12-event lookahead over a fixed set of touch
  types, and `shot_within_5` from a 5-event window. Both windows are event
  counts, not seconds, so they stretch and shrink with how densely the match was
  tagged. A time-based window would be better defined.
- The attacking team's own goalkeeper taking the first touch is scored as
  `"defender"`. That is deliberate — an attacking keeper is not receiving his
  own corner — but it does put a handful of odd rows in the defence class.

## modelling

- **The logistic-regression baseline beats the GNN on the shot head**
  (AUC 0.593 against 0.566), and on the defence-wins AUC (0.639 against 0.623).
  The GNN only wins the receiver ranking. See `docs/metrics.md`. Either the
  shot head needs more than positional features, or the shot label
  (`shot_within_5`) is too weakly determined by the freeze frame for message
  passing to add anything over a pooled linear model. Do not ship a claim that
  the GNN is better at predicting shots. Things worth trying, none of them
  tried: delivery-side and near/far-post structure as explicit features, a
  shared encoder trained jointly on both heads, class weighting on a 0.275 base
  rate, and simply more data than eight competitions.
- **The nearest-to-goal heuristic is barely distinguishable from chance, and the
  metrics table should not be read as "GNN 0.724 against a real baseline".**
  Chance over the candidates is 0.149 top-1 and 0.441 top-3; the heuristic
  scores 0.158 and 0.496. It is a floor, not a competitor. The top-3 floor is
  high because a corner with three or fewer candidate attackers is a free hit:
  20 of the 923 attacker-first rows (2.2%) are in that position, so at most 2.2
  points of every model's top-3 number is unearned. That is too small to move
  the GNN-versus-heuristic gap of 0.228, but the floor is now printed next to
  the column so nobody has to work it out.
- **The shot-prior baseline's 0.485 AUC is not evidence of anything.** It is a
  constant within each fold, and the training-fold rate is a strictly decreasing
  function of the held-out fold's own rate, so the score is anti-correlated with
  the label by construction and lands below 0.5 systematically rather than by
  noise. Measured: fold test rates [0.264, 0.294, 0.278, 0.266, 0.272] against
  constant scores [0.278, 0.270, 0.274, 0.277, 0.275], perfectly anti-monotone;
  a genuinely constant score scores exactly 0.500 on the same labels.
- **The receiver table is scored on 2345 of 2464 corners and that subset is not
  the population.** The 119 dropped rows are attacker-first corners whose
  receiver was off camera, so dropping them lifts the defence-wins share from
  0.577 to 0.606. All three models are scored on identical rows, so the columns
  compare cleanly; the overall-accuracy and defence-AUC values are not
  population estimates and should not be quoted as such.
- **GNN training is only reproducible at a fixed thread count, and the GNN
  column of `docs/metrics.md` therefore moves in the third decimal between
  runs.** Re-running `setpiece train` on identical data with identical seeds
  reproduced both baselines exactly (logistic regression 0.417 / 0.713 / 0.636 /
  0.639 and shot AUC 0.593; heuristic 0.158 / 0.496 / 0.606 / 0.476) while the
  GNN moved: top-1 0.426 to 0.427, overall 0.640 to 0.643, defence AUC 0.626 to
  0.623, shot AUC 0.569 to 0.566. The cause is torch's parallel reductions, which
  sum in thread-partition order: training the same model on the same rows with
  `OMP_NUM_THREADS=4` and with `OMP_NUM_THREADS=1` gives different weights
  (state-dict digests `561b3c60…` and `d0e77a07…`), while two runs at the same
  thread count are bit-identical. `set_seed` used to claim outright that CPU
  training is bit-reproducible; the docstring now states the condition and the
  model card records `training.torch_threads`. The practical consequence: a
  0.003 difference between two GNN numbers is noise, so the shot-head gap to the
  logistic baseline (0.027) is real but the defence-AUC gap (0.016) is only
  about five times the drift and should not be leaned on hard. Nobody has run
  the obvious fix, which is repeated seeds with a reported spread instead of one
  run reported to three decimals.
- **`setpiece suggest` optimises the weaker head.** The search minimises the GNN
  shot probability even though the logistic regression ranks shots better out of
  fold. No experiment has compared the two as search objectives, and no
  experiment has checked that the suggested moves survive being re-scored by a
  different model. Until one does, a delta from `suggest` is one model's
  opinion, not a measured improvement.
- **Freeze frames carry no velocities.** Every model here is `assume_static`.
  Nothing in the codebase fabricates a velocity, and the suggestion search has
  no way to know whether a defender could actually cover the ground it proposes.
  Any future "can he get there in time?" filter needs tracking data, which open
  data does not provide.
- The keeper is excluded from the search by default (`move_keeper=False`). That
  is sensible for a coaching tool but it is a default, not a constraint, and the
  CLI exposes no flag to change it.
- Real 360 frames sometimes place two players closer than `min_sep` (0.5 m). The
  search never creates such a pair but cannot repair one between two players it
  does not move. That is now stated in `suggest()`'s own docstring and asserted
  in `tests/test_suggest_stress.py::test_a_pre_existing_overlap_is_never_made_worse`,
  in the only form that is actually checkable: every tight pair left after a
  suggestion was tight before it, and involves no player the search moved.
- **Only the six defenders nearest the goal are ever candidates for the first
  move** (`MAX_CANDIDATE_DEFENDERS`), and only the nearest two at deeper levels
  (`DEEP_CANDIDATE_DEFENDERS`). On a frame with ten outfield defenders the four
  furthest from goal can never be repositioned at all, whatever the shape. That
  is a deliberate bound on the branching factor, chosen to fit `max_evals`, but
  it is a search restriction the CLI does not surface and no flag can lift.
- No calibration work has been done on either head. The probabilities are
  softmax and sigmoid outputs and have not been checked against observed
  frequencies, so a "0.33" is a ranking score, not a claim about a third of
  corners.

## tooling and docs

- **`setpiece train` does not update the bundled checkpoints.** It writes
  `receiver.pt`, `shot.pt` and `report.json` to `--out-dir` (default
  `runs/latest`), and the checkpoints under `setpiece/assets/` have to be
  replaced by hand. The missing-model message in `setpiece/cli.py`
  (`_train_hint`) says "Train one, which writes the bundled checkpoints", which
  is wrong. Either make `train` write the assets or fix the message; `cli.py` is
  not this task's file to edit.
- **There is no `LICENSE` file.** `pyproject.toml` declares MIT and `README.md`
  says so, but the text is not in the tree. Add it before publishing, along with
  a note that the StatsBomb Public Data User Agreement — not MIT — governs the
  data, the bundled demo corners and the trained checkpoints.
- **Screenshots are not in the tree.** `README.md` references
  `assets/pitch-editor.png`, `assets/suggest-overlay.png` and
  `assets/score-cli.png`; `assets/` is currently empty. They land at the first
  tagged release.
- `setpiece demo` meets the ten second budget in process — it reports about 1.3 s
  of scoring and searching plus about 7 s to import torch and load the
  checkpoints — but measured end to end through `uv run` on a cold cache the
  wall clock came in at around 11 s. The contract's "under 10 s" is only true
  once the interpreter is warm.
- **Nothing pins torch's thread count outside the CLI, and the Streamlit app is
  outside it.** `_tune_torch_threads` in `setpiece/cli.py` sets
  `torch.set_num_threads(1)` for `score`, `suggest` and `demo`, but `_cmd_app`
  does not call it and would not help if it did: it launches
  `streamlit run setpiece/app.py` as a fresh subprocess, which inherits the
  parent's environment and none of its in-process torch settings. Measured on a
  22-player frame with the bundled `shot.pt`: a default `suggest()` takes 0.6 s
  pinned to one thread and 2.5-2.9 s at torch's default four. The pitch editor's
  documented "under 1 s round trip on CPU" therefore does not hold as launched.
  The fix is one line — `torch.set_num_threads(1)` in `app.py`'s `main()`, or
  `OMP_NUM_THREADS=1` in the child environment `_cmd_app` builds — but it
  belongs to `app.py`, not to this audit.
- The test suite is slow: roughly 46 s for `test_features`, 14 s for
  `test_cli`, 8 s for `test_models`, and a few seconds for the rest. It is not
  hanging; it is doing real work on real graphs. `pyproject.toml` declares
  `slow` and `network` markers but exactly one test uses `slow` and none use
  `network`, so `pytest -m "not slow"` buys almost nothing. Marking the heavy
  cases would make the edit loop usable.
- `data/` and `runs/` are gitignored, so a fresh clone can score corners with
  the bundled checkpoints but cannot reproduce the metrics without a full
  `setpiece fetch` — several hundred matches at one request per second, about an
  hour of mostly waiting. There is no smaller published dataset artefact.

## contract items in `docs/architecture.md`

Checked against the code as it stands. Met unless listed here.

- `setpiece/suggest.py`: `suggest()` takes three keyword arguments the contract
  does not list — `max_evals` (evaluation budget), `move_keeper`, and `top_k`.
  They are additive and defaulted, so the contract signature still works, but
  the contract should be updated to match rather than left silently exceeded.
- `setpiece/cli.py`: `suggest` grew `--attack`, `--beam`, `--top-k` and
  `--format` beyond the documented flags, `demo` grew `--format` and `--model`,
  and `app` grew `--port` and `--headless`. Same conclusion: harmless, but the
  contract is now the smaller of the two.
- `setpiece/features.py`: `build_graph` returns `CornerData`, a `Data` subclass
  that pins `__inc__` for `y_receiver` so batching does not shift the target.
  The contract says `torch_geometric.data.Data`, which it is by inheritance;
  worth stating explicitly.
- The contract's note that freeze frames are "typically 12-22 visible players"
  understates the spread. Measured over the 2464-corner dataset: 8 to 22
  players, median 19.
