# architecture

Module contract. Every module below owns its files exclusively; nothing else in
the package may edit them. `setpiece/types.py`, `setpiece/_io.py` and
`setpiece/fetch.py` are frozen, read them, import them, do not modify them.

## Coordinates and conventions

- Metres on a 105x68 pitch. The attacking team always attacks towards **+x**.
- Goal centre `(105.0, 34.0)`; posts at `y = 30.34` and `y = 37.66`.
- Corner takers sit at `x ~= 105`, `y ~= 0` or `y ~= 68`.
- Freeze frames are **anonymous** (no player ids) and **variable size**.
  Measured over the 2464-corner dataset: **8 to 22 players, median 19**. Frames
  showing fewer than 8 players are dropped at load time.
- Freeze frames carry **no velocities**. Never fabricate them. Docs say
  `--assume-static`.

## Labels

`CornerFrame.first_touch` is one of `"attacker" | "defender" | "unknown"`.
On StatsBomb open data the defence wins first contact slightly more often than
the attack (measured: 10 defender / 8 attacker on a 18-corner sample).

- **Receiver head** is a softmax over *candidate attacker nodes plus one extra
  "defence wins first contact" class*. Candidates are attackers excluding the
  taker. `receiver_idx` is set only when `first_touch == "attacker"` and the
  positional match was unambiguous (`receiver_match_dist <= 8 m`); rows with
  `first_touch == "attacker"` but `receiver_idx is None` must be **dropped**
  from receiver training and evaluation (the receiver was off-camera).
- **Shot head** is binary: `shot_within_5`.

`CornerFrame.meta["end_xy"]` is the delivery destination. It is **label-adjacent
and must never be used as a model feature**, it is present for diagnostics only.
The same goes for `meta["pass_outcome"]`.

## `setpiece/features.py`

```python
FEATURE_NAMES: list[str]          # length N_NODE_FEATURES, human readable
N_NODE_FEATURES: int
N_EDGE_FEATURES: int

def build_graph(frame: CornerFrame) -> CornerData: ...
def build_flat(frame: CornerFrame) -> np.ndarray:  # [N, N_NODE_FEATURES], baseline use
def global_receiver_index(data) -> Tensor: ...     # int64 [B], -1 passed through
def mirror_side(frame: CornerFrame) -> float: ...  # +1.0 or -1.0
def taker_position(frame: CornerFrame) -> tuple[float, float]: ...
```

### Mirrored coordinates

Every frame is reflected about the pitch's long axis, `y = GOAL_Y`, so the
corner is always taken from the `y < GOAL_Y` side. A corner from the left and
its mirror image from the right are the **same coaching situation**, same near
post, same far post, same runs, same marking, and without the reflection the
network has to learn each half of the pitch separately from half the data.

`mirror_side(frame)` returns `+1.0` when the corner already comes from
`y < GOAL_Y` and `-1.0` otherwise; the side is decided by `taker_position`,
which reads the taker flag when the frame has one and otherwise infers the
nearest corner flag (user CSV input). The reflection is applied **once, to the
coordinates**, before any feature is computed. Distances, densities, adjacency
and the box flags are already invariant under it; every signed `y` quantity -
`y_norm`, `dy_goal`, `goal_angle_sin`, `taker_angle_sin`, the edge `dy` -
inherits the flip from the coordinates rather than carrying its own sign rule,
so node and edge features can never disagree about which way is which.

Consequences, all asserted in `tests/test_features.py`:

- `build_graph` is **exactly** invariant to reflection about `y = GOAL_Y`:
  measured over all 2464 corners in `data/interim/corners.jsonl`, max absolute
  difference **0.0** in both `x` and `edge_attr`, with `edge_index` identical.
  The dataset is well split between the flags, 1161 near / 1303 far.
- `N_NODE_FEATURES` and `N_EDGE_FEATURES` are unchanged; this is a change of
  coordinates, not extra columns.
- Player order, team flags and labels are untouched. `y_receiver` still indexes
  the same node.
- **Degenerate cases where invariance cannot hold**, both documented in the
  source and both unreachable from a real corner: a taker exactly on
  `y == GOAL_Y` (side `+1.0`, frame left alone, the frame and its mirror put
  the taker on the same line and so choose the same side, which no taker-based
  rule can avoid), and, on the no-taker-flag fallback, a player set exactly
  symmetric about `y = GOAL_Y`, where both flags are equidistant and the
  tie-break picks the same one for a frame and its mirror. Both still produce
  finite, well-formed features.

A fixture whose taker sits at `y < GOAL_Y` is reflected by the identity, so
assertions written against one stay green whether or not the mirror runs at all.
Geometry tests are therefore parametrised over a near-side **and** a far-side
fixture and state their expectations in mirrored coordinates.

`CornerData` is a `torch_geometric.data.Data` **subclass** that pins
`__inc__("y_receiver") == 0`, so `y_receiver` survives batching as a
**graph-local** node index. The batch-global index is
`batch.ptr[b] + batch.y_receiver[b]`; use `global_receiver_index(batch)`, which
does that and passes the `-1` sentinel through unshifted. There is no `__inc__`
formulation that both shifts real indices and preserves `-1`, because PyG applies
graph `b`'s increment from the cumulative sum of graphs `0..b-1`, a
value-dependent increment silently turns `-1` into a valid-looking node index.

`build_graph` returns a `Data` with exactly these attributes:

| attribute | dtype / shape | meaning |
|---|---|---|
| `x` | float32 `[N, N_NODE_FEATURES]` | node features |
| `edge_index` | int64 `[2, E]` | directed, both directions present |
| `edge_attr` | float32 `[E, N_EDGE_FEATURES]` | pairwise features |
| `is_attacker` | bool `[N]` | |
| `is_taker` | bool `[N]` | |
| `is_keeper` | bool `[N]` | |
| `candidate_mask` | bool `[N]` | attackers excluding the taker |
| `y_receiver` | int64 `[1]` | node index of receiver, or `-1` for "defence wins" |
| `y_receiver_valid` | bool `[1]` | False when the row must be excluded from receiver loss |
| `y_shot` | float32 `[1]` | 0.0 / 1.0 |
| `num_nodes` | int | |

Node features must include, at minimum: normalised `x`, `y`; `is_attacker`,
`is_taker`, `is_keeper` flags; euclidean distance to goal centre; angle to goal
centre; the angle subtended by the two posts; distance to the taker; and local
density (count of any player within 3 m and within 5 m). Distances are in metres
and should be scaled to roughly unit range before entering the network.

All of them are computed in the mirrored coordinates above, so `y` and the
signed angles are "distance from the taker's touchline", not "distance from
`y = 0`".

Edges: fully connected within a radius, or k-nearest-neighbour with k around 8,
plus edges to the taker. `edge_attr` carries at least relative `dx`, `dy`,
distance, and a same-team flag, `dy` in the same mirrored coordinates as the
node features, from the same reflected positions.

`build_graph` must work on a frame with **no labels** (inference path): in that
case `y_receiver = -1`, `y_receiver_valid = False`, `y_shot = 0.0`.

## `setpiece/models.py`

Three-layer message passing, under 1M parameters total per head, CPU inference
under 100 ms per corner.

```python
class CornerEncoder(nn.Module):        # 3 rounds of message passing
    def forward(self, data) -> Tensor: ...      # node embeddings [N, hidden]

class ReceiverModel(nn.Module):
    def forward(self, data) -> tuple[Tensor, Tensor]:
        """node_logits [N_total], defence_logit [B]. Non-candidate nodes are
        masked to -inf so a softmax over candidates + defence_logit is valid."""

class ShotModel(nn.Module):
    def forward(self, data) -> Tensor: ...      # shot logit [B]

def receiver_probabilities(model, frame) -> tuple[np.ndarray, float]:
    """Per-player probability (0.0 for non-candidates) and P(defence wins)."""

def shot_probability(model, frame) -> float: ...

def save_model(model, path: Path, card: dict) -> None: ...
def load_model(path: Path) -> tuple[nn.Module, dict]: ...
def load_bundled(name: str) -> tuple[nn.Module, dict]:
    """Load from setpiece/assets/. Names: 'receiver', 'shot'."""
```

Checkpoints store `state_dict`, the model class name, constructor kwargs, and
the model card. `load_model` must reconstruct without the caller knowing shapes,
and must set eval mode. Loading uses `weights_only=True` where possible.

The **model card** dict carries: `data` (competitions, seasons, n corners),
`splits` (how matches were partitioned, counts), `metrics`, `date`, `features`,
`params`, `assume_static: True`.

## `setpiece/train.py`

- Splits are **purged by match**: every corner from a match lands in exactly one
  fold. Never split within a match. Group by `match_id`.
- Baselines, both evaluated on the identical held-out rows as the GNN:
  1. nearest-to-goal heuristic (receiver), and a shot-rate prior (shot);
  2. logistic regression on the flat per-node features.
- Receiver metrics: top-1 and top-3 accuracy, reported on the attacker-first
  subset so the baselines and the GNN are compared on rows where both can be
  correct. Also report overall accuracy including the "defence wins" class, and
  the AUC for predicting that the defence wins.
- Shot metric: AUC.
- Emit a markdown metrics table to stdout and to `docs/metrics.md`. Report the
  numbers as measured. If a baseline wins, say so.

## `setpiece/suggest.py`

```python
@dataclass
class Move:
    player_idx: int
    dx: float
    dy: float

@dataclass
class Suggestion:
    moves: list[Move]
    base_prob: float
    new_prob: float
    delta: float           # new_prob - base_prob, negative is an improvement

def suggest(frame, shot_model, *, movers=3, max_move=3.0, min_sep=0.5,
            beam=8, seed=0, defend=True,
            max_evals=288, move_keeper=False, top_k=None,
            max_candidates=MAX_CANDIDATE_DEFENDERS,
            deep_candidates=DEEP_CANDIDATE_DEFENDERS) -> list[Suggestion]: ...
```

`max_evals` bounds model evaluations so the app stays interactive. `move_keeper`
is a default, not a constraint. `max_candidates` / `deep_candidates` bound the
branching factor: only the *N* defenders nearest the goal are candidates for the
first move, and only the nearest *M* at deeper levels. On a frame with ten
outfield defenders the remainder can never be repositioned, so both are
adjustable and surfaced by the CLI rather than being silent.

Hard constraints, which the tests assert:

- at most `movers` defenders move (**defenders only**; attackers never move);
- each displacement has euclidean length `<= max_move` (3 m) **within floating
  point tolerance**;
- no two players end up closer than `min_sep` (0.5 m), check against **all**
  players, moved and unmoved, attackers included;
- moved players stay on the pitch;
- deterministic given `seed`, same inputs produce identical output ordering.

`defend=True` minimises shot probability. The search is greedy with a beam.
Return suggestions ranked by `delta` ascending (best improvement first).

## `setpiece/cli.py`

```
setpiece score   <corner> [--format json|csv|pretty] [--model PATH]
setpiece suggest <corner> [--defend | --attack] [--movers 3] [--max-move 3.0]
                          [--seed 0] [--beam 8] [--top-k N] [--format ...]
                          [--max-candidates N] [--deep-candidates N]
setpiece demo    [--format ...] [--model PATH]
setpiece app     [--port 8501] [--headless] [--model PATH]
setpiece fetch   [--competition NAME] [--max-matches N] [--max-attempts 3]
setpiece train   [--install] [...]                         # wraps setpiece.train
```

`train` writes checkpoints to `--out-dir` (default `runs/latest`); `--install`
additionally copies them over the bundled ones in `setpiece/assets/`, which is
what `score`, `suggest`, `demo` and the app load by default.

`app` must pin `torch.set_num_threads(1)` inside `app.py`'s own `main()`. It is
launched as a fresh `streamlit run` subprocess, so no setting made in `cli.py`
reaches it. The pin is worth about 0.2 s: measured on the 22-player frame from
the dataset with the bundled `shot.pt`, five repeats each at a one-minute load
average of 16-21, a default `suggest()` runs in a median 0.69 s at torch's own
default of four threads and 0.45 s pinned to one. An earlier note here claimed
2.5-2.9 s at the default; that was machine contention and does not reproduce.
The pin stays because it is free, provably takes effect (asserted in
`tests/test_app.py` against a real second interpreter) and keeps the editor off
the other cores, not because the interaction budget is close to breaking. An
`OMP_NUM_THREADS` or `MKL_NUM_THREADS` the user set explicitly wins.

`main(argv=None) -> int`. `demo` runs on the three bundled Euro 2024 corners in
`setpiece/assets/demo/`, offline, in under 10 s, and must not touch the network.

## `setpiece/app.py`

Streamlit pitch editor: draggable players, live re-score on drop (under 1 s
round trip on CPU), a suggest button that overlays proposed moves, and PNG
export. Entry point `main()`, launched by `setpiece app` via `streamlit run`.
