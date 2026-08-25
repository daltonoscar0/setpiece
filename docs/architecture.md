# architecture

Module contract. Every module below owns its files exclusively; nothing else in
the package may edit them. `setpiece/types.py`, `setpiece/_io.py` and
`setpiece/fetch.py` are frozen — read them, import them, do not modify them.

## Coordinates and conventions

- Metres on a 105x68 pitch. The attacking team always attacks towards **+x**.
- Goal centre `(105.0, 34.0)`; posts at `y = 30.34` and `y = 37.66`.
- Corner takers sit at `x ~= 105`, `y ~= 0` or `y ~= 68`.
- Freeze frames are **anonymous** (no player ids) and **variable size**
  (typically 12-22 visible players, not always 22).
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
and must never be used as a model feature** — it is present for diagnostics only.
The same goes for `meta["pass_outcome"]`.

## `setpiece/features.py`

```python
FEATURE_NAMES: list[str]          # length N_NODE_FEATURES, human readable
N_NODE_FEATURES: int
N_EDGE_FEATURES: int

def build_graph(frame: CornerFrame) -> torch_geometric.data.Data: ...
def build_flat(frame: CornerFrame) -> np.ndarray:  # [N, N_NODE_FEATURES], baseline use
```

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

Edges: fully connected within a radius, or k-nearest-neighbour with k around 8,
plus edges to the taker. `edge_attr` carries at least relative `dx`, `dy`,
distance, and a same-team flag.

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
            beam=8, seed=0, defend=True) -> list[Suggestion]: ...
```

Hard constraints, which the tests assert:

- at most `movers` defenders move (**defenders only**; attackers never move);
- each displacement has euclidean length `<= max_move` (3 m) **within floating
  point tolerance**;
- no two players end up closer than `min_sep` (0.5 m) — check against **all**
  players, moved and unmoved, attackers included;
- moved players stay on the pitch;
- deterministic given `seed` — same inputs produce identical output ordering.

`defend=True` minimises shot probability. The search is greedy with a beam.
Return suggestions ranked by `delta` ascending (best improvement first).

## `setpiece/cli.py`

```
setpiece score   <corner> [--format json|csv|pretty] [--model PATH]
setpiece suggest <corner> [--defend] [--movers 3] [--max-move 3.0] [--seed 0]
setpiece demo
setpiece app
setpiece fetch   [--competition NAME] [--max-matches N]   # wraps setpiece.fetch
setpiece train   [...]                                     # wraps setpiece.train
```

`main(argv=None) -> int`. `demo` runs on the three bundled Euro 2024 corners in
`setpiece/assets/demo/`, offline, in under 10 s, and must not touch the network.

## `setpiece/app.py`

Streamlit pitch editor: draggable players, live re-score on drop (under 1 s
round trip on CPU), a suggest button that overlays proposed moves, and PNG
export. Entry point `main()`, launched by `setpiece app` via `streamlit run`.
