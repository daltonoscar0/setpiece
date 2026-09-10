"""Calibration for the two binary probabilities this project emits.

A softmax or a sigmoid is a *ranking* score until somebody checks it against
observed frequencies. Two of the numbers here are genuine binary probabilities
and can be checked:

- ``shot_within_5`` — the shot head's sigmoid;
- ``P(defence wins first contact)`` — the receiver head's extra softmax class.

The per-attacker receiver probabilities are **not** in that family: they are a
multiclass distribution over a variable-length candidate set, and a reliability
curve over them answers a different question. Only the two above are calibrated
here.

Everything is implemented on numpy so the package keeps its dependency list. The
pieces are:

``reliability_curve``
    Observed frequency against mean predicted probability, per bin.
``expected_calibration_error``
    The bin-count-weighted mean gap between those two, and its ``maximum_``
    sibling. Both take ``strategy="uniform"`` (equal-width bins, the usual
    definition) or ``strategy="quantile"`` (equal-count bins, which is the more
    informative view when a head only ever predicts inside a narrow band).
``PlattCalibrator`` / ``IsotonicCalibrator``
    Two monotone maps from a raw probability to a calibrated one. Both are pure
    post-processing: they cannot change any ranking metric, so an AUC is
    identical before and after. Platt is a two-parameter logistic on the logit
    and cannot do much harm on small samples; isotonic is non-parametric and
    will happily overfit a few hundred rows.

**A calibrator must be fitted on rows the model that produced the scores never
saw.** Fitting one on in-sample predictions learns to undo overconfidence that
only exists in-sample and makes held-out calibration worse. :mod:`setpiece.train`
fits these on out-of-fold predictions only; see ``docs/metrics.md``.

Fitted calibrators serialise to plain dicts via :meth:`to_dict`, which is how
they travel inside a model card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Bin",
    "Calibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "brier_score",
    "calibration_report",
    "calibrator_from_dict",
    "expected_calibration_error",
    "fit_calibrator",
    "format_reliability_markdown",
    "maximum_calibration_error",
    "reliability_curve",
]

#: Probabilities are clipped this far from 0 and 1 before a logit is taken.
EPS: float = 1e-6

#: Default bin count for reliability curves and calibration error.
DEFAULT_BINS: int = 10


# --------------------------------------------------------------------------- #
# shared input handling
# --------------------------------------------------------------------------- #
def _check(y_true: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a (labels, probabilities) pair.

    Labels are thresholded at 0.5 so booleans and 0/1 floats both work.
    Probabilities must be finite and inside ``[0, 1]``: a score outside that
    range is a logit somebody forgot to squash, and silently clipping it would
    turn a bug into a plausible-looking calibration curve.
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    q = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.size != q.size:
        raise ValueError(f"length mismatch: {y.size} labels, {q.size} probabilities")
    if q.size and not bool(np.isfinite(q).all()):
        raise ValueError(
            f"{int((~np.isfinite(q)).sum())} of {q.size} probabilities are not finite"
        )
    if q.size and (q.min() < -1e-9 or q.max() > 1.0 + 1e-9):
        raise ValueError(
            f"probabilities must lie in [0, 1]; got [{q.min():.4g}, {q.max():.4g}] "
            "— this looks like a logit rather than a probability"
        )
    return (y > 0.5).astype(np.float64), np.clip(q, 0.0, 1.0)


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


# --------------------------------------------------------------------------- #
# reliability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bin:
    """One bucket of a reliability curve."""

    lower: float
    upper: float
    count: int
    mean_pred: float
    frac_pos: float

    @property
    def gap(self) -> float:
        """Signed calibration gap: positive means the head is under-confident."""
        return self.frac_pos - self.mean_pred


def _edges(p: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, bins + 1)
    if strategy == "quantile":
        if p.size == 0:
            return np.linspace(0.0, 1.0, bins + 1)
        qs = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(p, qs))
        # A head that emits few distinct values collapses quantile edges. A
        # constant score collapses them to a single value, which is not an edge
        # array at all, widening `edges[0]` and `edges[-1]` in place would then
        # write to the same element twice and leave zero bins, so every row would
        # fall outside the curve and the error would come back `nan` rather than
        # the zero it actually is. Keep the ends open, and keep at least one bin.
        if edges.size < 2:
            return np.array([min(float(edges[0]), 0.0), max(float(edges[0]), 1.0)])
        edges[0], edges[-1] = min(edges[0], 0.0), max(edges[-1], 1.0)
        return edges
    raise ValueError(f"unknown binning strategy {strategy!r}; use 'uniform' or 'quantile'")


def reliability_curve(
    y_true: Any, p: Any, *, bins: int = DEFAULT_BINS, strategy: str = "uniform"
) -> list[Bin]:
    """Observed frequency against mean predicted probability, bin by bin.

    Bins are half-open ``[lower, upper)`` with the last one closed, so ``p == 1``
    is counted. Empty bins are dropped rather than reported as zeros, because a
    bin with no rows carries no evidence about calibration and would drag a
    plotted curve towards the origin.
    """
    y, q = _check(y_true, p)
    edges = _edges(q, bins, strategy)
    if q.size == 0:
        return []
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)
    out: list[Bin] = []
    for b in range(len(edges) - 1):
        here = idx == b
        n = int(here.sum())
        if n == 0:
            continue
        out.append(
            Bin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=n,
                mean_pred=float(q[here].mean()),
                frac_pos=float(y[here].mean()),
            )
        )
    return out


def expected_calibration_error(
    y_true: Any, p: Any, *, bins: int = DEFAULT_BINS, strategy: str = "uniform"
) -> float:
    """Bin-count-weighted mean of ``|observed frequency - mean prediction|``.

    Zero means the head's stated probabilities match observed frequencies at this
    resolution. It is a *binned* statistic: a coarser binning always reports less
    error, so quote the bin count alongside it. Returns ``nan`` on empty input.
    """
    curve = reliability_curve(y_true, p, bins=bins, strategy=strategy)
    if not curve:
        return float("nan")
    total = sum(b.count for b in curve)
    return float(sum(b.count * abs(b.gap) for b in curve) / total)


def maximum_calibration_error(
    y_true: Any, p: Any, *, bins: int = DEFAULT_BINS, strategy: str = "uniform"
) -> float:
    """The worst single-bin gap. Sensitive to thinly populated bins by design."""
    curve = reliability_curve(y_true, p, bins=bins, strategy=strategy)
    if not curve:
        return float("nan")
    return float(max(abs(b.gap) for b in curve))


def brier_score(y_true: Any, p: Any) -> float:
    """Mean squared error of the probabilities.

    Unlike ECE this needs no binning, and unlike AUC it moves when a calibrator
    is applied — so the two together separate "ranks well" from "states honest
    probabilities".
    """
    y, q = _check(y_true, p)
    if q.size == 0:
        return float("nan")
    return float(np.mean((q - y) ** 2))


# --------------------------------------------------------------------------- #
# calibrators
# --------------------------------------------------------------------------- #
@runtime_checkable
class Calibrator(Protocol):
    """A fitted, monotone map from a raw probability to a calibrated one."""

    kind: str

    def predict(self, p: Any) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class PlattCalibrator:
    """``sigmoid(a * logit(p) + b)``, fitted by maximum likelihood.

    Two parameters, so it survives a few hundred rows without overfitting, but it
    can only rescale and shift on the logit scale: a head whose miscalibration is
    not monotone in that sense will not be fully fixed. ``a < 1`` means the raw
    head was overconfident and the fit is shrinking it towards the base rate.

    Fitted by **damped** Newton on the penalised negative log-likelihood, with a
    tiny ridge (``1e-6``) so perfectly separable input converges instead of
    running the coefficients to infinity. The damping is not optional. An
    overconfident head — the case this class exists for — starts with many
    predictions already saturated near 0 and 1, so the IRLS weights
    ``mu * (1 - mu)`` are near zero, the Hessian is near-singular, and an
    undamped step overshoots into full saturation and never returns: fitting a
    head distorted by a factor of 2.5 that way returns ``a`` around 8.6e9 instead
    of around 0.4. Backtracking until the objective actually decreases makes each
    step a descent step, so the fit is monotone by construction.

    Deterministic: no seed, no shuffling.
    """

    a: float = 1.0
    b: float = 0.0
    kind: str = "platt"

    @classmethod
    def fit(cls, y_true: Any, p: Any, *, iters: int = 100, ridge: float = 1e-6) -> PlattCalibrator:
        y, q = _check(y_true, p)
        if y.size == 0 or y.min() == y.max():
            # One class, or nothing: there is no likelihood to maximise, and the
            # identity map is the only defensible answer.
            return cls()
        z = _logit(q)
        theta = np.array([1.0, 0.0], dtype=np.float64)
        design = np.stack([z, np.ones_like(z)], axis=1)

        def objective(t: np.ndarray) -> float:
            # logaddexp keeps this finite for large |z| where exp would overflow.
            u = design @ t
            return float(np.sum(np.logaddexp(0.0, u) - y * u) + 0.5 * ridge * float(t @ t))

        current = objective(theta)
        for _ in range(iters):
            mu = _sigmoid(design @ theta)
            grad = design.T @ (mu - y) + ridge * theta
            if float(np.max(np.abs(grad))) < 1e-10:
                break
            w = np.clip(mu * (1.0 - mu), 1e-12, None)
            hess = design.T @ (design * w[:, None]) + ridge * np.eye(2)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:  # pragma: no cover - the ridge keeps this PD
                break
            scale = 1.0
            for _ in range(60):
                candidate = theta - scale * step
                value = objective(candidate)
                if value <= current:
                    break
                scale *= 0.5
            else:
                # No descent direction survives halving; we are at the optimum to
                # the precision this arithmetic supports.
                break
            theta, previous, current = candidate, current, value
            if float(np.max(np.abs(scale * step))) < 1e-10 or previous - current < 1e-14:
                break
        return cls(a=float(theta[0]), b=float(theta[1]))

    def predict(self, p: Any) -> np.ndarray:
        q = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 0.0, 1.0)
        return _sigmoid(self.a * _logit(q) + self.b)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "platt", "a": self.a, "b": self.b}


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool adjacent violators: the least-squares non-decreasing fit."""
    level_v: list[float] = []
    level_w: list[float] = []
    level_n: list[int] = []
    for v, w in zip(values.tolist(), weights.tolist(), strict=True):
        level_v.append(float(v))
        level_w.append(float(w))
        level_n.append(1)
        while len(level_v) > 1 and level_v[-2] > level_v[-1]:
            v2, w2, n2 = level_v.pop(), level_w.pop(), level_n.pop()
            v1, w1, n1 = level_v.pop(), level_w.pop(), level_n.pop()
            total = w1 + w2
            level_v.append((v1 * w1 + v2 * w2) / total if total else 0.5 * (v1 + v2))
            level_w.append(total)
            level_n.append(n1 + n2)
    return np.repeat(np.asarray(level_v, dtype=np.float64), level_n)


@dataclass
class IsotonicCalibrator:
    """A non-decreasing step function fitted by pool-adjacent-violators.

    Strictly more expressive than Platt and correspondingly hungrier: on a few
    hundred out-of-fold rows it can chase noise into a staircase that does not
    generalise, which is why the choice between the two is made by measurement in
    :mod:`setpiece.train` rather than by preference.

    Predictions interpolate linearly between fitted knots and clamp outside their
    range, so the map stays monotone and the calibrated score preserves the raw
    ranking exactly (ties may be created, never reversed).
    """

    x: np.ndarray
    y: np.ndarray
    kind: str = "isotonic"

    @classmethod
    def fit(cls, y_true: Any, p: Any) -> IsotonicCalibrator:
        y, q = _check(y_true, p)
        if y.size == 0:
            return cls(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        # Tied predictions must receive one common fitted value, so aggregate on
        # the unique x first; running PAVA over raw ties would order them by
        # sort position and produce a map that is not a function of p.
        uniq, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
        sums = np.zeros(uniq.size, dtype=np.float64)
        np.add.at(sums, inverse, y)
        means = sums / counts
        fitted = _pava(means, counts.astype(np.float64))
        if uniq.size == 1:
            uniq = np.array([uniq[0], uniq[0]])
            fitted = np.array([fitted[0], fitted[0]])
        return cls(uniq.astype(np.float64), fitted.astype(np.float64))

    def predict(self, p: Any) -> np.ndarray:
        q = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 0.0, 1.0)
        return np.interp(q, self.x, self.y, left=float(self.y[0]), right=float(self.y[-1]))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "isotonic", "x": self.x.tolist(), "y": self.y.tolist()}


@dataclass
class IdentityCalibrator:
    """The null calibrator. Ships when measurement says the head needs no fix."""

    kind: str = "identity"

    def predict(self, p: Any) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "identity"}


def fit_calibrator(y_true: Any, p: Any, *, kind: str = "platt") -> Calibrator:
    """Fit ``kind`` in ``{"platt", "isotonic", "identity"}`` on held-out scores."""
    if kind == "platt":
        return PlattCalibrator.fit(y_true, p)
    if kind == "isotonic":
        return IsotonicCalibrator.fit(y_true, p)
    if kind == "identity":
        return IdentityCalibrator()
    raise ValueError(f"unknown calibrator {kind!r}; use 'platt', 'isotonic' or 'identity'")


def calibrator_from_dict(payload: dict[str, Any]) -> Calibrator:
    """Rebuild a calibrator from :meth:`Calibrator.to_dict`, e.g. out of a card."""
    kind = payload.get("kind")
    if kind == "platt":
        return PlattCalibrator(a=float(payload["a"]), b=float(payload["b"]))
    if kind == "isotonic":
        return IsotonicCalibrator(
            np.asarray(payload["x"], dtype=np.float64),
            np.asarray(payload["y"], dtype=np.float64),
        )
    if kind == "identity":
        return IdentityCalibrator()
    raise ValueError(f"unknown calibrator payload {kind!r}")


def calibrator_from_card(card: dict[str, Any], head: str) -> Calibrator | None:
    """The calibrator a model card ships for ``head``, or ``None`` if it ships none.

    ``head`` is ``"shot"`` on a shot checkpoint and ``"defence"`` on a receiver
    one. Returns ``None`` rather than an identity map so a caller can tell "this
    checkpoint predates calibration" from "this checkpoint was measured and left
    alone".
    """
    entry = (card.get("calibration") or {}).get(head)
    if not entry or not entry.get("calibrator"):
        return None
    return calibrator_from_dict(entry["calibrator"])


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def calibration_report(
    y_true: Any, p: Any, *, bins: int = DEFAULT_BINS
) -> dict[str, Any]:
    """Every calibration statistic this project quotes, for one set of scores.

    Both binning strategies are reported. Equal-width bins are the textbook ECE;
    equal-count bins are the honest view for a head whose predictions all sit in
    a narrow band, where uniform bins put every row in two buckets and report a
    flatteringly small number.
    """
    y, q = _check(y_true, p)
    return {
        "n": int(q.size),
        "base_rate": float(y.mean()) if y.size else float("nan"),
        "mean_prediction": float(q.mean()) if q.size else float("nan"),
        "min_prediction": float(q.min()) if q.size else float("nan"),
        "max_prediction": float(q.max()) if q.size else float("nan"),
        "bins": int(bins),
        "ece_uniform": expected_calibration_error(y, q, bins=bins, strategy="uniform"),
        "ece_quantile": expected_calibration_error(y, q, bins=bins, strategy="quantile"),
        "mce_uniform": maximum_calibration_error(y, q, bins=bins, strategy="uniform"),
        "brier": brier_score(y, q),
        "curve": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_pred": b.mean_pred,
                "frac_pos": b.frac_pos,
            }
            for b in reliability_curve(y, q, bins=bins, strategy="quantile")
        ],
    }


def format_reliability_markdown(curve: list[Bin] | list[dict[str, Any]], *, title: str) -> str:
    """Render a reliability curve as a markdown table."""
    rows = [b if isinstance(b, dict) else b.__dict__ for b in curve]
    lines = [f"**{title}**", "", "| predicted range | n | mean predicted | observed | gap |",
             "|---|---|---|---|---|"]
    for r in rows:
        gap = r["frac_pos"] - r["mean_pred"]
        lines.append(
            f"| {r['lower']:.3f}-{r['upper']:.3f} | {r['count']} | {r['mean_pred']:.3f} "
            f"| {r['frac_pos']:.3f} | {gap:+.3f} |"
        )
    return "\n".join(lines)
