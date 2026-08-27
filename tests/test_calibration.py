"""Tests for reliability curves, calibration error and the two calibrators.

The failure mode that matters here is a calibration statistic that *looks*
plausible while being wrong, because nothing downstream would catch it: a
reliability table is only ever read by a human. So the arithmetic is pinned
against hand-computed examples, the two invariants that make a calibrator safe
to ship are asserted directly (a monotone map cannot change AUC; a perfectly
calibrated input must survive being calibrated), and the degenerate inputs that
occur in practice — one class, a constant score, a single distinct value — are
required to return something defensible rather than raise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from setpiece.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    calibration_report,
    calibrator_from_card,
    calibrator_from_dict,
    expected_calibration_error,
    fit_calibrator,
    format_reliability_markdown,
    maximum_calibration_error,
    reliability_curve,
)
from setpiece.train import roc_auc


# --------------------------------------------------------------------------- #
# reliability curves
# --------------------------------------------------------------------------- #
def test_reliability_curve_hand_computed() -> None:
    """Four rows in two of ten equal-width bins.

    Predictions 0.05 and 0.15 land in bins [0.0, 0.1) and [0.1, 0.2); the two
    rows in each bin carry labels 0 and 1, so the observed frequency is 0.5 in
    both and the mean predictions are 0.05 and 0.15.
    """
    curve = reliability_curve([0, 1, 0, 1], [0.05, 0.05, 0.15, 0.15], bins=10)
    assert len(curve) == 2
    assert curve[0].count == 2
    assert curve[0].mean_pred == pytest.approx(0.05)
    assert curve[0].frac_pos == pytest.approx(0.5)
    assert curve[0].gap == pytest.approx(0.45)
    assert curve[1].mean_pred == pytest.approx(0.15)
    assert curve[1].gap == pytest.approx(0.35)


def test_empty_bins_are_dropped_not_reported_as_zero() -> None:
    """A bin with no rows is no evidence, and must not be plotted as a zero."""
    curve = reliability_curve([0, 1], [0.05, 0.95], bins=10)
    assert len(curve) == 2
    assert all(b.count > 0 for b in curve)


def test_probability_of_one_is_counted() -> None:
    """The last bin is closed, so ``p == 1`` must not fall off the end."""
    curve = reliability_curve([1, 1], [1.0, 1.0], bins=10)
    assert sum(b.count for b in curve) == 2


def test_ece_hand_computed() -> None:
    """Weighted mean gap over the two bins of the example above.

    Bin one: 2 rows, gap |0.5 - 0.05| = 0.45. Bin two: 2 rows, gap 0.35.
    Weighted: (2 * 0.45 + 2 * 0.35) / 4 = 0.40.
    """
    ece = expected_calibration_error([0, 1, 0, 1], [0.05, 0.05, 0.15, 0.15], bins=10)
    assert ece == pytest.approx(0.40)
    assert maximum_calibration_error([0, 1, 0, 1], [0.05, 0.05, 0.15, 0.15], bins=10) == (
        pytest.approx(0.45)
    )


def test_perfectly_calibrated_scores_have_zero_ece() -> None:
    """Ten rows at 0.3 with exactly three positives is perfect at this resolution."""
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    p = [0.3] * 10
    assert expected_calibration_error(y, p, bins=10) == pytest.approx(0.0)
    assert expected_calibration_error(y, p, bins=10, strategy="quantile") == pytest.approx(0.0)


def test_coarser_bins_never_report_more_error() -> None:
    """ECE is a binned statistic, so the bin count has to be quoted with it."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=500)
    y = (rng.uniform(size=500) < p**1.5).astype(float)
    fine = expected_calibration_error(y, p, bins=20)
    coarse = expected_calibration_error(y, p, bins=2)
    assert coarse <= fine + 1e-9


def test_brier_hand_computed() -> None:
    # (0.9-1)^2 + (0.2-0)^2 = 0.01 + 0.04, over two rows
    assert brier_score([1, 0], [0.9, 0.2]) == pytest.approx(0.025)


def test_ece_rejects_scores_outside_the_unit_interval() -> None:
    """A logit handed in by mistake must fail loudly, not be silently clipped."""
    with pytest.raises(ValueError, match="logit"):
        expected_calibration_error([0, 1], [-2.0, 3.0])


def test_ece_rejects_length_mismatch_and_non_finite() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1, 1], [0.1, 0.2])
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.1, float("nan")])


def test_empty_input_is_nan_not_a_crash() -> None:
    assert math.isnan(expected_calibration_error([], []))
    assert math.isnan(brier_score([], []))
    assert reliability_curve([], []) == []


def test_quantile_bins_split_a_narrow_band_that_uniform_bins_collapse() -> None:
    """The shot head lives inside a narrow band, which uniform bins hide."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.20, 0.34, size=400)
    y = (rng.uniform(size=400) < 0.9).astype(float)  # badly over-confident downward
    uniform = reliability_curve(y, p, bins=10, strategy="uniform")
    quantile = reliability_curve(y, p, bins=10, strategy="quantile")
    assert len(uniform) <= 2
    assert len(quantile) == 10


# --------------------------------------------------------------------------- #
# Platt
# --------------------------------------------------------------------------- #
def test_platt_recovers_a_known_distortion() -> None:
    """Squash a well-calibrated score towards the middle; Platt must undo it.

    ``p_bad = sigmoid(0.5 * logit(p))`` is the classic under-confident head. The
    inverse is ``a = 2, b = 0``, and the fit should land near it.
    """
    rng = np.random.default_rng(11)
    p_true = rng.uniform(0.02, 0.98, size=6000)
    y = (rng.uniform(size=6000) < p_true).astype(float)
    z = np.log(p_true / (1.0 - p_true))
    p_bad = 1.0 / (1.0 + np.exp(-0.5 * z))

    cal = PlattCalibrator.fit(y, p_bad)
    assert cal.a == pytest.approx(2.0, abs=0.15)
    assert cal.b == pytest.approx(0.0, abs=0.15)
    assert expected_calibration_error(y, cal.predict(p_bad)) < expected_calibration_error(y, p_bad)


def test_platt_shrinks_an_overconfident_head() -> None:
    """``a < 1`` is the signature of a head that was too sure of itself."""
    rng = np.random.default_rng(5)
    p_true = rng.uniform(0.05, 0.95, size=4000)
    y = (rng.uniform(size=4000) < p_true).astype(float)
    z = np.log(p_true / (1.0 - p_true))
    p_bad = 1.0 / (1.0 + np.exp(-2.5 * z))
    assert PlattCalibrator.fit(y, p_bad).a < 1.0


def test_platt_on_one_class_is_the_identity() -> None:
    """No likelihood to maximise: refuse to invent a map."""
    cal = PlattCalibrator.fit([1, 1, 1], [0.2, 0.5, 0.9])
    assert (cal.a, cal.b) == (1.0, 0.0)
    assert cal.predict([0.4]) == pytest.approx([0.4])


def test_platt_survives_perfect_separation() -> None:
    """Separable data sends the unregularised MLE to infinity; the ridge stops it."""
    y = [0.0] * 50 + [1.0] * 50
    p = [0.01] * 50 + [0.99] * 50
    cal = PlattCalibrator.fit(y, p)
    assert np.isfinite(cal.a) and np.isfinite(cal.b)
    out = cal.predict(p)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_platt_maps_a_constant_score_to_the_base_rate() -> None:
    """With no spread there is nothing to rank, so the only honest output is the rate."""
    y = np.array([1.0] * 30 + [0.0] * 70)
    p = np.full(100, 0.9)
    out = PlattCalibrator.fit(y, p).predict(p)
    assert out.mean() == pytest.approx(0.30, abs=0.02)


# --------------------------------------------------------------------------- #
# isotonic
# --------------------------------------------------------------------------- #
def test_isotonic_output_is_non_decreasing() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(size=300)
    y = (rng.uniform(size=300) < p).astype(float)
    cal = IsotonicCalibrator.fit(y, p)
    grid = np.linspace(0.0, 1.0, 101)
    out = cal.predict(grid)
    assert (np.diff(out) >= -1e-12).all()


def test_isotonic_fits_a_step_it_can_see() -> None:
    """Two clean groups: below 0.5 never happens, above 0.5 always does."""
    p = np.concatenate([np.linspace(0.1, 0.4, 50), np.linspace(0.6, 0.9, 50)])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    cal = IsotonicCalibrator.fit(y, p)
    assert cal.predict([0.2])[0] == pytest.approx(0.0, abs=1e-9)
    assert cal.predict([0.8])[0] == pytest.approx(1.0, abs=1e-9)


def test_isotonic_gives_tied_predictions_one_value() -> None:
    """Rows with identical p must get identical output, whatever their labels."""
    p = [0.4, 0.4, 0.4, 0.9, 0.9]
    y = [1.0, 0.0, 0.0, 1.0, 1.0]
    out = IsotonicCalibrator.fit(y, p).predict(p)
    assert out[0] == pytest.approx(out[1]) == pytest.approx(out[2])
    assert out[0] == pytest.approx(1 / 3)


def test_isotonic_clamps_outside_its_fitted_range() -> None:
    p = np.linspace(0.3, 0.7, 40)
    y = (p > 0.5).astype(float)
    cal = IsotonicCalibrator.fit(y, p)
    assert cal.predict([0.0])[0] == pytest.approx(cal.y[0])
    assert cal.predict([1.0])[0] == pytest.approx(cal.y[-1])


def test_isotonic_on_a_single_distinct_value() -> None:
    out = IsotonicCalibrator.fit([1, 0, 0, 0], [0.5] * 4).predict([0.1, 0.5, 0.9])
    assert out == pytest.approx([0.25, 0.25, 0.25])


# --------------------------------------------------------------------------- #
# the invariant that makes calibration safe to ship
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["platt", "isotonic", "identity"])
def test_calibration_cannot_change_auc(kind: str) -> None:
    """A monotone map cannot reorder anything, so ranking metrics are untouched.

    This is the whole reason calibration is free: every AUC in ``docs/metrics.md``
    is identical before and after. Isotonic can create ties, which move an AUC by
    at most the tied mass, so it is compared with a tolerance rather than exactly.
    """
    rng = np.random.default_rng(9)
    p = rng.uniform(0.1, 0.6, size=800)
    y = (rng.uniform(size=800) < p * 0.8).astype(float)
    cal = fit_calibrator(y, p, kind=kind)
    before = roc_auc(y, p)
    after = roc_auc(y, cal.predict(p))
    assert after == pytest.approx(before, abs=0.02)


def test_calibrating_an_already_calibrated_head_does_no_harm() -> None:
    rng = np.random.default_rng(13)
    p = rng.uniform(0.05, 0.95, size=4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    before = expected_calibration_error(y, p)
    after = expected_calibration_error(y, PlattCalibrator.fit(y, p).predict(p))
    assert after <= before + 0.01


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["platt", "isotonic", "identity"])
def test_round_trip_through_a_dict(kind: str) -> None:
    """A calibrator ships inside a JSON model card, so it must survive that trip."""
    import json

    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.8, size=200)
    y = (rng.uniform(size=200) < p).astype(float)
    cal = fit_calibrator(y, p, kind=kind)
    payload = json.loads(json.dumps(cal.to_dict()))
    restored = calibrator_from_dict(payload)
    assert restored.predict(p) == pytest.approx(cal.predict(p))


def test_unknown_calibrator_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        fit_calibrator([0, 1], [0.1, 0.9], kind="magic")
    with pytest.raises(ValueError):
        calibrator_from_dict({"kind": "magic"})


def test_calibrator_from_card_distinguishes_absent_from_identity() -> None:
    """"This checkpoint predates calibration" is not "this head needed no fix"."""
    assert calibrator_from_card({}, "shot") is None
    assert calibrator_from_card({"calibration": {}}, "shot") is None
    card = {"calibration": {"shot": {"calibrator": IdentityCalibrator().to_dict()}}}
    assert isinstance(calibrator_from_card(card, "shot"), IdentityCalibrator)
    assert calibrator_from_card(card, "defence") is None


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_calibration_report_has_everything_the_table_quotes() -> None:
    rng = np.random.default_rng(17)
    p = rng.uniform(0.15, 0.45, size=300)
    y = (rng.uniform(size=300) < 0.275).astype(float)
    rep = calibration_report(y, p, bins=10)
    for key in ("n", "base_rate", "mean_prediction", "ece_uniform", "ece_quantile", "brier"):
        assert key in rep
    assert rep["n"] == 300
    assert len(rep["curve"]) > 1
    assert sum(b["count"] for b in rep["curve"]) == 300


def test_reliability_markdown_renders_every_bin() -> None:
    curve = reliability_curve([0, 1, 0, 1], [0.05, 0.05, 0.15, 0.15], bins=10)
    text = format_reliability_markdown(curve, title="shot head")
    assert "shot head" in text
    assert text.count("\n|") >= len(curve) + 1
    assert "+0.450" in text


# --------------------------------------------------------------------------- #
# regressions
# --------------------------------------------------------------------------- #
def test_quantile_bins_survive_a_head_that_emits_one_value() -> None:
    """A constant score has no quantile spread; the curve must still exist.

    ``np.unique`` collapses every quantile edge to a single value, and widening
    "the first" and "the last" edge in place then writes the same element twice
    and leaves zero bins. Every row falls outside the curve, and a perfectly
    calibrated constant head reports ``nan`` instead of ``0.0``.
    """
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    p = [0.3] * 10
    curve = reliability_curve(y, p, bins=10, strategy="quantile")
    assert len(curve) == 1
    assert curve[0].count == 10
    assert curve[0].mean_pred == pytest.approx(0.3)
    assert curve[0].frac_pos == pytest.approx(0.3)
    assert expected_calibration_error(y, p, bins=10, strategy="quantile") == pytest.approx(0.0)


def test_quantile_bins_survive_a_head_with_two_distinct_values() -> None:
    y = [1, 0, 1, 1, 0, 0]
    p = [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]
    curve = reliability_curve(y, p, bins=10, strategy="quantile")
    assert sum(b.count for b in curve) == 6


def test_platt_does_not_diverge_on_a_saturated_head() -> None:
    """The failure the damping exists to stop, pinned as a number.

    Undamped Newton on a head distorted by a factor of 2.5 leaves the saturated
    region and never comes back: it returns ``a`` around 8.6e9. A damped step
    lands near the 1/2.5 = 0.4 that inverts the distortion.
    """
    rng = np.random.default_rng(5)
    p_true = rng.uniform(0.05, 0.95, size=4000)
    y = (rng.uniform(size=4000) < p_true).astype(float)
    z = np.log(p_true / (1.0 - p_true))
    p_bad = 1.0 / (1.0 + np.exp(-2.5 * z))
    fitted = PlattCalibrator.fit(y, p_bad)
    assert 0.2 < fitted.a < 0.7
    assert abs(fitted.b) < 1.0


def test_platt_never_increases_the_negative_log_likelihood() -> None:
    """Every accepted step is a descent step, so the fit beats where it started."""
    rng = np.random.default_rng(11)
    p = rng.uniform(0.001, 0.999, size=500)
    y = (rng.uniform(size=500) < np.clip(p * 0.4, 0, 1)).astype(float)

    def nll(a: float, b: float) -> float:
        u = a * np.log(p / (1.0 - p)) + b
        return float(np.sum(np.logaddexp(0.0, u) - y * u))

    fitted = PlattCalibrator.fit(y, p)
    assert nll(fitted.a, fitted.b) <= nll(1.0, 0.0)
