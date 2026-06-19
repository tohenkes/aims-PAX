"""
Pure-logic unit tests for the adaptive-threshold workflow in
ALRunningManager.  No real constructors are invoked — stubs are
assembled with __new__ + attribute injection.

Targets:
  al_managers.py: ALRunningManager._update_threshold_if_needed
  al_managers.py: ALRunningManager._should_freeze_threshold
"""

from types import SimpleNamespace

import pytest

from aims_PAX.procedures.al_managers import ALRunningManager
from aims_PAX.tools.uncertainty import get_threshold

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

MIN_COUNT = (
    10  # mirrors the hard-coded constant in _update_threshold_if_needed
)
MAX_HIST = (
    400  # mirrors max_uncertainty_history in _update_threshold_if_needed
)


def make_running_manager(
    uncertainties=None,
    threshold=0.5,
    train_dataset_len=0,
    freeze_threshold=False,
    freeze_threshold_dataset=50,
    c_x=0.0,
    analysis=False,
):
    """Minimal ALRunningManager stub for threshold tests."""
    mgr = ALRunningManager.__new__(ALRunningManager)
    mgr.config = SimpleNamespace(
        freeze_threshold=freeze_threshold,
        freeze_threshold_dataset=freeze_threshold_dataset,
        c_x=c_x,
        analysis=analysis,
    )
    mgr.state_manager = SimpleNamespace(
        uncertainties=list(uncertainties) if uncertainties is not None else [],
        threshold=threshold,
        collect_thresholds={0: []},
    )
    mgr.ensemble_manager = SimpleNamespace(
        train_dataset_len=train_dataset_len,
    )
    return mgr


# ---------------------------------------------------------------------------
# §1 — Accumulation
# ---------------------------------------------------------------------------


def test_accumulation_uncertainties_grows():
    """Pushing uncertainties via _update_threshold_if_needed accumulates them."""
    mgr = make_running_manager(freeze_threshold=True)
    # We push directly onto state_manager.uncertainties (as the caller does)
    # and call _update_threshold_if_needed.  With freeze_threshold=True the
    # threshold is not recomputed, so we can focus on accumulation alone.
    values = [float(i) for i in range(15)]
    for v in values:
        mgr.state_manager.uncertainties.append(v)
        mgr._update_threshold_if_needed(0)

    assert mgr.state_manager.uncertainties == values


def test_accumulation_no_update_below_min_count():
    """Threshold stays unchanged when fewer than MIN_COUNT + 1 uncertainties exist."""
    original_threshold = 0.5
    mgr = make_running_manager(
        uncertainties=list(range(MIN_COUNT)),  # exactly min_uncertainty_count
        threshold=original_threshold,
        freeze_threshold=False,
        c_x=0.0,
    )
    mgr._update_threshold_if_needed(0)
    assert mgr.state_manager.threshold == original_threshold


# ---------------------------------------------------------------------------
# §2 — Recompute
# ---------------------------------------------------------------------------


def test_recompute_matches_hand_computation():
    """
    threshold == mean(uncertainties[-MAX_HIST:]) * (1 + c_x).

    The asserted value is computed independently — changing a single
    uncertainty entry will change the expected result.
    """
    c_x = 0.2
    uncertainties = [float(i) for i in range(MIN_COUNT + 1)]
    mgr = make_running_manager(
        uncertainties=uncertainties,
        freeze_threshold=False,
        c_x=c_x,
    )
    mgr._update_threshold_if_needed(0)

    expected = get_threshold(uncertainties, c_x=c_x, max_len=MAX_HIST)
    assert mgr.state_manager.threshold == pytest.approx(expected)


def test_recompute_is_mutation_sensitive():
    """Changing a single uncertainty value changes the computed threshold."""
    c_x = 0.0
    base = [float(i) for i in range(MIN_COUNT + 1)]

    mgr_a = make_running_manager(
        uncertainties=base, freeze_threshold=False, c_x=c_x
    )
    mgr_a._update_threshold_if_needed(0)

    mutated = list(base)
    mutated[-1] += 100.0  # change the last value significantly
    mgr_b = make_running_manager(
        uncertainties=mutated, freeze_threshold=False, c_x=c_x
    )
    mgr_b._update_threshold_if_needed(0)

    assert mgr_a.state_manager.threshold != mgr_b.state_manager.threshold


# ---------------------------------------------------------------------------
# §3 — Freeze
# ---------------------------------------------------------------------------


def test_freeze_at_exact_boundary():
    """
    When train_dataset_len == freeze_threshold_dataset the threshold is
    frozen and NOT recomputed (>= boundary).
    """
    freeze_size = 50
    original_threshold = 0.99
    uncertainties = [float(i) for i in range(MIN_COUNT + 1)]
    mgr = make_running_manager(
        uncertainties=uncertainties,
        threshold=original_threshold,
        train_dataset_len=freeze_size,
        freeze_threshold=False,
        freeze_threshold_dataset=freeze_size,
        c_x=0.5,
    )
    mgr._update_threshold_if_needed(0)

    # Threshold must NOT have changed — freeze kicked in at exactly the boundary
    assert mgr.state_manager.threshold == original_threshold
    # And the freeze flag must now be set
    assert mgr.config.freeze_threshold is True


def test_freeze_below_boundary_does_not_freeze():
    """
    When train_dataset_len < freeze_threshold_dataset the threshold IS
    recomputed (one below the freeze boundary).
    """
    freeze_size = 50
    original_threshold = 0.99
    uncertainties = [float(i) for i in range(MIN_COUNT + 1)]
    mgr = make_running_manager(
        uncertainties=uncertainties,
        threshold=original_threshold,
        train_dataset_len=freeze_size - 1,
        freeze_threshold=False,
        freeze_threshold_dataset=freeze_size,
        c_x=0.0,
    )
    mgr._update_threshold_if_needed(0)

    # Threshold must have been recomputed
    expected = get_threshold(uncertainties, c_x=0.0, max_len=MAX_HIST)
    assert mgr.state_manager.threshold == pytest.approx(expected)
    assert mgr.config.freeze_threshold is False


def test_freeze_above_boundary():
    """train_dataset_len > freeze_threshold_dataset also freezes."""
    freeze_size = 50
    original_threshold = 0.77
    uncertainties = [float(i) for i in range(MIN_COUNT + 1)]
    mgr = make_running_manager(
        uncertainties=uncertainties,
        threshold=original_threshold,
        train_dataset_len=freeze_size + 10,
        freeze_threshold=False,
        freeze_threshold_dataset=freeze_size,
        c_x=0.5,
    )
    mgr._update_threshold_if_needed(0)

    assert mgr.state_manager.threshold == original_threshold
    assert mgr.config.freeze_threshold is True


# ---------------------------------------------------------------------------
# §4 — c_x sign
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "c_x_neg, c_x_pos",
    [(-0.3, 0.3), (-0.1, 0.1), (-0.5, 0.5)],
)
def test_cx_sign_negative_tightens_positive_loosens(c_x_neg, c_x_pos):
    """
    Negative c_x produces a lower (tighter) threshold;
    positive c_x produces a higher (looser) threshold.
    """
    uncertainties = [float(i) for i in range(MIN_COUNT + 1)]

    mgr_neg = make_running_manager(
        uncertainties=list(uncertainties),
        freeze_threshold=False,
        c_x=c_x_neg,
    )
    mgr_neg._update_threshold_if_needed(0)

    mgr_base = make_running_manager(
        uncertainties=list(uncertainties),
        freeze_threshold=False,
        c_x=0.0,
    )
    mgr_base._update_threshold_if_needed(0)

    mgr_pos = make_running_manager(
        uncertainties=list(uncertainties),
        freeze_threshold=False,
        c_x=c_x_pos,
    )
    mgr_pos._update_threshold_if_needed(0)

    assert mgr_neg.state_manager.threshold < mgr_base.state_manager.threshold
    assert mgr_pos.state_manager.threshold > mgr_base.state_manager.threshold
