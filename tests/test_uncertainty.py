"""
Phase 1 — numeric-kernel tests for `aims_PAX.tools.uncertainty`.

These exercise the real NumPy math (no mocks). Values are hand-computed and
mutation-sensitive: perturbing a kernel must make a test fail.

NOTE on `ensemble_sd`: it averages the squared deviation over BOTH the ensemble
members and the xyz components (uncertainty.py:25-28). The fixtures below differ
in all three components so the expected std-dev is a clean integer.
"""

import numpy as np
import pytest

from aims_PAX.tools.uncertainty import (
    HandleUncertainty,
    MolForceUncertainty,
    UDDCalculator,
    get_threshold,
)


def ensemble_4d():
    # [n_members=2, n_mols=1, n_atoms=2, xyz=3]
    m0 = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    m1 = np.array([[[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]])
    return np.stack([m0, m1], axis=0)  # atom0 differs by 2 in every component


# ---------------------------------------------------------------------------
# §1 HandleUncertainty — ensemble_sd / max_atomic_sd / mean_atomic_sd
# ---------------------------------------------------------------------------


def test_ensemble_sd_known_values():
    h = HandleUncertainty("ensemble_sd")
    sd = h.ensemble_sd(ensemble_4d())
    # shape [n_mols, n_atoms]; atom0 sd == 1.0, atom1 sd == 0.0
    assert sd.shape == (1, 2)
    np.testing.assert_allclose(sd, [[1.0, 0.0]])


def test_max_atomic_sd():
    h = HandleUncertainty("max_atomic_sd")
    out = h.max_atomic_sd(np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(out, [1.0])


def test_mean_atomic_sd():
    h = HandleUncertainty("mean_atomic_sd")
    out = h.mean_atomic_sd(np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(out, [0.5])


@pytest.mark.parametrize(
    "utype,expected,expected_shape",
    [
        ("ensemble_sd", [[1.0, 0.0]], (1, 2)),
        ("max_atomic_sd", [1.0], (1,)),
        ("mean_atomic_sd", [0.5], (1,)),
    ],
)
def test_call_dispatch(utype, expected, expected_shape):
    h = HandleUncertainty(utype)
    out = h(ensemble_4d())
    assert out.shape == expected_shape
    np.testing.assert_allclose(out, expected)


def test_call_unknown_type_raises():
    h = HandleUncertainty("bogus")
    with pytest.raises(ValueError, match="not recognized"):
        h(ensemble_4d())


# ---------------------------------------------------------------------------
# §2 MolForceUncertainty
# ---------------------------------------------------------------------------


def test_molforce_global_uncertainty_value():
    mfu = MolForceUncertainty(mol_idxs=[0], uncertainty_type="max_atomic_sd")
    mfu.get_global_uncertainty(ensemble_4d())
    # max over atoms of [[1.0, 0.0]] -> [[1.0]], reshaped to column
    assert mfu.global_uncerstainty.shape == (1, 1)
    np.testing.assert_allclose(mfu.global_uncerstainty, [[1.0]])


def test_molforce_global_uncertainty_mean_variant():
    mfu = MolForceUncertainty(mol_idxs=[0], uncertainty_type="mean_atomic_sd")
    mfu.get_global_uncertainty(ensemble_4d())
    # mean over atoms of [[1.0, 0.0]] -> 0.5
    np.testing.assert_allclose(mfu.global_uncerstainty, [[0.5]])


@pytest.mark.parametrize("n_mols", [1, 2, 3])
def test_compute_mol_forces_4d_shape_any_n_mols(n_mols):
    # Regression: previously only worked when n_mols == n_members (== 2),
    # raising ValueError otherwise. A single-atom molecule must work for any
    # n_mols. [n_members=2, n_mols, n_atoms=2, xyz=3]
    pred = np.arange(2 * n_mols * 2 * 3, dtype=float).reshape(2, n_mols, 2, 3)
    mfu = MolForceUncertainty(mol_idxs=[[0]], uncertainty_type="max_atomic_sd")
    out = mfu.compute_mol_forces_ensemble(pred, select_idxs=[[0]])
    assert out.shape == (2, n_mols, 1, 3)
    # single-atom molecule -> net force equals that atom's force
    np.testing.assert_allclose(out[:, :, 0, :], pred[:, :, 0, :])


def test_compute_mol_forces_4d_group_sums_atoms():
    # A molecule defined by atoms {0, 1}: net force == sum over those atoms.
    pred = np.arange(2 * 1 * 2 * 3, dtype=float).reshape(2, 1, 2, 3)
    mfu = MolForceUncertainty(
        mol_idxs=[[0, 1]], uncertainty_type="max_atomic_sd"
    )
    out = mfu.compute_mol_forces_ensemble(pred, select_idxs=[[0, 1]])
    assert out.shape == (2, 1, 1, 3)
    np.testing.assert_allclose(
        out[:, :, 0, :], pred[:, :, 0, :] + pred[:, :, 1, :]
    )


def test_compute_mol_forces_shape_3d():
    # 3-D input [n_members, n_atoms, xyz]
    pred_3d = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    mfu = MolForceUncertainty(mol_idxs=[0], uncertainty_type="max_atomic_sd")
    out = mfu.compute_mol_forces_ensemble(pred_3d, select_idxs=[0])
    assert out.ndim == 3
    assert out.shape == (2, 1, 3)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# §3 get_threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "c_x,expected",
    [(0.0, 2.0), (0.5, 3.0), (-0.5, 1.0)],
)
def test_get_threshold_scaling(c_x, expected):
    assert get_threshold([2.0, 2.0, 2.0], c_x=c_x) == pytest.approx(expected)


def test_get_threshold_maxlen_truncates():
    # only the last 400 of 0..999 are used -> mean of 600..999 == 799.5
    out = get_threshold(list(range(1000)), c_x=0.0, max_len=400)
    assert out == pytest.approx(799.5)


def test_get_threshold_loosens_vs_tightens():
    hist = [1.0, 2.0, 3.0]
    tight = get_threshold(hist, c_x=-0.5)
    base = get_threshold(hist, c_x=0.0)
    loose = get_threshold(hist, c_x=0.5)
    assert tight < base < loose


# ---------------------------------------------------------------------------
# §4 UDDCalculator bias-potential math (no SO3LR backend constructed)
# ---------------------------------------------------------------------------


def test_udd_bias_potential_sign_and_scale():
    obj = UDDCalculator.__new__(UDDCalculator)
    obj.A = 1.0
    obj.num_heads = 2
    obj.results = {
        # [n_heads=2, n_atoms=2]; deviation from consensus energy 2.0 is ±1
        "energies": np.array([[1.0, 1.0], [3.0, 3.0]]),
        "energy": 2.0,
        "forces": np.zeros((2, 3)),
        "forces_comm": np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
    }

    obj._apply_bias_potential_linear(num_atoms=2)

    # sigma_E_2 = 0.5 * sum(dev^2) = 0.5 * 4 = 2.0
    # E_bias = -A * sigma_E_2 / (num_heads * num_atoms) = -1 * 2 / 4 = -0.5
    assert obj.results["energy_bias"] == pytest.approx(-0.5)
    assert obj.results["energy"] == pytest.approx(1.5)

    # f_bias = -A/(num_heads*num_atoms) * sum_h(dev_h * force_dev_h)
    #        = -1/4 * [[1,0,0],[0,0,0]] = [[-0.25,0,0],[0,0,0]]
    assert obj.results["forces_bias"].shape == (2, 3)
    np.testing.assert_allclose(
        obj.results["forces_bias"], [[-0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(
        obj.results["forces"], [[-0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
