"""
Phase 1 — numeric-kernel tests for `aims_PAX.tools.utilities.eval_utils.MACEEval`.

`MACEEval.update` only reads attributes off `batch` and keys off the `output`
dict, so a `SimpleNamespace` with torch tensors is a sufficient stand-in for a
torch_geometric batch — no model required. All values are hand-computed.

(`ensemble_prediction` / `evaluate_model` need a real MACE model and are covered
in Phase 3/6, not here.)
"""

from types import SimpleNamespace

import pytest
import torch

from aims_PAX.tools.utilities.eval_utils import MACEEval


def make_batch(*, energy=None, forces=None, stress=None, num_graphs=1):
    return SimpleNamespace(
        num_graphs=num_graphs,
        energy=energy,
        forces=forces,
        stress=stress,
        virials=None,
        dipole=None,
        ptr=torch.tensor([0, 2]),  # one graph, two atoms
    )


def test_maceeval_energy_mae_rmse():
    metric = MACEEval()
    batch = make_batch(energy=torch.tensor([10.0]))
    metric.update(batch, {"energy": torch.tensor([9.0])})
    aux = metric.compute()
    assert aux["mae_e"] == pytest.approx(1.0)
    assert aux["rmse_e"] == pytest.approx(1.0)
    # per-atom: delta 1.0 over 2 atoms -> 0.5
    assert aux["mae_e_per_atom"] == pytest.approx(0.5)


def test_maceeval_forces_mae():
    metric = MACEEval()
    forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    output = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )  # delta in one comp
    batch = make_batch(forces=forces)
    metric.update(batch, {"forces": output})
    aux = metric.compute()
    # mean(|delta|) over 6 components = 1/6
    assert aux["mae_f"] == pytest.approx(1.0 / 6.0)
    assert "mae_e" not in aux  # energy not supplied


def test_maceeval_skips_absent_channels():
    metric = MACEEval()
    batch = make_batch(
        energy=torch.tensor([1.0]),
        forces=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    metric.update(
        batch,
        {"energy": torch.tensor([1.0]), "forces": batch.forces},
    )
    aux = metric.compute()
    # stress / virials / dipole were None -> their metrics must be absent
    assert "mae_stress" not in aux
    assert "mae_virials" not in aux
    assert "mae_mu" not in aux


def test_maceeval_zero_error():
    metric = MACEEval()
    energy = torch.tensor([5.0])
    forces = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    batch = make_batch(energy=energy, forces=forces)
    metric.update(batch, {"energy": energy.clone(), "forces": forces.clone()})
    aux = metric.compute()
    assert aux["mae_e"] == pytest.approx(0.0)
    assert aux["rmse_e"] == pytest.approx(0.0)
    assert aux["mae_f"] == pytest.approx(0.0)


def test_maceeval_stress_channel():
    metric = MACEEval()
    stress = torch.zeros((1, 3, 3))
    output_stress = torch.full((1, 3, 3), 0.1)
    batch = make_batch(stress=stress)
    metric.update(batch, {"stress": output_stress})
    aux = metric.compute()
    # delta is uniformly -0.1 -> mae 0.1
    assert aux["mae_stress"] == pytest.approx(0.1)
    assert "mae_e" not in aux
