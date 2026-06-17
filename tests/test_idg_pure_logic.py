from types import SimpleNamespace

import numpy as np
import pytest

from aims_PAX.procedures.initial_dataset import (
    InitialDatasetPARSL,
    InitialDatasetPARSLTeacher,
    InitialDatasetProcedure,
)


def make_point():
    return SimpleNamespace(info={}, arrays={})


RESULT = {"energy": -10.5, "forces": [[1.0, 0.0, 0.0]]}


# ===========================================================================
# §1.1 — _get_member_points
# ===========================================================================


def test_distinct_true_member0():
    stub = SimpleNamespace(
        distinct_model_sets=True,
        atoms=[None, None],
        n_points_per_sampling_step_idg=5,
        sampled_points=list(range(60)),
    )
    result = InitialDatasetProcedure._get_member_points(stub, 0)
    assert result == list(range(0, 10))


def test_distinct_true_member1():
    stub = SimpleNamespace(
        distinct_model_sets=True,
        atoms=[None, None],
        n_points_per_sampling_step_idg=5,
        sampled_points=list(range(60)),
    )
    result = InitialDatasetProcedure._get_member_points(stub, 1)
    assert result == list(range(10, 20))


def test_distinct_true_disjoint():
    stub = SimpleNamespace(
        distinct_model_sets=True,
        atoms=[None, None],
        n_points_per_sampling_step_idg=5,
        sampled_points=list(range(60)),
    )
    s0 = set(InitialDatasetProcedure._get_member_points(stub, 0))
    s1 = set(InitialDatasetProcedure._get_member_points(stub, 1))
    s2 = set(InitialDatasetProcedure._get_member_points(stub, 2))
    assert s0.isdisjoint(s1)
    assert s0.isdisjoint(s2)
    assert s1.isdisjoint(s2)


def test_distinct_false_returns_all():
    sampled = list(range(60))
    stub = SimpleNamespace(
        distinct_model_sets=False,
        atoms=[None, None],
        n_points_per_sampling_step_idg=5,
        sampled_points=sampled,
    )
    result = InitialDatasetProcedure._get_member_points(stub, 0)
    assert result == sampled


def test_get_member_points_empty_when_past_end():
    stub = SimpleNamespace(
        distinct_model_sets=True,
        atoms=[None, None],
        n_points_per_sampling_step_idg=5,
        sampled_points=list(range(10)),  # only 10 elements
    )
    # member_number=1: start=10, end=20 — Python slicing returns [] silently
    result = InitialDatasetProcedure._get_member_points(stub, 1)
    assert result == []


# ===========================================================================
# §1.2 — _num_samples_per_traj
# ===========================================================================


def test_distinct_true_multiplies():
    stub = SimpleNamespace(
        distinct_model_sets=True,
        ensemble_size=4,
        n_points_per_sampling_step_idg=10,
    )
    result = InitialDatasetPARSL._num_samples_per_traj(stub)
    assert result == 40


def test_distinct_false_ignores_ensemble():
    stub = SimpleNamespace(
        distinct_model_sets=False,
        ensemble_size=4,
        n_points_per_sampling_step_idg=10,
    )
    result = InitialDatasetPARSL._num_samples_per_traj(stub)
    assert result == 10


# ===========================================================================
# §1.3 — InitialDatasetPARSL._process_reference_result
# ===========================================================================


def test_parsl_none_returns_none():
    stub = SimpleNamespace(compute_stress=False)
    result = InitialDatasetPARSL._process_reference_result(
        stub, None, make_point()
    )
    assert result is None


def test_parsl_energy_forces_set():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    InitialDatasetPARSL._process_reference_result(stub, RESULT, point)
    assert point.info["REF_energy"] == -10.5
    assert "REF_forces" in point.arrays
    assert np.array(point.arrays["REF_forces"]).shape == (1, 3)


def test_parsl_returns_point():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    result = InitialDatasetPARSL._process_reference_result(stub, RESULT, point)
    assert result is point


def test_parsl_stress_set():
    stub = SimpleNamespace(compute_stress=True)
    point = make_point()
    result_dict = RESULT | {"stress": [1.0, 2.0]}
    InitialDatasetPARSL._process_reference_result(stub, result_dict, point)
    assert "REF_stress" in point.info


def test_parsl_stress_not_set():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    InitialDatasetPARSL._process_reference_result(stub, RESULT, point)
    assert "REF_stress" not in point.info


def test_parsl_stress_not_set_when_key_absent():
    # Guard matches teacher variant: stress is skipped if absent from result_dict
    stub = SimpleNamespace(compute_stress=True)
    point = make_point()
    InitialDatasetPARSL._process_reference_result(stub, RESULT, point)
    assert "REF_stress" not in point.info


def test_parsl_hirshfeld_ratios():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    result_dict = RESULT | {"hirshfeld_ratios": [0.9]}
    InitialDatasetPARSL._process_reference_result(stub, result_dict, point)
    assert "REF_hirshfeld_ratios" in point.arrays


def test_parsl_hirshfeld_charges():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    result_dict = RESULT | {"hirshfeld_charges": [0.1]}
    InitialDatasetPARSL._process_reference_result(stub, result_dict, point)
    assert "REF_charges" in point.arrays


def test_parsl_dipole():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    result_dict = RESULT | {"dipole": [0.1, 0.2, 0.3]}
    InitialDatasetPARSL._process_reference_result(stub, result_dict, point)
    assert "REF_dipole" in point.info


# ===========================================================================
# §1.4 — InitialDatasetPARSLTeacher._process_reference_result
# ===========================================================================


def test_teacher_none_returns_none():
    stub = SimpleNamespace(compute_stress=False)
    result = InitialDatasetPARSLTeacher._process_reference_result(
        stub, None, make_point()
    )
    assert result is None


def test_teacher_energy_forces_set():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    InitialDatasetPARSLTeacher._process_reference_result(stub, RESULT, point)
    assert point.info["REF_energy"] == -10.5
    assert "REF_forces" in point.arrays


def test_teacher_stress_set_when_both():
    stub = SimpleNamespace(compute_stress=True)
    point = make_point()
    result_dict = RESULT | {"stress": [1.0]}
    InitialDatasetPARSLTeacher._process_reference_result(
        stub, result_dict, point
    )
    assert "REF_stress" in point.info


def test_teacher_stress_not_set_key_absent():
    stub = SimpleNamespace(compute_stress=True)
    point = make_point()
    InitialDatasetPARSLTeacher._process_reference_result(stub, RESULT, point)
    assert "REF_stress" not in point.info


def test_teacher_hirshfeld_not_set():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    result_dict = RESULT | {
        "hirshfeld_ratios": [0.9],
        "hirshfeld_charges": [0.1],
    }
    InitialDatasetPARSLTeacher._process_reference_result(
        stub, result_dict, point
    )
    assert "REF_hirshfeld_ratios" not in point.arrays
    assert "REF_charges" not in point.arrays


def test_teacher_dipole_set():
    stub = SimpleNamespace(compute_stress=False)
    point = make_point()
    result_dict = RESULT | {"dipole": [0.1, 0.2, 0.3]}
    InitialDatasetPARSLTeacher._process_reference_result(
        stub, result_dict, point
    )
    assert "REF_dipole" in point.info
