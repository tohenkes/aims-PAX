"""
Phase 4 — CLI smoke tests (updated: ASI4py removed, PARSL-only).

§1  --help exits zero (all 4 entry points, parametrized)
§2  create_initial_ds — strategy dispatch (3 cases) + error on foundational without CLUSTER
§3  al_procedure_only — PARSL dispatch + error without CLUSTER
§4  full-workflow (__main__) — IDG + AL dispatch (2 cases) + error without CLUSTER
§5  recalculate_data — ReCalculatorPARSL wired up with correct kwargs

Convention focus:
  #1 (parametrize variations)
  #3 (assert the dispatched class, not just "ran without error")
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from aims_PAX.cli import __main__ as full_wf_mod
from aims_PAX.cli import al_procedure_only as al_only_mod
from aims_PAX.cli import create_initial_ds as initial_ds_mod
from aims_PAX.cli import recalculate_data as recalc_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(
    initial_sampling="aimd",
    has_cluster=False,
    converge_initial=False,
    use_teacher_reference=False,
    converge_al=False,
):
    """Minimal AimsPAXSettings mock with configurable dispatch attributes."""
    s = MagicMock()
    s.CLUSTER = MagicMock() if has_cluster else None
    # __main__.py uses Pydantic's model_fields_set membership test
    s.model_fields_set = {"CLUSTER"} if has_cluster else set()
    s.INITIAL_DATASET_GENERATION.initial_sampling = initial_sampling
    s.INITIAL_DATASET_GENERATION.converge_initial = converge_initial
    s.INITIAL_DATASET_GENERATION.use_teacher_reference = use_teacher_reference
    s.ACTIVE_LEARNING.converge_al = converge_al
    return s


def _rif_patch(module_path, settings):
    """Patch read_input_files in *module_path* to return the given settings."""
    stub = MagicMock(return_value=(MagicMock(), settings, "ctrl.in", "geo.in"))
    return patch(f"{module_path}.read_input_files", new=stub)


def _done_instance():
    """Procedure instance whose check_*_done() returns True (skips .run())."""
    inst = MagicMock()
    inst.check_initial_ds_done.return_value = True
    inst.check_al_done.return_value = True
    return inst


# ---------------------------------------------------------------------------
# §1 — --help exits zero (all 4 commands)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "main_func",
    [full_wf_mod.main, initial_ds_mod.main, al_only_mod.main, recalc_mod.main],
    ids=["full", "initial-ds", "al-only", "recalc"],
)
def test_help_exits_zero(main_func, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cmd", "--help"])
    with pytest.raises(SystemExit) as exc:
        main_func()
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# §2 — create_initial_ds: strategy dispatch
# ---------------------------------------------------------------------------

_IDG_CASES = [
    ("aimd", False, False, "InitialDatasetAIMD"),
    ("foundational", True, False, "InitialDatasetPARSL"),
    ("foundational", True, True, "InitialDatasetPARSLTeacher"),
]


@pytest.mark.parametrize(
    "sampling,has_cluster,use_teacher,expected_cls",
    _IDG_CASES,
    ids=[c[3] for c in _IDG_CASES],
)
def test_initial_ds_selects_strategy(
    sampling, has_cluster, use_teacher, expected_cls, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cmd", "--model-settings", "m.yaml", "--aimsPAX-settings", "a.yaml"],
    )
    settings = _mock_settings(
        initial_sampling=sampling,
        has_cluster=has_cluster,
        use_teacher_reference=use_teacher,
    )
    MOD = "aims_PAX.cli.create_initial_ds"
    with (
        _rif_patch(MOD, settings),
        patch(f"{MOD}.InitialDatasetAIMD") as MockAIMD,
        patch(f"{MOD}.InitialDatasetPARSL") as MockPARSL,
        patch(f"{MOD}.InitialDatasetPARSLTeacher") as MockTeacher,
    ):
        for cls in (MockAIMD, MockPARSL, MockTeacher):
            cls.return_value = _done_instance()
        initial_ds_mod.main()

    name_to_mock = {
        "InitialDatasetAIMD": MockAIMD,
        "InitialDatasetPARSL": MockPARSL,
        "InitialDatasetPARSLTeacher": MockTeacher,
    }
    name_to_mock[expected_cls].assert_called_once()
    for name, mock in name_to_mock.items():
        if name != expected_cls:
            mock.assert_not_called()


def test_initial_ds_foundational_without_cluster_raises(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cmd", "--model-settings", "m.yaml", "--aimsPAX-settings", "a.yaml"],
    )
    settings = _mock_settings(
        initial_sampling="foundational", has_cluster=False
    )
    MOD = "aims_PAX.cli.create_initial_ds"
    with _rif_patch(MOD, settings):
        with pytest.raises(ValueError, match="CLUSTER"):
            initial_ds_mod.main()


# ---------------------------------------------------------------------------
# §3 — al_procedure_only: PARSL dispatch + error without CLUSTER
# ---------------------------------------------------------------------------


def test_al_only_selects_parsl(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cmd", "--model-settings", "m.yaml", "--aimsPAX-settings", "a.yaml"],
    )
    settings = _mock_settings(has_cluster=True)
    MOD = "aims_PAX.cli.al_procedure_only"
    with (
        _rif_patch(MOD, settings),
        patch(f"{MOD}.ALProcedurePARSL") as MockPARSL,
    ):
        MockPARSL.return_value = _done_instance()
        al_only_mod.main()
    MockPARSL.assert_called_once()


def test_al_only_without_cluster_raises(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cmd", "--model-settings", "m.yaml", "--aimsPAX-settings", "a.yaml"],
    )
    settings = _mock_settings(has_cluster=False)
    MOD = "aims_PAX.cli.al_procedure_only"
    with _rif_patch(MOD, settings):
        with pytest.raises(ValueError, match="CLUSTER"):
            al_only_mod.main()


# ---------------------------------------------------------------------------
# §4 — full-workflow: IDG + AL dispatch
# ---------------------------------------------------------------------------

_FULL_WF_CASES = [
    ("aimd", True, "InitialDatasetAIMD", "ALProcedurePARSL"),
    ("foundational", True, "InitialDatasetPARSL", "ALProcedurePARSL"),
]


@pytest.mark.parametrize(
    "sampling,has_cluster,expected_idg,expected_al",
    _FULL_WF_CASES,
    ids=["aimd-parsl", "foundational-parsl"],
)
def test_full_workflow_dispatches_idg_and_al(
    sampling, has_cluster, expected_idg, expected_al, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cmd", "--model-settings", "m.yaml", "--aimsPAX-settings", "a.yaml"],
    )
    settings = _mock_settings(
        initial_sampling=sampling, has_cluster=has_cluster
    )
    MOD = "aims_PAX.cli.__main__"
    with (
        _rif_patch(MOD, settings),
        patch(f"{MOD}.InitialDatasetAIMD") as MockAIMD,
        patch(f"{MOD}.InitialDatasetPARSL") as MockIDGPARSL,
        patch(f"{MOD}.InitialDatasetPARSLTeacher") as MockTeacher,
        patch(f"{MOD}.ALProcedurePARSL") as MockALPARSL,
    ):
        for cls in (MockAIMD, MockIDGPARSL, MockTeacher, MockALPARSL):
            cls.return_value = _done_instance()
        full_wf_mod.main()

    idg_map = {
        "InitialDatasetAIMD": MockAIMD,
        "InitialDatasetPARSL": MockIDGPARSL,
        "InitialDatasetPARSLTeacher": MockTeacher,
    }
    al_map = {"ALProcedurePARSL": MockALPARSL}
    idg_map[expected_idg].assert_called_once()
    al_map[expected_al].assert_called_once()


def test_full_workflow_without_cluster_raises(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cmd", "--model-settings", "m.yaml", "--aimsPAX-settings", "a.yaml"],
    )
    settings = _mock_settings(has_cluster=False)
    MOD = "aims_PAX.cli.__main__"
    with _rif_patch(MOD, settings):
        with pytest.raises(ValueError, match="CLUSTER"):
            full_wf_mod.main()


# ---------------------------------------------------------------------------
# §5 — recalculate_data: ReCalculatorPARSL wired up
# ---------------------------------------------------------------------------


def test_recalc_constructs_calculator_with_correct_kwargs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cmd",
            "--data",
            "dataset.xyz",
            "--control",
            "ctrl.in",
            "--start_idx",
            "5",
            "--end_idx",
            "20",
        ],
    )
    with patch("aims_PAX.cli.recalculate_data.ReCalculatorPARSL") as MockCalc:
        MockCalc.return_value = MagicMock()
        recalc_mod.main()
    MockCalc.assert_called_once()
    kw = MockCalc.call_args.kwargs
    assert kw["path_to_data"] == "dataset.xyz"
    assert kw["path_to_control"] == "ctrl.in"
    assert kw["start_idx"] == 5
    assert kw["end_idx"] == 20
