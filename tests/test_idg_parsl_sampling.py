from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ase.build
import pytest

from aims_PAX.procedures.initial_dataset import (
    InitialDatasetPARSL,
    InitialDatasetPARSLTeacher,
)

# ===========================================================================
# §4.0 — Shared helpers
# ===========================================================================

RESULT = {"energy": -10.5, "forces": [[1.0, 0.0, 0.0]]}


def make_cu_atoms():
    return ase.build.bulk("Cu", "fcc", a=3.6)


class FakeFuture:
    """Immediately-done PARSL AppFuture stand-in."""

    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class SlowFuture(FakeFuture):
    """Becomes done only after `polls_until_done` calls to .done()."""

    def __init__(self, result, polls_until_done=2):
        super().__init__(result)
        self._polls = 0
        self._threshold = polls_until_done

    def done(self):
        self._polls += 1
        return self._polls >= self._threshold


# ===========================================================================
# §4a — DFT _submit_reference_job: 5 tests
# ===========================================================================


def test_dft_submit_args_forwarded(tmp_path):
    stub = SimpleNamespace(
        calc_idx=0,
        calc_dir=Path(tmp_path),
        parsl_func_input={
            "ase_aims_command": "aims.x",
            "properties": ["energy", "forces"],
        },
        aims_settings=[{"xc": "pbe"}],
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_dft_parsl"
    ) as mock_recalc:
        mock_recalc.return_value = FakeFuture(RESULT)
        InitialDatasetPARSL._submit_reference_job(stub, make_cu_atoms(), 0)
    kwargs = mock_recalc.call_args.kwargs
    for key in (
        "positions",
        "species",
        "cell",
        "pbc",
        "aims_settings",
        "directory",
        "ase_aims_command",
    ):
        assert key in kwargs, f"Expected key '{key}' in call kwargs"
    assert kwargs["species"] == ["Cu"]
    assert kwargs["ase_aims_command"] == "aims.x"
    assert kwargs["aims_settings"] == {"xc": "pbe"}
    assert len(kwargs["positions"]) == 1


def test_dft_submit_increments_calc_idx(tmp_path):
    stub = SimpleNamespace(
        calc_idx=0,
        calc_dir=Path(tmp_path),
        parsl_func_input={
            "ase_aims_command": "aims.x",
            "properties": ["energy", "forces"],
        },
        aims_settings=[{"xc": "pbe"}],
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_dft_parsl"
    ) as mock_recalc:
        mock_recalc.return_value = FakeFuture(RESULT)
        InitialDatasetPARSL._submit_reference_job(stub, make_cu_atoms(), 0)
    assert stub.calc_idx == 1


def test_dft_submit_directory_naming(tmp_path):
    stub = SimpleNamespace(
        calc_idx=0,
        calc_dir=Path(tmp_path),
        parsl_func_input={
            "ase_aims_command": "aims.x",
            "properties": ["energy", "forces"],
        },
        aims_settings=[{"xc": "pbe"}],
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_dft_parsl"
    ) as mock_recalc:
        mock_recalc.return_value = FakeFuture(RESULT)
        InitialDatasetPARSL._submit_reference_job(stub, make_cu_atoms(), 0)
    assert mock_recalc.call_args.kwargs["directory"].name == "initial_calc_1"


def test_dft_submit_single_system_settings_idx(tmp_path):
    stub = SimpleNamespace(
        calc_idx=0,
        calc_dir=Path(tmp_path),
        parsl_func_input={
            "ase_aims_command": "aims.x",
            "properties": ["energy", "forces"],
        },
        aims_settings=[{"xc": "pbe"}],
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_dft_parsl"
    ) as mock_recalc:
        mock_recalc.return_value = FakeFuture(RESULT)
        InitialDatasetPARSL._submit_reference_job(stub, make_cu_atoms(), 5)
    assert mock_recalc.call_args.kwargs["aims_settings"] == {"xc": "pbe"}


def test_dft_submit_multi_system_settings_idx(tmp_path):
    s0 = {"xc": "lda"}
    s1 = {"xc": "pbe"}
    stub = SimpleNamespace(
        calc_idx=0,
        calc_dir=Path(tmp_path),
        parsl_func_input={
            "ase_aims_command": "aims.x",
            "properties": ["energy", "forces"],
        },
        aims_settings=[s0, s1],
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_dft_parsl"
    ) as mock_recalc:
        mock_recalc.return_value = FakeFuture(RESULT)
        InitialDatasetPARSL._submit_reference_job(stub, make_cu_atoms(), 1)
    assert mock_recalc.call_args.kwargs["aims_settings"] == s1


# ===========================================================================
# §4b — Teacher _submit_reference_job: 5 tests
# ===========================================================================


def test_teacher_submit_model_type_and_path():
    stub = SimpleNamespace(
        calc_idx=0,
        workqueue_resource_spec=None,
        device="cpu",
        dtype="float32",
        properties=["energy", "forces"],
        teacher_reference_settings={
            "model_type": "mace-mp",
            "model_path": "/models/mp.pt",
            "extra_key": "val",
        },
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_teacher_model_parsl"
    ) as mock:
        mock.return_value = FakeFuture(RESULT)
        InitialDatasetPARSLTeacher._submit_reference_job(
            stub, make_cu_atoms(), 0
        )
    assert mock.call_args.kwargs["model_type"] == "mace-mp"
    assert mock.call_args.kwargs["model_path"] == "/models/mp.pt"
    assert mock.call_args.kwargs["properties"] == stub.properties


def test_teacher_submit_model_settings_excludes_type_and_path():
    stub = SimpleNamespace(
        calc_idx=0,
        workqueue_resource_spec=None,
        device="cpu",
        dtype="float32",
        properties=["energy", "forces"],
        teacher_reference_settings={
            "model_type": "mace-mp",
            "model_path": "/models/mp.pt",
            "extra_key": "val",
        },
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_teacher_model_parsl"
    ) as mock:
        mock.return_value = FakeFuture(RESULT)
        InitialDatasetPARSLTeacher._submit_reference_job(
            stub, make_cu_atoms(), 0
        )
    model_settings = mock.call_args.kwargs["model_settings"]
    assert "model_type" not in model_settings
    assert "model_path" not in model_settings
    assert "extra_key" in model_settings


def test_teacher_submit_device_dtype_defaults():
    stub = SimpleNamespace(
        calc_idx=0,
        workqueue_resource_spec=None,
        device="cpu",
        dtype="float32",
        properties=["energy", "forces"],
        teacher_reference_settings={
            "model_type": "mace-mp",
            "model_path": "/models/mp.pt",
            "extra_key": "val",
        },
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_teacher_model_parsl"
    ) as mock:
        mock.return_value = FakeFuture(RESULT)
        InitialDatasetPARSLTeacher._submit_reference_job(
            stub, make_cu_atoms(), 0
        )
    model_settings = mock.call_args.kwargs["model_settings"]
    assert model_settings["device"] == "cpu"
    assert model_settings["default_dtype"] == "float32"


def test_teacher_submit_device_not_overridden():
    stub = SimpleNamespace(
        calc_idx=0,
        workqueue_resource_spec=None,
        device="cpu",
        dtype="float32",
        properties=["energy", "forces"],
        teacher_reference_settings={
            "model_type": "mace-mp",
            "device": "cuda",
        },
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_teacher_model_parsl"
    ) as mock:
        mock.return_value = FakeFuture(RESULT)
        InitialDatasetPARSLTeacher._submit_reference_job(
            stub, make_cu_atoms(), 0
        )
    assert mock.call_args.kwargs["model_settings"]["device"] == "cuda"


def test_teacher_submit_workqueue_resource_spec():
    spec = {"cores": 4, "memory": 8000}
    stub = SimpleNamespace(
        calc_idx=0,
        workqueue_resource_spec=spec,
        device="cpu",
        dtype="float32",
        properties=["energy", "forces"],
        teacher_reference_settings={"model_type": "mace-mp"},
    )
    with patch(
        "aims_PAX.procedures.initial_dataset.recalc_teacher_model_parsl"
    ) as mock:
        mock.return_value = FakeFuture(RESULT)
        InitialDatasetPARSLTeacher._submit_reference_job(
            stub, make_cu_atoms(), 0
        )
    assert mock.call_args.kwargs["parsl_resource_specification"] == spec


# ===========================================================================
# §4c — Polling loop in _sample_points: 6 tests
# ===========================================================================


def make_poll_stub(sampled_points, future_factory, monkeypatch):
    stub = SimpleNamespace(
        sampled_points=sampled_points,
        clean_dirs=False,
        compute_stress=False,
        calc_dir=None,  # not used when clean_dirs=False
    )
    stub._md_w_foundational = lambda: None
    stub._submit_reference_job = lambda atoms, idx: future_factory(atoms, idx)
    stub._process_reference_result = lambda r, p: (
        None
        if r is None
        else (p.info.update({"REF_energy": r["energy"]}) or p)
    )
    monkeypatch.setattr("time.sleep", lambda t: None)
    return stub


def test_poll_all_succeed(monkeypatch):
    sampled_points = {0: [make_cu_atoms(), make_cu_atoms()]}
    stub = make_poll_stub(
        sampled_points,
        lambda atoms, idx: FakeFuture(RESULT),
        monkeypatch,
    )
    result = InitialDatasetPARSL._sample_points(stub)
    assert len(result) == 2
    assert all(p.info.get("REF_energy") == -10.5 for p in result)


def test_poll_some_fail_returns_none(monkeypatch):
    sampled_points = {0: [make_cu_atoms(), make_cu_atoms()]}
    futures = iter([FakeFuture(RESULT), FakeFuture(None)])
    stub = make_poll_stub(
        sampled_points,
        lambda atoms, idx: next(futures),
        monkeypatch,
    )
    result = InitialDatasetPARSL._sample_points(stub)
    assert len(result) == 1


def test_poll_all_fail(monkeypatch):
    sampled_points = {0: [make_cu_atoms(), make_cu_atoms()]}
    stub = make_poll_stub(
        sampled_points,
        lambda atoms, idx: FakeFuture(None),
        monkeypatch,
    )
    result = InitialDatasetPARSL._sample_points(stub)
    assert len(result) == 0


def test_poll_slow_futures(monkeypatch):
    sampled_points = {0: [make_cu_atoms(), make_cu_atoms()]}
    stub = make_poll_stub(
        sampled_points,
        lambda atoms, idx: SlowFuture(RESULT, polls_until_done=2),
        monkeypatch,
    )
    result = InitialDatasetPARSL._sample_points(stub)
    assert len(result) == 2


def test_poll_multiple_systems(monkeypatch):
    sampled_points = {
        0: [make_cu_atoms(), make_cu_atoms()],
        1: [make_cu_atoms()],
    }
    stub = make_poll_stub(
        sampled_points,
        lambda atoms, idx: FakeFuture(RESULT),
        monkeypatch,
    )
    result = InitialDatasetPARSL._sample_points(stub)
    assert len(result) == 3


def test_poll_clean_dirs_skipped_when_false(monkeypatch):
    sampled_points = {0: [make_cu_atoms()]}
    stub = make_poll_stub(
        sampled_points,
        lambda atoms, idx: FakeFuture(RESULT),
        monkeypatch,
    )
    stub.clean_dirs = False
    rmtree_mock = MagicMock()
    monkeypatch.setattr(
        "aims_PAX.procedures.initial_dataset.shutil.rmtree", rmtree_mock
    )
    InitialDatasetPARSL._sample_points(stub)
    assert rmtree_mock.call_count == 0


def test_poll_clean_dirs_called_when_true(monkeypatch, tmp_path):
    sampled_points = {0: [make_cu_atoms()]}
    stub = make_poll_stub(
        sampled_points,
        lambda atoms, idx: FakeFuture(RESULT),
        monkeypatch,
    )
    stub.clean_dirs = True
    stub.calc_dir = tmp_path
    rmtree_mock = MagicMock()
    monkeypatch.setattr(
        "aims_PAX.procedures.initial_dataset.shutil.rmtree", rmtree_mock
    )
    # Create a matching directory for glob to find
    (tmp_path / "initial_calc_1").mkdir()
    InitialDatasetPARSL._sample_points(stub)
    assert rmtree_mock.call_count == 1


# ===========================================================================
# §4d — Integration test
# ===========================================================================


@pytest.fixture(scope="module")
def local_parsl_config():
    parsl = pytest.importorskip("parsl")
    from parsl import Config
    from parsl.executors import ThreadPoolExecutor as ParslThreadPoolExecutor
    import tempfile

    run_dir = tempfile.mkdtemp(prefix="parsl_test_")
    config = Config(
        executors=[ParslThreadPoolExecutor(label="local", max_threads=2)],
        run_dir=run_dir,
        app_cache=False,
        initialize_logging=False,
        retries=0,
    )
    parsl.load(config)
    yield parsl
    parsl.dfk().cleanup()


@pytest.mark.slow
def test_integration_parsl_local_sample_points(
    local_parsl_config, monkeypatch
):
    from parsl import python_app

    @python_app
    def stub_teacher_app(
        positions,
        species,
        cell,
        pbc,
        model_type,
        model_path,
        model_settings,
        properties,
    ):
        return {"energy": -10.5, "forces": [[1.0, 0.0, 0.0]]}

    monkeypatch.setattr(
        "aims_PAX.procedures.initial_dataset.recalc_teacher_model_parsl",
        stub_teacher_app,
    )

    atoms = make_cu_atoms()
    stub = SimpleNamespace(
        sampled_points={0: [atoms]},
        clean_dirs=False,
        compute_stress=False,
        calc_idx=0,
        workqueue_resource_spec=None,
        device="cpu",
        dtype="float32",
        properties=["energy", "forces"],
        teacher_reference_settings={"model_type": "mace-mp"},
        calc_dir=None,
    )
    stub._md_w_foundational = lambda: None
    stub._submit_reference_job = lambda a, i: (
        InitialDatasetPARSLTeacher._submit_reference_job(stub, a, i)
    )
    stub._process_reference_result = lambda r, p: (
        InitialDatasetPARSLTeacher._process_reference_result(stub, r, p)
    )

    result = InitialDatasetPARSL._sample_points(stub)

    assert len(result) == 1
    assert result[0].info["REF_energy"] == pytest.approx(-10.5)
