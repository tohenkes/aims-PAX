"""Tests for the single SO3LR loader (bundled, settings-file, pickled)."""

import logging
import shutil
from importlib import resources

import numpy as np
import pytest
import torch
import yaml
from ase.build import molecule
from so3krates_torch.calculator.so3 import SO3LRCalculator
from so3krates_torch.tools.multihead_utils import DEFAULT_THEORY_LEVEL

from aims_PAX.settings.model import SO3LRArchitectureSettings
from aims_PAX.tools.model_tools.so3lr_loading import (
    BUNDLED_SO3LR_MODELS,
    load_so3lr,
    make_so3lr_calculator,
    so3lr_architecture_dict,
)

WATER = molecule("H2O")


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """The SO3LR constructor and calculators set the global dtype."""
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


def _energy(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces()


def _model_energy(model):
    calc = make_so3lr_calculator_from_model(model)
    return _energy(calc, WATER)


def make_so3lr_calculator_from_model(model):
    from so3krates_torch.calculator.so3 import TorchkratesCalculator

    return TorchkratesCalculator(
        models=[model],
        device="cpu",
        default_dtype="float32",
        r_max_lr=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
    )


def test_bundled_keywords():
    assert "v1" in BUNDLED_SO3LR_MODELS
    assert "tauri-s-2606" in BUNDLED_SO3LR_MODELS
    assert isinstance(BUNDLED_SO3LR_MODELS, frozenset)


def test_tauri_matches_upstream_calculator():
    # r_max_lr changes the water energy (the bundled model's own is 1000,
    # the upstream calculator defaults to 12), so pin it on both sides.
    ours = make_so3lr_calculator(
        "tauri-s-2606",
        device="cpu",
        dtype="float32",
        theory_level=5,
        r_max_lr=12.0,
    )
    ref = SO3LRCalculator(model="tauri-s-2606", theory_level=5, r_max_lr=12.0)
    e, f = _energy(ours, WATER)
    e_ref, f_ref = _energy(ref, WATER)
    assert e == pytest.approx(e_ref, abs=1e-5)
    np.testing.assert_allclose(f, f_ref, atol=1e-5)


def test_tauri_collapsed_to_one_level():
    m = load_so3lr("tauri-s-2606", "cpu", "float32")
    assert m.atomic_energy_output_block.num_theory_levels == 1


def test_default_theory_level_used(caplog):
    with caplog.at_level(logging.INFO):
        default = load_so3lr("tauri-s-2606", "cpu", "float32")
    assert f"theory_level={DEFAULT_THEORY_LEVEL}" in caplog.text
    explicit = load_so3lr(
        "tauri-s-2606", "cpu", "float32", theory_level=DEFAULT_THEORY_LEVEL
    )
    e_default, f_default = _model_energy(default)
    e_explicit, f_explicit = _model_energy(explicit)
    assert e_default == pytest.approx(e_explicit, abs=1e-6)
    np.testing.assert_allclose(f_default, f_explicit, atol=1e-6)


def test_levels_give_different_energies():
    e5, _ = _model_energy(
        load_so3lr("tauri-s-2606", "cpu", "float32", theory_level=5)
    )
    e0, _ = _model_energy(
        load_so3lr("tauri-s-2606", "cpu", "float32", theory_level=0)
    )
    assert e5 != pytest.approx(e0, abs=1e-4)


def test_theory_level_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        load_so3lr("tauri-s-2606", "cpu", "float32", theory_level=16)


def test_untrained_theory_level_warns(caplog):
    with caplog.at_level(logging.INFO):
        load_so3lr("tauri-s-2606", "cpu", "float32", theory_level=3)
    warnings_ = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings_
    # names the levels that were trained
    assert "[0, 4, 5]" in warnings_[0].getMessage()


def test_trained_theory_level_does_not_warn(caplog):
    with caplog.at_level(logging.INFO):
        load_so3lr("tauri-s-2606", "cpu", "float32", theory_level=5)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_theory_level_on_v1_rejected():
    with pytest.raises(ValueError):
        load_so3lr("v1", "cpu", "float32", theory_level=0)


def test_tauri_license_notice(caplog):
    with caplog.at_level(logging.INFO):
        load_so3lr("tauri-s-2606", "cpu", "float32")
    notices = [r for r in caplog.records if "CC BY-NC 4.0" in r.getMessage()]
    assert len(notices) == 1 and notices[0].levelname == "INFO"


def test_v1_has_no_license_notice(caplog):
    with caplog.at_level(logging.INFO):
        load_so3lr("v1", "cpu", "float32")
    assert "CC BY-NC" not in caplog.text


def test_load_keeps_global_dtype():
    torch.set_default_dtype(torch.float64)
    m = load_so3lr("tauri-s-2606", "cpu", "float64")
    assert torch.get_default_dtype() == torch.float64
    assert next(m.parameters()).dtype == torch.float64


def test_load_float32_keeps_global_float64_default():
    torch.set_default_dtype(torch.float64)
    m = load_so3lr("tauri-s-2606", "cpu", "float32")
    assert torch.get_default_dtype() == torch.float64
    assert next(m.parameters()).dtype == torch.float32


def test_settings_pt_roundtrip(tmp_path):
    model = load_so3lr("tauri-s-2606", "cpu", "float32")
    torch.save(model.state_dict(), tmp_path / "m.pt")
    bundled = resources.files("so3krates_torch.pretrained.so3lr") / "tauri"
    with resources.as_file(
        bundled / "so3lr-tauri-s-2606_settings.yaml"
    ) as src:
        with open(src) as f:
            settings = yaml.safe_load(f)
    settings["ARCHITECTURE"]["num_theory_levels"] = 1
    with open(tmp_path / "m_settings.yaml", "w") as f:
        yaml.safe_dump(settings, f)

    loaded = load_so3lr(str(tmp_path / "m.pt"), "cpu", "float32")
    assert loaded.atomic_energy_output_block.num_theory_levels == 1
    e, f = _model_energy(model)
    e_loaded, f_loaded = _model_energy(loaded)
    assert e_loaded == pytest.approx(e, abs=1e-6)
    np.testing.assert_allclose(f_loaded, f, atol=1e-6)


def test_settings_pt_accepts_overrides(tmp_path):
    shutil.copy(
        resources.files("so3krates_torch.pretrained.so3lr")
        / "v1"
        / "so3lr.pt",
        tmp_path / "m.pt",
    )
    shutil.copy(
        resources.files("so3krates_torch.pretrained.so3lr")
        / "v1"
        / "so3lr_settings.yaml",
        tmp_path / "m_settings.yaml",
    )
    m = load_so3lr(
        str(tmp_path / "m.pt"), "cpu", "float32", dispersion_energy_scale=2.0
    )
    assert m.dispersion_energy_scale == 2.0


def test_pickled_model(tmp_path):
    model = load_so3lr("v1", "cpu", "float32")
    torch.save(model, tmp_path / "m.model")
    loaded = load_so3lr(str(tmp_path / "m.model"), "cpu", "float32")
    e, f = _model_energy(model)
    e_loaded, f_loaded = _model_energy(loaded)
    assert e_loaded == pytest.approx(e, abs=1e-6)
    np.testing.assert_allclose(f_loaded, f, atol=1e-6)


def test_pickled_model_cast_to_requested_dtype(tmp_path):
    torch.save(load_so3lr("v1", "cpu", "float32"), tmp_path / "m.model")
    loaded = load_so3lr(str(tmp_path / "m.model"), "cpu", "float64")
    assert next(loaded.parameters()).dtype == torch.float64


def test_pickled_multilevel_model_is_collapsed(tmp_path):
    from so3krates_torch.calculator.so3 import load_pretrained_so3lr

    torch.save(load_pretrained_so3lr("tauri-s-2606"), tmp_path / "multi.model")
    loaded = load_so3lr(str(tmp_path / "multi.model"), "cpu", "float32")
    assert loaded.atomic_energy_output_block.num_theory_levels == 1


def test_pickled_multihead_returned_as_is(tmp_path):
    from so3krates_torch.tools.multihead_utils import to_multihead_model

    mh = to_multihead_model(
        load_so3lr("v1", "cpu", "float32"), num_output_heads=2, seed=0
    )
    torch.save(mh, tmp_path / "mh.model")
    loaded = load_so3lr(str(tmp_path / "mh.model"), "cpu", "float32")
    assert type(loaded).__name__ == "MultiHeadSO3LR"
    assert loaded.num_output_heads == 2


def test_overrides_rejected_for_pickle(tmp_path):
    torch.save(load_so3lr("v1", "cpu", "float32"), tmp_path / "m.model")
    with pytest.raises(ValueError):
        load_so3lr(str(tmp_path / "m.model"), "cpu", "float32", use_pme=True)


def test_unknown_spec():
    with pytest.raises(FileNotFoundError, match="tauri-s-2606"):
        load_so3lr("tauri-xl", "cpu", "float32")


def test_architecture_dict_tauri():
    d = so3lr_architecture_dict(load_so3lr("tauri-s-2606", "cpu", "float32"))
    assert d["r_max"] == 4.5 and d["use_rms_norm"] is True
    assert d["nhl_repulsion_bool"] is True and d["c6_ratios_bool"] is True
    assert d["legacy_dispersion_bool"] is False
    assert d["cutoff_fn"] == "phys"
    assert d["activation_fn"] == "silu"
    assert d["energy_activation_fn"] == "silu"


def test_architecture_dict_keys_are_settings_fields():
    d = so3lr_architecture_dict(load_so3lr("tauri-s-2606", "cpu", "float32"))
    fields = set(SO3LRArchitectureSettings.model_fields)
    assert set(d) <= fields
    for aims_pax_only in (
        "model_choice",
        "use_multihead_model",
        "num_multihead_heads",
    ):
        assert aims_pax_only not in d
    # PME is off, so the pme_* attributes do not exist
    assert "pme_smearing" not in d
    assert d["use_pme"] is False


def test_architecture_dict_validates_as_settings():
    d = so3lr_architecture_dict(load_so3lr("tauri-s-2606", "cpu", "float32"))
    settings = SO3LRArchitectureSettings(**d)
    assert settings.r_max == 4.5


def test_architecture_override_applied():
    pytest.importorskip("torchpme")
    m = load_so3lr(
        "tauri-s-2606",
        "cpu",
        "float32",
        use_pme=True,
        pme_smearing=1.0,
        pme_mesh_spacing=1.0,
    )
    d = so3lr_architecture_dict(m)
    assert d["use_pme"] is True and d["pme_smearing"] == 1.0


def test_make_calculator_sets_r_max_lr():
    calc = make_so3lr_calculator(
        "v1", device="cpu", dtype="float32", r_max_lr=7.0
    )
    assert calc.models[0].r_max_lr == 7.0


def test_make_calculator_defaults_to_model_r_max_lr():
    ours = make_so3lr_calculator(
        "tauri-s-2606", device="cpu", dtype="float32", theory_level=5
    )
    ref = SO3LRCalculator(
        model="tauri-s-2606", theory_level=5, r_max_lr=1000.0
    )
    e, _ = _energy(ours, WATER)
    e_ref, _ = _energy(ref, WATER)
    assert e == pytest.approx(e_ref, abs=1e-4)


def test_make_calculator_needs_r_max_lr_when_model_has_none():
    with pytest.raises(ValueError, match="r_max_lr"):
        make_so3lr_calculator("v1", device="cpu", dtype="float32")
