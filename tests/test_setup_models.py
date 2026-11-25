import logging
from pathlib import Path
from unittest.mock import patch

import ase.io
import numpy as np
import pytest
import torch
from mace.tools import AtomicNumberTable
from so3krates_torch.modules.models import MultiHeadSO3LR

from aims_PAX.settings import ModelSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.model_tools.setup_so3 import setup_so3krates, setup_so3lr
from aims_PAX.tools.model_tools.so3lr_loading import load_so3lr
from aims_PAX.tools.utilities.utilities import (
    apply_finetuning_settings,
    apply_model_settings,
    compute_average_E0s,
    create_keyspec,
    refresh_architecture_attrs,
    setup_pretrained,
)

from tests.helpers import make_loader

_TEST_DATA = Path(__file__).parent / "test_data"
TRAIN_XYZ = (
    _TEST_DATA / "datasets/initial/training/combined_initial_train_set.xyz"
)


def mace_settings(seed=42):
    return ModelSettings(
        **{
            "GENERAL": {"name_exp": "test", "seed": seed},
            "ARCHITECTURE": {
                "model_choice": "mace",
                "num_channels": 8,
                "num_interactions": 1,
                "max_L": 0,
            },
            "MISC": {"device": "cpu"},
        }
    )


def so3krates_settings(seed=42):
    return ModelSettings(
        **{
            "GENERAL": {"name_exp": "test", "seed": seed},
            "ARCHITECTURE": {
                "model_choice": "so3krates",
                "num_features": 8,
                "num_layers": 1,
                "degrees": [1],
                "num_heads": 1,
                "energy_regression_dim": 8,
            },
            "MISC": {"device": "cpu"},
        }
    )


def so3lr_settings(seed=42):
    return ModelSettings(
        **{
            "GENERAL": {"name_exp": "test", "seed": seed},
            "ARCHITECTURE": {
                "model_choice": "so3lr",
                "num_features": 8,
                "num_layers": 1,
                "degrees": [1],
                "num_heads": 1,
                "energy_regression_dim": 8,
                "zbl_repulsion_bool": False,
                "electrostatic_energy_bool": False,
                "dispersion_energy_bool": False,
            },
            "MISC": {"device": "cpu"},
        }
    )


def load_si_fixtures():
    atoms_list = ase.io.read(TRAIN_XYZ, index=":")
    z_table = AtomicNumberTable([14])
    energies = [a.info["REF_energy"] for a in atoms_list]
    zs = [a.get_atomic_numbers() for a in atoms_list]
    atomic_energies_dict = compute_average_E0s(energies, zs, z_table)
    return atoms_list, z_table, atomic_energies_dict


@pytest.mark.slow
def test_setup_mace_distinct_seeds():
    _, z_table, atomic_energies_dict = load_si_fixtures()
    model_a = setup_mace(mace_settings(seed=1), z_table, atomic_energies_dict)
    model_b = setup_mace(mace_settings(seed=2), z_table, atomic_energies_dict)
    assert isinstance(model_a, torch.nn.Module)
    assert isinstance(model_b, torch.nn.Module)
    params_a = list(model_a.parameters())[0].detach().flatten()
    params_b = list(model_b.parameters())[0].detach().flatten()
    assert not torch.allclose(params_a, params_b)


@pytest.mark.slow
def test_setup_mace_output_shapes():
    atoms_list, z_table, atomic_energies_dict = load_si_fixtures()
    settings = mace_settings()
    model = setup_mace(settings, z_table, atomic_energies_dict)
    loader = make_loader(
        atoms_list, create_keyspec(), z_table, r_max=5.0, batch_size=1
    )
    for batch in loader:
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        break
    assert output["energy"].shape == (1,)
    assert output["forces"].shape[1] == 3


@pytest.mark.slow
def test_setup_so3krates_distinct_seeds():
    _, z_table, atomic_energies_dict = load_si_fixtures()
    model_a = setup_so3krates(
        so3krates_settings(seed=1), atomic_energies_dict, z_table
    )
    model_b = setup_so3krates(
        so3krates_settings(seed=2), atomic_energies_dict, z_table
    )
    assert isinstance(model_a, torch.nn.Module)
    assert isinstance(model_b, torch.nn.Module)
    params_a = list(model_a.parameters())[0].detach().flatten()
    params_b = list(model_b.parameters())[0].detach().flatten()
    assert not torch.allclose(params_a, params_b)


@pytest.mark.slow
def test_setup_so3krates_output_shapes():
    atoms_list, z_table, atomic_energies_dict = load_si_fixtures()
    settings = so3krates_settings()
    model = setup_so3krates(settings, atomic_energies_dict, z_table)
    loader = make_loader(
        atoms_list, create_keyspec(), z_table, r_max=4.5, batch_size=1
    )
    for batch in loader:
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        break
    assert output["energy"].shape == (1,)
    assert output["forces"].shape[1] == 3


@pytest.mark.slow
def test_setup_so3lr_distinct_seeds():
    _, z_table, atomic_energies_dict = load_si_fixtures()
    model_a = setup_so3lr(
        so3lr_settings(seed=1), z_table, atomic_energies_dict
    )
    model_b = setup_so3lr(
        so3lr_settings(seed=2), z_table, atomic_energies_dict
    )
    assert isinstance(model_a, torch.nn.Module)
    assert isinstance(model_b, torch.nn.Module)
    params_a = list(model_a.parameters())[0].detach().flatten()
    params_b = list(model_b.parameters())[0].detach().flatten()
    assert not torch.allclose(params_a, params_b)


@pytest.mark.slow
def test_setup_so3lr_output_shapes():
    atoms_list, z_table, atomic_energies_dict = load_si_fixtures()
    settings = so3lr_settings()
    model = setup_so3lr(settings, z_table, atomic_energies_dict)
    loader = make_loader(
        atoms_list, create_keyspec(), z_table, r_max=4.5, batch_size=1
    )
    for batch in loader:
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        break
    assert output["energy"].shape == (1,)
    assert output["forces"].shape[1] == 3


# -- SO3LR-tauri architecture options ---------------------------------------


def test_tauri_flags_reach_model(so3lr_model_settings, z_table):
    s = so3lr_model_settings(
        ARCHITECTURE={
            "use_rms_norm": True,
            "qk_norm": True,
            "use_residual_scalars": True,
            "final_layer_bias": False,
            "nhl_repulsion_bool": True,
            "dispersion_energy_bool": True,
            "r_max_lr": 6.0,
            "c6_ratios_bool": True,
            "use_simple_hirshfeld": True,
            "legacy_dispersion_bool": False,
        }
    )
    m = setup_so3lr(s, z_table, {1: 0.0, 8: 0.0})
    assert m.use_rms_norm and m.qk_norm and m.use_residual_scalars
    assert m.nhl_repulsion_bool and hasattr(m, "c6_ratios_output_block")
    assert m.legacy_dispersion_bool is False


def test_defaults_unchanged(so3lr_model_settings, z_table):
    m = setup_so3lr(so3lr_model_settings(), z_table, {1: 0.0, 8: 0.0})
    assert not hasattr(m, "c6_ratios_output_block")
    assert hasattr(m, "zbl_repulsion")


# -- pretrained SO3LR as starting point for AL --------------------------------


def _small_multihead(num_heads, z_table):
    from aims_PAX.tools.model_tools.setup_so3 import setup_multihead_so3lr
    from tests.helpers import so3lr_cfg

    s = ModelSettings.model_validate(
        so3lr_cfg(
            zbl_repulsion_bool=False,
            electrostatic_energy_bool=False,
            dispersion_energy_bool=False,
            use_multihead_model=True,
            num_multihead_heads=num_heads,
        )
    )
    return s, setup_multihead_so3lr(s, z_table, {1: 0.0, 8: 0.0})


def test_pretrained_tauri_keyword_builds_multihead(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=4
    )
    m = setup_pretrained(s, z_table=None, atomic_energies_dict={})
    assert isinstance(m, MultiHeadSO3LR) and m.num_output_heads == 4
    assert m.use_rms_norm and m.qk_norm
    assert m.select_heads is True
    assert m.atomic_energy_output_block.energy_shifts.shape == (118,)


def test_pretrained_backbone_unchanged(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2
    )
    m = setup_pretrained(s, None, {})
    ref = load_so3lr("tauri-s-2606", "cpu", s.GENERAL.default_dtype)
    for k, v in ref.state_dict().items():
        if not k.startswith("atomic_energy_output_block"):
            torch.testing.assert_close(m.state_dict()[k], v)
    torch.testing.assert_close(
        m.atomic_energy_output_block.energy_shifts,
        ref.atomic_energy_output_block.energy_shifts,
    )


def test_pretrained_overrides_architecture(pretrained_settings, caplog):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2, r_max=5.0
    )
    with caplog.at_level(logging.WARNING):
        setup_pretrained(s, None, {})
    assert s.ARCHITECTURE.r_max == 4.5
    assert s.ARCHITECTURE.c6_ratios_bool is True
    assert "r_max" in caplog.text
    assert "5.0" in caplog.text and "4.5" in caplog.text


def test_pretrained_sync_silent_for_unset_keys(pretrained_settings, caplog):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2
    )
    with caplog.at_level(logging.WARNING):
        m = setup_pretrained(s, None, {})
    # defaults differ from tauri (e.g. use_rms_norm), but were not set
    assert "differs from the pretrained model" not in caplog.text
    assert s.ARCHITECTURE.use_rms_norm is True
    assert s.ARCHITECTURE.final_layer_bias is False
    # r_max_lr not set by the user: the model's own value is kept
    assert m.r_max_lr == 1000.0 and s.ARCHITECTURE.r_max_lr == 1000.0


def test_user_r_max_lr_wins(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2, r_max_lr=12.0
    )
    m = setup_pretrained(s, None, {})
    assert m.r_max_lr == 12.0 and s.ARCHITECTURE.r_max_lr == 12.0


def test_user_lr_damping_wins(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        dispersion_energy_cutoff_lr_damping=3.0,
    )
    m = setup_pretrained(s, None, {})
    assert m.dispersion_energy_cutoff_lr_damping == 3.0
    assert s.ARCHITECTURE.dispersion_energy_cutoff_lr_damping == 3.0


def test_pretrained_keeps_avg_num_neighbors(
    pretrained_settings, target, caplog
):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2
    )
    with caplog.at_level(logging.INFO):
        apply_model_settings(target, s)
    assert target.update_avg_num_neighbors is False
    assert "update_avg_num_neighbors" in caplog.text


def test_explicit_avg_num_neighbors_respected(pretrained_settings, target):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        update_avg_num_neighbors=True,
    )
    apply_model_settings(target, s)
    assert target.update_avg_num_neighbors is True


def test_avg_num_neighbors_default_without_pretrained(
    so3lr_model_settings, target
):
    apply_model_settings(target, so3lr_model_settings())
    assert target.update_avg_num_neighbors is True


def test_pretrained_theory_level_passed(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        pretrained_theory_level=16,
    )
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        setup_pretrained(s, None, {})


def test_only_explicit_pme_settings_reach_loader(pretrained_settings):
    import aims_PAX.tools.utilities.utilities as utilities

    calls = []

    def fake_load(spec, device, dtype, theory_level=None, **overrides):
        calls.append(overrides)
        raise RuntimeError("stop")

    s_default = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2
    )
    s_pme = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        use_pme=True,
        pme_smearing=1.5,
        r_max=4.5,
    )
    with patch.object(utilities, "load_so3lr", side_effect=fake_load):
        for s in (s_default, s_pme):
            with pytest.raises(RuntimeError, match="stop"):
                setup_pretrained(s, None, {})
    assert calls == [{}, {"use_pme": True, "pme_smearing": 1.5}]


def test_v1_pickled_multihead_still_loads(
    tmp_path, pretrained_settings, z_table
):
    # today's path: a pickled MultiHeadSO3LR file
    _, mh = _small_multihead(3, z_table)
    path = tmp_path / "mh.model"
    torch.save(mh, path)
    s = pretrained_settings(pretrained_model=str(path), num_multihead_heads=3)
    m = setup_pretrained(s, None, {})
    assert isinstance(m, MultiHeadSO3LR) and m.num_output_heads == 3
    assert m.select_heads is True
    for k, v in mh.state_dict().items():
        torch.testing.assert_close(m.state_dict()[k], v)
    # ARCHITECTURE now describes the small pickled model, not the defaults
    assert s.ARCHITECTURE.num_features == 8
    assert s.ARCHITECTURE.degrees == [1]
    assert s.ARCHITECTURE.zbl_repulsion_bool is False


def test_pickled_multihead_head_count_mismatch(
    tmp_path, pretrained_settings, z_table
):
    _, mh = _small_multihead(3, z_table)
    path = tmp_path / "mh.model"
    torch.save(mh, path)
    s = pretrained_settings(pretrained_model=str(path), num_multihead_heads=2)
    with pytest.raises(AssertionError, match="output heads"):
        setup_pretrained(s, None, {})


def test_pickled_model_rejects_explicit_pme(
    tmp_path, pretrained_settings, z_table
):
    _, mh = _small_multihead(3, z_table)
    path = tmp_path / "mh.model"
    torch.save(mh, path)
    s = pretrained_settings(
        pretrained_model=str(path), num_multihead_heads=3, use_pme=True
    )
    with pytest.raises(ValueError, match="pickled"):
        setup_pretrained(s, None, {})


def test_pretrained_weights_keep_meaning(tmp_path, z_table):
    from tests.helpers import so3lr_cfg

    _, mh = _small_multihead(2, z_table)
    path = tmp_path / "weights.pt"
    torch.save(mh.state_dict(), path)
    cfg = so3lr_cfg(
        zbl_repulsion_bool=False,
        electrostatic_energy_bool=False,
        dispersion_energy_bool=False,
        use_multihead_model=True,
        num_multihead_heads=2,
    )
    cfg["TRAINING"] = {"pretrained_weights": str(path)}
    s = ModelSettings.model_validate(cfg)
    m = setup_pretrained(s, z_table, {1: 0.0, 8: 0.0})
    assert isinstance(m, MultiHeadSO3LR) and m.num_output_heads == 2
    for k, v in mh.state_dict().items():
        torch.testing.assert_close(m.state_dict()[k], v)


def test_convert_to_multihead_deprecated(pretrained_settings, caplog):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        perform_finetuning=True,
        convert_to_multihead=True,
    )
    with caplog.at_level(logging.WARNING):
        m = setup_pretrained(s, None, {})
    assert "convert_to_multihead" in caplog.text
    assert "deprecated" in caplog.text
    # converted once by setup_pretrained, not again by setup_finetuning
    assert isinstance(m, MultiHeadSO3LR) and m.num_output_heads == 2


def test_finetuning_passes_c6_and_no_conversion(pretrained_settings):
    import aims_PAX.tools.utilities.utilities as utilities

    s = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        perform_finetuning=True,
        freeze_c6_ratios=False,
    )
    model = object()
    with patch.object(
        utilities, "setup_finetuning", return_value=model
    ) as fake:
        assert apply_finetuning_settings(model, s) is model
    kwargs = fake.call_args.kwargs
    assert kwargs["freeze_c6_ratios"] is False
    assert kwargs["convert_to_multihead"] is False


# -- procedures build graphs with the pretrained model's cutoffs --------------

_GRAPH_KEYS = ("r_max", "r_max_lr", "dispersion_energy_cutoff_lr_damping")
_ALL_ELEMENTS = list(range(1, 119))


def _idg_stub(model_settings):
    """PrepareInitialDatasetProcedure with only what ``_setup_ensemble``
    needs (no geometry, control file or PARSL)."""
    from aims_PAX.procedures.preparation import (
        PrepareInitialDatasetProcedure,
    )

    obj = PrepareInitialDatasetProcedure.__new__(
        PrepareInitialDatasetProcedure
    )
    apply_model_settings(obj, model_settings)
    obj.seeds_tags_dict = {"test-42": 42}
    obj.z_table = AtomicNumberTable(_ALL_ELEMENTS)
    obj.ensemble_atomic_energies_dict = {
        "test-42": {z: 0.0 for z in _ALL_ELEMENTS}
    }
    return obj


def _two_waters(distance_oo):
    """Two water molecules whose O atoms are ``distance_oo`` apart."""
    from ase.build import molecule

    a, b = molecule("H2O"), molecule("H2O")
    b.positions += [distance_oo, 0.0, 0.0]
    atoms = a + b
    atoms.info["REF_energy"] = 0.0
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    return atoms


def _edge_lengths(atoms, r_max, r_max_lr):
    from aims_PAX.tools.utilities.data_handling import create_model_dataset

    (data,) = create_model_dataset(
        data=[atoms],
        seed=42,
        z_table=AtomicNumberTable(_ALL_ELEMENTS),
        r_max=r_max,
        r_max_lr=r_max_lr,
        key_specification=create_keyspec(),
    )
    senders, receivers = data.edge_index
    return (data.positions[receivers] - data.positions[senders]).norm(dim=1)


def test_idg_procedure_uses_pretrained_r_max(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2, r_max=5.0
    )
    idg = _idg_stub(s)
    assert idg.r_max == 5.0 and idg.r_max_lr is None
    idg._setup_ensemble()
    assert idg.r_max == 4.5
    assert idg.r_max_lr == 1000.0
    assert idg.dispersion_energy_cutoff_lr_damping == 2.0


def test_idg_procedure_keeps_user_r_max_lr(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        r_max=5.0,
        r_max_lr=12.0,
    )
    idg = _idg_stub(s)
    idg._setup_ensemble()
    assert idg.r_max == 4.5 and idg.r_max_lr == 12.0
    assert idg.ensemble["test-42"].r_max_lr == 12.0


def test_idg_procedure_graph_uses_pretrained_cutoff(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="tauri-s-2606",
        num_multihead_heads=2,
        r_max=5.0,
        r_max_lr=12.0,
    )
    idg = _idg_stub(s)
    idg._setup_ensemble()
    atoms = _two_waters(distance_oo=4.75)
    lengths = _edge_lengths(atoms, idg.r_max, idg.r_max_lr)
    lengths_user = _edge_lengths(atoms, 5.0, idg.r_max_lr)
    # the O-O pair (4.75 A) is an edge at 5.0 but not at the model's 4.5
    assert lengths.max() < 4.5
    assert len(lengths) < len(lengths_user)
    assert ((lengths_user > 4.5) & (lengths_user < 5.0)).any()


def test_idg_procedure_without_pretrained_unchanged(so3lr_model_settings):
    s = so3lr_model_settings(ARCHITECTURE={"r_max": 5.0, "r_max_lr": 9.0})
    idg = _idg_stub(s)
    before = {key: getattr(idg, key) for key in _GRAPH_KEYS}
    architecture_before = s.ARCHITECTURE.model_dump()
    idg._setup_ensemble()
    assert {key: getattr(idg, key) for key in _GRAPH_KEYS} == before
    assert s.ARCHITECTURE.model_dump() == architecture_before


def test_refresh_is_noop_for_mace(target):
    from tests.helpers import mace_cfg

    s = ModelSettings.model_validate(mace_cfg(r_max=5.5))
    apply_model_settings(target, s)
    before = {key: getattr(target, key) for key in _GRAPH_KEYS}
    refresh_architecture_attrs(target, s)
    assert {key: getattr(target, key) for key in _GRAPH_KEYS} == before
    assert before == {
        "r_max": 5.5,
        "r_max_lr": None,
        "dispersion_energy_cutoff_lr_damping": None,
    }


def test_al_ensemble_uses_pretrained_r_max(
    tmp_path, pretrained_settings, target
):
    """An AL-only run reads model.yaml afresh (user r_max 5.0); the
    models it loads come from the pretrained tauri model (r_max 4.5)."""
    from aims_PAX.procedures.preparation import ALEnsemble

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    idg_model = setup_pretrained(
        pretrained_settings(
            pretrained_model="tauri-s-2606", num_multihead_heads=2
        ),
        None,
        {},
    )
    torch.save(idg_model, model_dir / "test-42.model")

    s = pretrained_settings(
        pretrained_model="tauri-s-2606", num_multihead_heads=2, r_max=5.0
    )
    apply_model_settings(target, s)
    target.model_dir = model_dir
    target.checkpoints_dir = tmp_path / "checkpoints"
    target.restart = False
    target.mol_idxs = None
    al_ensemble = ALEnsemble.__new__(ALEnsemble)
    al_ensemble.config = target
    assert target.r_max == 5.0

    al_ensemble._setup_ensemble()
    assert target.r_max == 4.5 and target.r_max_lr == 1000.0
    assert s.ARCHITECTURE.r_max == 4.5


# -- v1 stores no long-range values: settings and model must still agree ------


def _training_forward(model, model_settings):
    """One training-mode forward pass of a two-head model on two waters,
    with graphs built through aims-PAX's data path from
    ``model_settings.ARCHITECTURE``."""
    from so3krates_torch.tools import torch_geometric

    from aims_PAX.tools.utilities.data_handling import create_model_dataset

    architecture = model_settings.ARCHITECTURE
    torch.set_default_dtype(
        getattr(torch, model_settings.GENERAL.default_dtype)
    )
    dataset = create_model_dataset(
        data=[_two_waters(distance_oo=3.0)] * 2,
        seed=42,
        z_table=AtomicNumberTable(_ALL_ELEMENTS),
        r_max=architecture.r_max,
        r_max_lr=architecture.r_max_lr,
        key_specification=create_keyspec(),
        all_heads=["head_0", "head_1"],
        head_name="head_0",
    )
    batch = next(
        iter(torch_geometric.dataloader.DataLoader(dataset, batch_size=2))
    )
    return model(batch.to_dict(), training=True)


@pytest.mark.parametrize(
    "long_range", [{}, {"r_max_lr": 12.0}], ids=["default", "r_max_lr"]
)
def test_v1_pretrained_long_range_defaults(pretrained_settings, long_range):
    s = pretrained_settings(
        pretrained_model="v1", num_multihead_heads=2, **long_range
    )
    m = setup_pretrained(s, None, {})
    architecture = s.ARCHITECTURE
    assert m.r_max_lr == architecture.r_max_lr == 12.0
    assert (
        m.dispersion_energy_cutoff_lr_damping
        == architecture.dispersion_energy_cutoff_lr_damping
        == 2.0
    )
    out = _training_forward(m, s)
    assert torch.isfinite(out["energy"]).all()


def test_v1_pretrained_user_damping_wins(pretrained_settings):
    s = pretrained_settings(
        pretrained_model="v1",
        num_multihead_heads=2,
        dispersion_energy_cutoff_lr_damping=1.5,
    )
    m = setup_pretrained(s, None, {})
    assert m.dispersion_energy_cutoff_lr_damping == 1.5
    assert s.ARCHITECTURE.dispersion_energy_cutoff_lr_damping == 1.5
    assert m.r_max_lr == s.ARCHITECTURE.r_max_lr == 12.0


# -- AL-only: the loaded models' architecture wins, pretrained_model or not ---


def _al_ensemble(tmp_path, model_settings, target):
    from aims_PAX.procedures.preparation import ALEnsemble

    apply_model_settings(target, model_settings)
    target.model_dir = tmp_path / "model"
    target.checkpoints_dir = tmp_path / "checkpoints"
    target.restart = False
    target.mol_idxs = None
    al_ensemble = ALEnsemble.__new__(ALEnsemble)
    al_ensemble.config = target
    return al_ensemble


def _save_model(tmp_path, model):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    torch.save(model, model_dir / "test-42.model")


def test_al_ensemble_syncs_without_pretrained_model(
    tmp_path, pretrained_settings, target
):
    """A previous run's tauri-derived model in model_dir, model.yaml
    without TRAINING.pretrained_model."""
    _save_model(
        tmp_path,
        setup_pretrained(
            pretrained_settings(
                pretrained_model="tauri-s-2606", num_multihead_heads=2
            ),
            None,
            {},
        ),
    )
    s = pretrained_settings(num_multihead_heads=2, r_max=5.0)
    al_ensemble = _al_ensemble(tmp_path, s, target)
    assert target.use_pretrained_model is False
    assert target.r_max == 5.0 and target.r_max_lr is None

    al_ensemble._setup_ensemble()
    (model,) = al_ensemble.ensemble.values()
    assert target.r_max == s.ARCHITECTURE.r_max == model.r_max == 4.5
    assert target.r_max_lr == s.ARCHITECTURE.r_max_lr == model.r_max_lr
    assert model.r_max_lr == 1000.0


def test_al_ensemble_long_range_fallback_without_pretrained_model(
    tmp_path, pretrained_settings, target
):
    """A v1-derived model that stores no long-range values gets the
    SO3LRCalculator defaults, on the model and in the settings."""
    model = setup_pretrained(
        pretrained_settings(pretrained_model="v1", num_multihead_heads=2),
        None,
        {},
    )
    model.r_max_lr = None
    model.dispersion_energy_cutoff_lr_damping = None
    _save_model(tmp_path, model)
    s = pretrained_settings(num_multihead_heads=2)
    al_ensemble = _al_ensemble(tmp_path, s, target)

    al_ensemble._setup_ensemble()
    (model,) = al_ensemble.ensemble.values()
    assert target.r_max_lr == model.r_max_lr == 12.0
    assert target.dispersion_energy_cutoff_lr_damping == 2.0
    assert model.dispersion_energy_cutoff_lr_damping == 2.0


def _assert_al_sync_noop(tmp_path, model, build_settings, target, caplog):
    _save_model(tmp_path, model)
    s = build_settings()
    al_ensemble = _al_ensemble(tmp_path, s, target)
    architecture_before = s.ARCHITECTURE.model_dump()
    graph_before = {key: getattr(target, key) for key in _GRAPH_KEYS}

    with caplog.at_level(logging.WARNING):
        al_ensemble._setup_ensemble()
    assert s.ARCHITECTURE.model_dump() == architecture_before
    assert {key: getattr(target, key) for key in _GRAPH_KEYS} == graph_before
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    "long_range",
    [
        {},
        {
            "electrostatic_energy_bool": True,
            "dispersion_energy_bool": True,
            "r_max_lr": 9.0,
        },
    ],
    ids=["short_range", "long_range"],
)
def test_al_ensemble_sync_noop_for_from_scratch_so3lr(
    tmp_path, so3lr_model_settings, target, caplog, long_range
):
    def build_settings():
        return so3lr_model_settings(ARCHITECTURE=long_range)

    model = setup_so3lr(
        build_settings(),
        AtomicNumberTable(_ALL_ELEMENTS),
        {z: 0.0 for z in _ALL_ELEMENTS},
    )
    _assert_al_sync_noop(tmp_path, model, build_settings, target, caplog)


def test_al_ensemble_sync_noop_for_mace(tmp_path, target, caplog):
    from tests.helpers import mace_cfg

    def build_settings():
        return ModelSettings.model_validate(mace_cfg())

    model = setup_mace(
        build_settings(), AtomicNumberTable([1, 8]), {1: -13.6, 8: -2040.0}
    )
    _assert_al_sync_noop(tmp_path, model, build_settings, target, caplog)
