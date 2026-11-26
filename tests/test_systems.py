import logging
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from pyfhiaims.control import AimsControl

from aims_PAX.settings.project import MDSettings
from aims_PAX.tools.utilities.systems import (
    SystemSpec,
    annotate_charge_spin,
    backfill_charge_spin,
    charge_and_spin,
    check_systems_distinguishable,
    check_systems_for_model,
    load_aims_settings,
    log_systems,
    model_embeds,
    resolve_systems,
    system_for_trajectory,
)

_NVT = {"stat_ensemble": "NVT", "thermostat": "Langevin", "timestep": 1.0}


def _md_single():
    return MDSettings.model_validate({**_NVT, "temperature": 300})


def _md_dict():
    return MDSettings.model_validate(
        {
            0: {**_NVT, "temperature": 300},
            1: {**_NVT, "temperature": 500},
        }
    )


def _geometry(path, z=0.0):
    path.write_text("atom 0.0 0.0 0.0 H\n" f"atom 0.0 0.0 {0.74 + z} H\n")
    return path


def _control(path):
    path.write_text("xc pbe\n")
    return path


@pytest.fixture
def files(tmp_path):
    geos = [_geometry(tmp_path / f"geo{i}.in", z=0.1 * i) for i in range(3)]
    ctrls = [_control(tmp_path / f"control{i}.in") for i in range(3)]
    return geos, ctrls


def test_single_sources_give_one_system(files):
    geos, ctrls = files
    systems = resolve_systems(geos[0], ctrls[0], _md_single())
    assert list(systems) == [0]
    assert systems[0].index == 0
    assert systems[0].control_path == ctrls[0]
    assert systems[0].md_settings["temperature"] == 300


def test_control_dict_broadcasts_single_geometry(files):
    geos, ctrls = files
    systems = resolve_systems(
        geos[0], {0: ctrls[0], 1: ctrls[1]}, _md_single()
    )
    assert list(systems) == [0, 1]
    assert systems[1].control_path == ctrls[1]
    assert systems[0].atoms is not systems[1].atoms
    assert np.allclose(systems[0].atoms.positions, systems[1].atoms.positions)


def test_md_dict_defines_systems(files):
    geos, ctrls = files
    systems = resolve_systems(geos[0], ctrls[0], _md_dict())
    assert [s.md_settings["temperature"] for s in systems.values()] == [
        300,
        500,
    ]
    assert systems[1].control_path == ctrls[0]


def test_dicts_with_equal_keys(files):
    geos, ctrls = files
    systems = resolve_systems(
        {0: geos[0], 1: geos[1]}, {0: ctrls[0], 1: ctrls[1]}, _md_dict()
    )
    assert list(systems) == [0, 1]
    assert not np.allclose(
        systems[0].atoms.positions, systems[1].atoms.positions
    )


def test_geometry_folder_gives_systems_in_read_order(tmp_path, files):
    _, ctrls = files
    folder = tmp_path / "geoms"
    folder.mkdir()
    for i in range(3):
        _geometry(folder / f"g{i}.in", z=0.1 * i)
    systems = resolve_systems(folder, ctrls[0], _md_single())
    assert list(systems) == [0, 1, 2]
    assert all(s.control_path == ctrls[0] for s in systems.values())


def test_mismatched_indices_raise(files):
    geos, ctrls = files
    with pytest.raises(ValueError, match=r"control.*\{0, 1, 2\}"):
        resolve_systems(
            {0: geos[0], 1: geos[1]},
            {0: ctrls[0], 1: ctrls[1], 2: ctrls[2]},
            _md_single(),
        )


def test_geometry_folder_vs_control_dict_mismatch(tmp_path, files):
    _, ctrls = files
    folder = tmp_path / "geoms"
    folder.mkdir()
    for i in range(3):
        _geometry(folder / f"g{i}.in")
    with pytest.raises(ValueError, match="geometry"):
        resolve_systems(folder, {0: ctrls[0], 1: ctrls[1]}, _md_single())


def test_md_dict_vs_control_dict_mismatch(files):
    geos, ctrls = files
    with pytest.raises(ValueError, match="MD"):
        resolve_systems(
            geos[0],
            {0: ctrls[0], 1: ctrls[1], 2: ctrls[2]},
            _md_dict(),
        )


def test_geometry_folder_gaps_map_to_files(tmp_path, files, caplog):
    _, ctrls = files
    folder = tmp_path / "geoms"
    folder.mkdir()
    _geometry(folder / "a.in")
    (folder / "b_notes.txt").write_text("not a structure\n")
    _geometry(folder / "c.in", z=0.1)
    systems = resolve_systems(folder, ctrls[0], _md_single())
    # unreadable entries keep their index, leaving a gap
    assert list(systems) == [0, 2]
    assert systems[2].geometry_path == folder / "c.in"
    with caplog.at_level(logging.INFO):
        log_systems(systems)
    assert "c.in" in caplog.text


def test_geometry_paths_of_dict_and_single_file(files):
    geos, ctrls = files
    systems = resolve_systems({3: geos[1]}, ctrls[0], _md_single())
    assert systems[3].geometry_path == geos[1]
    systems = resolve_systems(
        geos[0], {0: ctrls[0], 1: ctrls[1]}, _md_single()
    )
    assert systems[1].geometry_path == geos[0]


def test_md_dict_as_only_per_index_source_warns(files, caplog):
    geos, ctrls = files
    with caplog.at_level(logging.WARNING):
        resolve_systems(geos[0], ctrls[0], _md_dict())
    assert "one trajectory per" in caplog.text


def test_md_dict_with_other_per_index_source_does_not_warn(files, caplog):
    geos, ctrls = files
    with caplog.at_level(logging.WARNING):
        resolve_systems(geos[0], {0: ctrls[0], 1: ctrls[1]}, _md_dict())
    assert "one trajectory per" not in caplog.text


def test_no_control_gives_systems_without_control_path(files):
    geos, _ = files
    systems = resolve_systems({0: geos[0], 1: geos[1]}, None, _md_single())
    assert list(systems) == [0, 1]
    assert all(s.control_path is None for s in systems.values())


def test_load_aims_settings(tmp_path):
    control = tmp_path / "control.in"
    control.write_text("xc pbe\nmarker a\noutput hirshfeld\n")
    settings = load_aims_settings(control, tmp_path, postprocess_anyway=False)
    assert settings["marker"] == "a"
    assert settings["compute_forces"] is True
    assert settings["species_dir"] == tmp_path
    assert settings["output"] == ["hirshfeld"]
    assert "postprocess_anyway" not in settings
    settings = load_aims_settings(control, tmp_path, postprocess_anyway=True)
    assert settings["postprocess_anyway"] is True


def test_system_for_trajectory_cycles():
    assert [system_for_trajectory(i, [0, 1]) for i in range(4)] == [
        0,
        1,
        0,
        1,
    ]
    assert system_for_trajectory(1, [5, 2]) == 5


_TEMPLATE = """\
# TEMPLATE FHI-aims header for periodic MgH2 labels on the SO3LR-Tauri reference.
# Append the defaults_2020/tight species blocks of Mg and H (as aims_label.py does).
xc                         pbe0
many_body_dispersion_nl
relativistic               atomic_zora scalar
charge                     0
occupation_type            gaussian 0.01
sc_accuracy_etot           1e-06
sc_accuracy_eev            1E-4
sc_accuracy_rho            1e-05
sc_accuracy_forces         1E-4
sc_iter_limit              1000
compute_forces             .true.
final_forces_cleaned       .true.
 compute_analytical_stress .true. 
k_grid                     2 2 3      
# --- spin: odd-electron cells (47-atom neutral vacancy) ---
spin                       collinear
fixed_spin_moment          1           # N_up - N_down = 1 (doublet)
default_initial_moment     0
# --- even-electron cells (48-atom perfect): replace the spin block by ---
# spin                     none
"""


def test_charge_spin_defaults():
    assert charge_and_spin({}) == (0.0, 0.0)


def test_charge_spin_cation_closed_shell():
    assert charge_and_spin({"charge": "1", "spin": "none"}) == (1.0, 0.0)


def test_charge_spin_triplet_from_fixed_moment():
    assert charge_and_spin(
        {"spin": "collinear", "fixed_spin_moment": "2"}
    ) == (
        0.0,
        1.0,
    )


def test_charge_spin_doublet_from_user_template(tmp_path):
    (tmp_path / "control.in").write_text(_TEMPLATE)
    params = AimsControl.from_file(tmp_path / "control.in").parameters
    assert charge_and_spin(params) == (0.0, 0.5)


def test_charge_spin_fortran_exponents():
    assert charge_and_spin(
        {"charge": "1.d0", "spin": "collinear", "fixed_spin_moment": "1.0D0"}
    ) == (1.0, 0.5)


def test_charge_spin_unreadable_value_names_keyword_and_source():
    with pytest.raises(ValueError, match=r"fixed_spin_moment.*control_x\.in"):
        charge_and_spin(
            {"spin": "collinear", "fixed_spin_moment": "two"},
            source="control_x.in",
        )
    with pytest.raises(ValueError, match="charge"):
        charge_and_spin({"charge": "+1e"})


def test_charge_spin_unconstrained_collinear_is_unknown():
    assert charge_and_spin({"spin": "collinear"}) == (0.0, None)


def test_annotate_sets_charge_and_spin():
    atoms = Atoms("H2")
    annotate_charge_spin(atoms, 1.0, 0.5, "total_charge", "total_spin")
    assert atoms.info["total_charge"] == 1.0
    assert atoms.info["total_spin"] == 0.5


def test_annotate_skips_unknown_spin():
    atoms = Atoms("H2")
    annotate_charge_spin(atoms, 1.0, None, "total_charge", "total_spin")
    assert atoms.info["total_charge"] == 1.0 and "total_spin" not in atoms.info


def test_model_embeds():
    assert model_embeds("mace", None) == (False, False)
    arch = SimpleNamespace(use_charge_embed=True, use_spin_embed=False)
    assert model_embeds("so3lr", arch) == (True, False)
    assert model_embeds("so3krates", arch) == (True, False)


def test_model_embeds_unknown_model_raises():
    with pytest.raises(ValueError, match="foo"):
        model_embeds("foo", None)


def _spec(index, atoms):
    return SystemSpec(
        index=index,
        atoms=atoms,
        control_path=f"control{index}.in",
        md_settings={},
    )


@pytest.fixture
def two_o2():
    return {i: _spec(i, Atoms("O2")) for i in (0, 1)}


@pytest.fixture
def h2o_h3o():
    return {0: _spec(0, Atoms("H2O")), 1: _spec(1, Atoms("H3O"))}


def test_spin_states_ok_with_spin_embed(two_o2):
    check_systems_distinguishable(
        two_o2, {0: (0.0, 0.0), 1: (0.0, 1.0)}, True, True
    )


def test_spin_states_rejected_without_spin_embed(two_o2):
    with pytest.raises(ValueError, match=r"spin.*systems 0 and 1") as exc:
        check_systems_distinguishable(
            two_o2, {0: (0.0, 0.0), 1: (0.0, 1.0)}, True, False
        )
    assert "control0.in" in str(exc.value)
    assert "control1.in" in str(exc.value)


def test_charge_states_rejected_without_charge_embed(two_o2):
    with pytest.raises(ValueError, match=r"charge.*systems 0 and 1"):
        check_systems_distinguishable(
            two_o2, {0: (0.0, 0.0), 1: (1.0, 0.0)}, False, True
        )


def test_charge_states_ok_with_charge_embed(two_o2):
    check_systems_distinguishable(
        two_o2, {0: (0.0, 0.0), 1: (1.0, 0.0)}, True, False
    )


def test_identical_charge_spin_ok_without_embeds(two_o2):
    check_systems_distinguishable(
        two_o2, {0: (0.0, 0.0), 1: (0.0, 0.0)}, False, False
    )


def test_unknown_spin_in_shared_composition_rejected(two_o2):
    with pytest.raises(ValueError, match="fixed_spin_moment"):
        check_systems_distinguishable(
            two_o2, {0: (0.0, None), 1: (0.0, 1.0)}, True, True
        )


def test_different_compositions_never_conflict(h2o_h3o):
    check_systems_distinguishable(
        h2o_h3o, {0: (0.0, 0.0), 1: (1.0, 0.0)}, False, False
    )


def test_check_systems_for_model_rejects_mace_spin_states(two_o2):
    with pytest.raises(ValueError, match="spin"):
        check_systems_for_model(
            two_o2, {0: (0.0, 0.0), 1: (0.0, 1.0)}, "mace", None
        )


def test_check_systems_for_model_uses_architecture(two_o2):
    architecture = SimpleNamespace(use_charge_embed=True, use_spin_embed=True)
    check_systems_for_model(
        two_o2, {0: (0.0, 0.0), 1: (0.0, 1.0)}, "so3lr", architecture
    )


def test_check_systems_for_model_needs_two_systems_with_control(two_o2):
    # one system, or systems without control files (teacher reference),
    # are never checked, so even an unknown model choice passes
    check_systems_for_model(two_o2, {0: (0.0, 0.0)}, "unknown", None)
    check_systems_for_model(two_o2, {}, "unknown", None)


def _ensemble_sets(*frames):
    """Two ensemble members sharing the same frames."""
    return {
        tag: {"train": list(frames[:-1]), "valid": [frames[-1]]}
        for tag in ("a", "b")
    }


def _labelled(spin):
    atoms = Atoms("O2")
    atoms.info["total_charge"] = 0.0
    atoms.info["total_spin"] = spin
    return atoms


def test_backfill_single_system(caplog):
    old_1, old_2, new = Atoms("O2"), Atoms("O2"), _labelled(0.0)
    with caplog.at_level(logging.INFO):
        backfill_charge_spin(
            _ensemble_sets(old_1, new, old_2),
            {0: (1.0, 1.0)},
            "total_charge",
            "total_spin",
        )
    assert old_1.info == {"total_charge": 1.0, "total_spin": 1.0}
    assert old_2.info == {"total_charge": 1.0, "total_spin": 1.0}
    assert new.info == {"total_charge": 0.0, "total_spin": 0.0}
    # frames shared by the ensemble members are counted once
    assert "2 dataset frame(s)" in caplog.text


def test_backfill_single_system_with_unknown_spin():
    old = Atoms("O2")
    backfill_charge_spin(
        _ensemble_sets(old, Atoms("O2")),
        {4: (0.0, None)},
        "total_charge",
        "total_spin",
    )
    assert old.info == {"total_charge": 0.0}


def test_backfill_several_systems_only_warns(caplog):
    old, new = Atoms("O2"), _labelled(1.0)
    with caplog.at_level(logging.WARNING):
        backfill_charge_spin(
            _ensemble_sets(old, new),
            {0: (0.0, 0.0), 1: (0.0, 1.0)},
            "total_charge",
            "total_spin",
        )
    assert old.info == {}
    assert new.info == {"total_charge": 0.0, "total_spin": 1.0}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "1 dataset frame(s)" in warnings[0].getMessage()
    assert "charge 0" in warnings[0].getMessage()


def test_backfill_without_control_files_does_nothing(caplog):
    old = Atoms("O2")
    with caplog.at_level(logging.INFO):
        backfill_charge_spin(
            _ensemble_sets(old, Atoms("O2")), {}, "total_charge", "total_spin"
        )
    assert old.info == {}
    assert caplog.text == ""
