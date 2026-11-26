import numpy as np
import pytest

from aims_PAX.settings.project import MDSettings
from aims_PAX.tools.utilities.systems import (
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


def test_system_for_trajectory_cycles():
    assert [system_for_trajectory(i, [0, 1]) for i in range(4)] == [
        0,
        1,
        0,
        1,
    ]
    assert system_for_trajectory(1, [5, 2]) == 5
