"""
Resolution of the per-system inputs (geometry, control, MD settings).

A *system* is one (geometry, control.in, MD settings) combination. Every
per-index source (a geometry dict or folder, a control dict, an MD dict) has
to define the same set of system indices; a single-valued source is
broadcast to all of them. If no source is per-index, there is only system 0.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import ase

from aims_PAX.settings.project import MDSettings, check_system_indices
from aims_PAX.tools.utilities.input_utils import read_geometry


@dataclass
class SystemSpec:
    index: int
    atoms: ase.Atoms
    control_path: Path
    md_settings: dict


def resolve_systems(
    path_to_geometry: Path | dict[int, Path],
    path_to_control: Path | dict[int, Path],
    md_settings: MDSettings,
) -> dict[int, SystemSpec]:
    """
    Combines geometry, control and MD sources into one SystemSpec per index.

    Raises:
        ValueError: If the per-index sources disagree on the system indices.
    """
    per_index: dict[str, set[int]] = {}

    if isinstance(path_to_geometry, dict) or Path(path_to_geometry).is_dir():
        geometries = read_geometry(path_to_geometry)
        per_index["geometry"] = set(geometries)
    else:
        geometries = {None: read_geometry(path_to_geometry)[0]}

    if isinstance(path_to_control, dict):
        controls = {i: Path(p) for i, p in path_to_control.items()}
        per_index["control"] = set(controls)
    else:
        controls = {None: Path(path_to_control)}

    if isinstance(md_settings.root, dict):
        per_index["MD"] = set(md_settings.root)

    check_system_indices(per_index)
    indices = sorted(next(iter(per_index.values()), {0}))

    return {
        i: SystemSpec(
            index=i,
            atoms=geometries.get(i, geometries.get(None)).copy(),
            control_path=controls.get(i, controls.get(None)),
            md_settings=md_settings.get_for_index(i).model_dump(),
        )
        for i in indices
    }


def system_for_trajectory(
    trajectory_idx: int, system_ids: Sequence[int]
) -> int:
    """Maps a trajectory to a system id, cycling through the sorted ids."""
    ids = sorted(system_ids)
    return ids[trajectory_idx % len(ids)]
