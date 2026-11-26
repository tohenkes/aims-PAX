"""
Resolution of the per-system inputs (geometry, control, MD settings).

A *system* is one (geometry, control.in, MD settings) combination. Every
per-index source (a geometry dict or folder, a control dict, an MD dict) has
to define the same set of system indices; a single-valued source is
broadcast to all of them. If no source is per-index, there is only system 0.
"""

from dataclasses import dataclass
from itertools import combinations
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


def charge_and_spin(
    control_parameters: dict,
) -> tuple[float, float | None]:
    """
    Derives the total charge and total spin S from FHI-aims parameters.

    FHI-aims' ``fixed_spin_moment`` is N_up - N_down, so S is half of it
    (the models use 2 * S unpaired electrons). ``spin none`` (or no ``spin``
    keyword) means S = 0. ``spin collinear`` without ``fixed_spin_moment``
    leaves S undetermined (None).

    Args:
        control_parameters: ``AimsControl.parameters`` (values are strings).

    Returns:
        (total_charge, total_spin); total_spin is None if undetermined.
    """
    charge = float(control_parameters.get("charge", 0.0))
    spin_mode = str(control_parameters.get("spin", "none")).strip().lower()
    if spin_mode == "none":
        return charge, 0.0
    if "fixed_spin_moment" in control_parameters:
        return charge, float(control_parameters["fixed_spin_moment"]) / 2.0
    return charge, None


def annotate_charge_spin(
    atoms: ase.Atoms,
    total_charge: float,
    total_spin: float | None,
    total_charge_key: str,
    total_spin_key: str,
) -> None:
    """Stores charge (and spin, if known) in ``atoms.info`` in place."""
    atoms.info[total_charge_key] = total_charge
    if total_spin is not None:
        atoms.info[total_spin_key] = total_spin


def model_embeds(model_choice: str, architecture) -> tuple[bool, bool]:
    """
    Returns whether the model embeds (total charge, total spin).

    Raises:
        ValueError: If the model choice is unknown.
    """
    if model_choice == "mace":
        return False, False
    if model_choice in ("so3lr", "so3krates"):
        return architecture.use_charge_embed, architecture.use_spin_embed
    raise ValueError(f"Unknown model choice '{model_choice}'.")


def check_systems_distinguishable(
    systems: dict[int, SystemSpec],
    charge_spin: dict[int, tuple[float, float | None]],
    uses_charge_embed: bool,
    uses_spin_embed: bool,
) -> None:
    """
    Checks that systems with the same composition can be told apart.

    Two systems with the same chemical formula but different total charge
    (or spin) would get contradicting labels for the same input unless the
    model embeds that quantity.

    Raises:
        ValueError: If two systems of one composition differ in charge/spin
            in a way the model cannot represent, or if a spin is unknown.
    """
    groups: dict[str, list[int]] = {}
    for idx in sorted(systems):
        formula = systems[idx].atoms.get_chemical_formula()
        groups.setdefault(formula, []).append(idx)

    for formula, indices in groups.items():
        for i, j in combinations(indices, 2):
            where = (
                f"systems {i} and {j} ({formula}; "
                f"{systems[i].control_path} and {systems[j].control_path})"
            )
            (q_i, s_i), (q_j, s_j) = charge_spin[i], charge_spin[j]
            if q_i != q_j and not uses_charge_embed:
                raise ValueError(
                    f"The total charge differs ({q_i} vs {q_j}) between "
                    f"{where}, but the model has no charge embedding; enable "
                    "it or use different compositions."
                )
            if s_i == s_j:
                continue
            if s_i is None or s_j is None:
                raise ValueError(
                    f"The total spin of {where} cannot be told apart: one "
                    "is undetermined. Set fixed_spin_moment in that "
                    "system's control.in."
                )
            if not uses_spin_embed:
                raise ValueError(
                    f"The total spin differs ({s_i} vs {s_j}) between "
                    f"{where}, but the model has no spin embedding; enable "
                    "it or use different compositions."
                )
