"""
Resolution of the per-system inputs (geometry, control, MD settings).

A *system* is one (geometry, control.in, MD settings) combination. Every
per-index source (a geometry dict or folder, a control dict, an MD dict) has
to define the same set of system indices; a single-valued source is
broadcast to all of them. If no source is per-index, there is only system 0.
Without a control file (teacher-model reference) a system has no
control_path.
"""

import logging
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import ase
from pyfhiaims import AimsControl

from aims_PAX.settings.project import MDSettings, check_system_indices
from aims_PAX.tools.utilities.input_utils import read_geometry


@dataclass
class SystemSpec:
    index: int
    atoms: ase.Atoms
    md_settings: dict
    control_path: Path | None = None
    geometry_path: Path | None = None


def resolve_systems(
    path_to_geometry: Path | dict[int, Path],
    path_to_control: Path | dict[int, Path] | None,
    md_settings: MDSettings,
) -> dict[int, SystemSpec]:
    """
    Combines geometry, control and MD sources into one SystemSpec per index.

    Raises:
        ValueError: If the per-index sources disagree on the system indices.
    """
    per_index: dict[str, set[int]] = {}

    if isinstance(path_to_geometry, dict):
        geometries = read_geometry(path_to_geometry)
        geometry_paths = {i: Path(p) for i, p in path_to_geometry.items()}
        per_index["geometry"] = set(geometries)
    elif Path(path_to_geometry).is_dir():
        # read_geometry numbers every directory entry, so entries it
        # cannot read leave gaps in the indices
        geometries = read_geometry(path_to_geometry)
        folder = Path(path_to_geometry)
        geometry_paths = {
            i: folder / name
            for i, name in enumerate(sorted(os.listdir(folder)))
        }
        per_index["geometry"] = set(geometries)
    else:
        geometries = {None: read_geometry(path_to_geometry)[0]}
        geometry_paths = {None: Path(path_to_geometry)}

    if path_to_control is None:
        controls = {None: None}
    elif isinstance(path_to_control, dict):
        controls = {i: Path(p) for i, p in path_to_control.items()}
        per_index["control"] = set(controls)
    else:
        controls = {None: Path(path_to_control)}

    if isinstance(md_settings.root, dict):
        per_index["MD"] = set(md_settings.root)

    check_system_indices(per_index)
    indices = sorted(next(iter(per_index.values()), {0}))
    if list(per_index) == ["MD"] and len(indices) > 1:
        logging.warning(
            f"The MD settings are given per index ({indices}) while the "
            "geometry and control file are not: each MD entry defines a "
            "system, and the initial dataset generation runs one trajectory "
            "per system."
        )

    return {
        i: SystemSpec(
            index=i,
            atoms=geometries.get(i, geometries.get(None)).copy(),
            control_path=controls.get(i, controls.get(None)),
            geometry_path=geometry_paths.get(i, geometry_paths.get(None)),
            md_settings=md_settings.get_for_index(i).model_dump(),
        )
        for i in indices
    }


def log_systems(systems: dict[int, SystemSpec]) -> None:
    """Logs the composition, geometry and control file of each system."""
    for i, system in systems.items():
        control = system.control_path or "none (teacher reference)"
        logging.info(
            f"System {i}: {system.atoms.get_chemical_formula()} from "
            f"{system.geometry_path}, control file: {control}."
        )


def load_aims_settings(
    control_path: Path, species_dir: Path, postprocess_anyway: bool
) -> dict:
    """
    Parses a control.in into the settings for ASE's FHI-aims calculator.

    Args:
        control_path: The control.in to parse.
        species_dir: FHI-aims species defaults directory.
        postprocess_anyway: Whether to add FHI-aims' ``postprocess_anyway``
            (needed to check SCF convergence during active learning).

    Returns:
        The calculator settings.
    """
    aims_control = AimsControl.from_file(control_path)
    aims_settings = aims_control.parameters
    if aims_control.outputs:
        aims_settings["output"] = aims_control.outputs
    aims_settings["compute_forces"] = True
    aims_settings["species_dir"] = species_dir
    if postprocess_anyway:
        aims_settings["postprocess_anyway"] = True
    return aims_settings


def system_for_trajectory(
    trajectory_idx: int, system_ids: Sequence[int]
) -> int:
    """Maps a trajectory to a system id, cycling through the sorted ids."""
    ids = sorted(system_ids)
    return ids[trajectory_idx % len(ids)]


def _aims_float(control_parameters: dict, keyword: str, source) -> float:
    """Reads a number from FHI-aims parameters (Fortran 'd' exponents ok)."""
    value = str(control_parameters[keyword]).strip()
    try:
        return float(value.replace("d", "e").replace("D", "e"))
    except ValueError:
        where = f" in {source}" if source is not None else ""
        raise ValueError(
            f"Cannot read {keyword} '{value}'{where} as a number."
        ) from None


def charge_and_spin(
    control_parameters: dict,
    source: str | Path | None = None,
) -> tuple[float, float | None]:
    """
    Derives the total charge and total spin S from FHI-aims parameters.

    FHI-aims' ``fixed_spin_moment`` is N_up - N_down, so S is half of it
    (the models use 2 * S unpaired electrons). ``spin none`` (or no ``spin``
    keyword) means S = 0. ``spin collinear`` without ``fixed_spin_moment``
    leaves S undetermined (None).

    Args:
        control_parameters: ``AimsControl.parameters`` (values are strings).
        source: Where the parameters come from (for error messages).

    Returns:
        (total_charge, total_spin); total_spin is None if undetermined.

    Raises:
        ValueError: If ``charge`` or ``fixed_spin_moment`` is not a number.
    """
    charge = 0.0
    if "charge" in control_parameters:
        charge = _aims_float(control_parameters, "charge", source)
    spin_mode = str(control_parameters.get("spin", "none")).strip().lower()
    if spin_mode == "none":
        return charge, 0.0
    if "fixed_spin_moment" in control_parameters:
        moment = _aims_float(control_parameters, "fixed_spin_moment", source)
        return charge, moment / 2.0
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


def backfill_charge_spin(
    ensemble_ase_sets: dict[str, dict[str, list[ase.Atoms]]],
    charge_spin: dict[int, tuple[float, float | None]],
    total_charge_key: str,
    total_spin_key: str,
) -> None:
    """
    Handles dataset frames without a total charge, i.e. data written before
    charge/spin were stored (or provided by the user without them).

    With exactly one system with a control file, such frames get that
    system's charge (and spin, if known). With several, the system of a
    frame is unknown, so it only warns. Without any (teacher reference) it
    does nothing. Frames that have the keys are never changed.

    Args:
        ensemble_ase_sets: tag -> {"train": [...], "valid": [...]}.
        charge_spin: (total charge, total spin) per system with a control
            file.
        total_charge_key: Info key of the total charge.
        total_spin_key: Info key of the total spin.
    """
    if not charge_spin:
        return
    frames = {
        id(atoms): atoms
        for sets in ensemble_ase_sets.values()
        for split in sets.values()
        for atoms in split
    }
    lacking = [
        atoms
        for atoms in frames.values()
        if total_charge_key not in atoms.info
    ]
    if not lacking:
        return
    if len(charge_spin) > 1:
        logging.warning(
            f"{len(lacking)} dataset frame(s) have no '{total_charge_key}'/"
            f"'{total_spin_key}' and cannot be assigned to one of the "
            f"{len(charge_spin)} systems; SO3LR/so3krates models treat them "
            "as charge 0 and spin 0."
        )
        return
    ((system_idx, (charge, spin)),) = charge_spin.items()
    for atoms in lacking:
        atoms.info[total_charge_key] = charge
        if spin is not None and total_spin_key not in atoms.info:
            atoms.info[total_spin_key] = spin
    logging.info(
        f"Stored the total charge/spin of system {system_idx} in "
        f"{len(lacking)} dataset frame(s) that had none."
    )


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


def check_systems_for_model(
    systems: dict[int, SystemSpec],
    charge_spin: dict[int, tuple[float, float | None]],
    model_choice: str,
    architecture,
) -> None:
    """
    Checks that the trained model can tell the systems apart.

    Only systems with a control file have a charge/spin (``charge_spin``);
    with fewer than two of them there is nothing to check.

    Raises:
        ValueError: See ``check_systems_distinguishable`` and
            ``model_embeds``.
    """
    if len(charge_spin) < 2:
        return
    check_systems_distinguishable(
        {i: systems[i] for i in charge_spin},
        charge_spin,
        *model_embeds(model_choice, architecture),
    )
