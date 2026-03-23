"""
Helper utilities for Atomate2 aims-PAX IDG workflow
"""
from pathlib import Path

from ase import units
from atomate2.ase.md import MDEnsemble
from atomate2.forcefields import MLFF
from pymatgen.io.aims.sets.core import StaticSetGenerator as AimsStaticSetGenerator

from aims_PAX.atomate2 import *
from aims_PAX.atomate2.random import RandomState
from aims_PAX.settings.project import FMSettings, MDSettings, IDGSettings, MiscSettings
from aims_PAX.tools.utilities.utilities import AIMSControlParser


def get_idg_makers_from_settings(
        idg_settings: IDGSettings,
        md_settings: MDSettings,
        misc_settings: MiscSettings) -> tuple[dict[int, AllowedMDMakers], AllowedReferenceMakers]:
    """
    Returns initialized md and reference makers from project and model
    settings.
    """
    # a factory dictionary, mapping the model used to a function that returns a maker
    # for the model
    available_md_makers = {
        ("foundational", "mace-mp"): get_mace_mp_maker,
    }
    # a key to the dictionary
    model_used = (idg_settings.initial_sampling,
                  idg_settings.foundational_model)

    if model_used not in available_md_makers:
        raise NotImplementedError(f"{model_used} MD model"
                                  f"sampling is not currently supported in atomate2 aims-PAX.")
    md_maker = available_md_makers[model_used](
        idg_settings.foundational_model_settings,
        md_settings
    )
    # here only aims static maker is used, can be extended to include teacher reference
    ref_maker = get_aims_maker(misc_settings.path_to_control,
                               idg_settings.species_dir)
    return md_maker, ref_maker


def get_mace_mp_maker(model_settings: FMSettings,
                      md_settings: MDSettings) -> dict[int, AllowedMDMakers]:
    """Create a maker for MD runs."""
    makers = {}
    idx_makers = md_settings.root.keys() if isinstance(md_settings.root, dict) else [1,]
    valid_dynamics = {
        ("nvt", "langevin"): "langevin",
        ("npt", "berendsen"): "berendsen",
        ("npt", "mtk"): "nose-hoover-chain"
    }
    for idx in idx_makers:
        md_setting = md_settings.get_for_index(idx).model_dump()
        ensemble = md_setting.pop("stat_ensemble")
        kwargs = {
            "name": f"mace-mp-{idx}",
            "force_field_name": MLFF.MACE,
            "time_step": md_setting.pop("timestep"),
            "ensemble": getattr(MDEnsemble, ensemble),
            "temperature": md_setting.pop("temperature"),
            "ionic_step_data": ("structure", "energy", "forces", "stress"),
            "store_trajectory": "full",
            "traj_file": "traj.extxyz",
            "traj_file_fmt": "ase"
        }
        if ensemble == "nvt":
            dynamics = md_setting.pop("thermostat")
        else:
            dynamics = md_setting.pop("barostat")
            kwargs["pressure"] = md_setting.pop("pressure") / 1e8    # AseMDMaker asks for pressure in kBar
        kwargs["dynamics"] = valid_dynamics[(ensemble, dynamics)]
        # here only ASE.MolecularDynamics specific settings should be left, we will translate them to
        # ASE internal units; below are translation rules
        if "friction" in md_setting:
            md_setting["friction"] /= units.fs
        if "seed" in md_setting:
            md_setting["rng"] = RandomState(md_setting.pop("seed"))
        # set ASE calculator kwargs
        kwargs["ase_md_kwargs"] = md_setting
        makers[idx] = ForceFieldMDMaker(**kwargs)
    return makers


def get_aims_maker(path_to_control: Path, species_dir: Path) -> AllowedReferenceMakers:
    """Create a maker for reference runs."""
    control_dict = AIMSControlParser()(path_to_control.as_posix())
    control_dict["species_dir"] = species_dir
    return AimsStaticMaker(
        name="static-aims",
        input_set_generator=AimsStaticSetGenerator(
            user_params=control_dict
        )
    )