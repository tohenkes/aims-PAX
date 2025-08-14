import logging
import numpy as np

SCHEME = {
    "required_global": ["MD", "CLUSTER"],
    "at_least_one_required_global": [
        "ACTIVE_LEARNING",
        "INITIAL_DATASET_GENERATION",
    ],
    "optional_global": ["MISC"],
    "required_al": [
        "species_dir",
        "num_trajectories",
    ],
    "required_idg": ["species_dir", "n_points_per_sampling_step_idg"],
    "required_cluster": [
        "project_name",
        "parsl_options",
        "slurm_str",
        "worker_str",
        "launch_str",
        "calc_dir",
    ],
    "required_parsl_options": [
        "nodes_per_block",
        "init_blocks",
        "min_blocks",
        "max_blocks",
        "label",
    ],
    "idg_least_one_required": {
        "desired_acc",
        "max_initial_set_size",
        "max_initial_epochs",
    },
    "al_least_one_required": [
        "desired_acc",
        "max_MD_steps",
        "max_train_set_size",
    ],
    "optional_idg": {
        "valid_skip": 1,
        "valid_ratio": 0.1,
        "analysis": False,
        "scheduler_initial": True,
        "converge_initial": False,
        "convergence_patience": 50,
        "initial_sampling": "mace-mp0",
        "initial_foundational_size": "small",
        "skip_step_initial": 25,
        "desired_acc": 0.0,
        "desired_acc_scale_idg": 10.0,
        "intermediate_epochs_idg": 5,
        "max_initial_epochs": np.inf,
        "max_initial_set_size": np.inf,
        "ensemble_size": 4,
        "margin": 0.002,
        "max_convergence_epochs": 500,
        "aims_lib_path": None,
        "progress_dft_update": 10,
    },
    "optional_al": {
        "freeze_threshold_dataset": np.inf,
        "c_x": 0.0,
        "analysis": False,
        "margin": 0.002,
        "seeds_tags_dict": None,
        "uncertainty_type": "max_atomic_sd",
        "uncert_not_crossed_limit": 50000,
        "intermol_crossed_limit": 10,
        "intermol_forces_weight": 100,
        "per_epochs_worker": 2,
        "intermediate_epochs_al": 1,
        "valid_skip": 1,
        "valid_ratio": 0.1,
        "converge_al": True,
        "converge_best": True,
        "convergence_patience": 50,
        "skip_step_mlff": 25,
        "ensemble_size": 4,
        "max_convergence_epochs": 500,
        "aims_lib_path": None,
        "desired_acc": 0.0,
        "max_MD_steps": np.inf,
        "max_train_set_size": np.inf,
        "parallel": False,
        "analysis_skip": 50,
    },
    "optional_cluster": {
        "type": "slurm",
        "clean_dirs": True,
    },
    "required_md": [
        "stat_ensemble",
    ],
    "optional_nvt": {
        "thermostat": "Langevin",
        "timestep": 0.5,
        "friction": 0.001,
        "MD_seed": 42,
        "temperature": 300,
    },
    "optional_npt": {
        "barostat": "Berendsen",
        "pressure_au": 3.4e-7,  # 1 bar in atomic units
        "temperature": 300,
        "timestep": 0.5,
        "externalstress": 6.24e-7,  # 1 bar in eV/Angstrom**3
        "ttime": 30.0,
        "pfactor": 100.0,
    },
    "optional_misc": {
        "path_to_control": "./control.in",
        "path_to_geometry": "./geometry.in",
        "dataset_dir": "./data",
        "log_dir": "./logs",
        "create_restart": True,
        "mol_idxs": None,
    },
    "conflicts": {
        "parallel": "CLUSTER",
        "NVT": "NPT",
        "NPT": "NVT",
    },
}

SCHEME_DTYPES = {
    "floats": [
        "valid_ratio",
        "desired_acc",
        "desired_acc_scale_idg",
        "freeze_threshold_dataset",
        "c_x",
        "margin",
        "timesep",
        "friction",
        "temperature",
        "pressure_au",
        "timestep",
        "ttime",
        "pfactor" "externalstress",
    ],
    "ints": [
        "num_trajectories",
        "nodes_per_block",
        "init_blocks",
        "min_blocks",
        "max_blocks",
        "valid_skip",
        "convergence_patience",
        "skip_step_initial",
        "max_initial_epochs",
        "max_initial_set_size",
        "uncert_not_crossed_limit",
        "intermol_crossed_limit",
        "intermol_forces_weight",
        "per_epochs_worker",
        "intermediate_epochs_al",
        "intermediate_epochs_idg",
        "skip_step_mlff",
        "ensemble_size",
        "max_convergence_epochs",
        "max_MD_steps",
        "max_train_set_size",
        "MD_seed",
        "initial_epochs_worker",
        "n_points_per_sampling_step_idg",
        "analysis_skip",
        "progress_dft_update",
    ],
    "strings": [
        "species_dir",
        "project_name",
        "label",
        "slurm_str",
        "worker_str",
        "launch_str",
        "calc_dir",
        "initial_sampling",
        "initial_foundational_size",
        "uncertainty_type",
        "type",
        "stat_ensemble",
        "thermostat",
        "path_to_control",
        "path_to_geometry",
        "dataset_dir",
        "log_dir",
        "barostat",
    ],
    "optional_strings": [  # Fields that can be None or string
        "seeds_tags_dict",
        "aims_lib_path",
    ],
    "bools": [
        "analysis",
        "scheduler_initial",
        "converge_initial",
        "converge_al",
        "converge_best",
        "parallel",
        "create_restart",
        "clean_dirs",
    ],
    "lists": ["mol_idxs"],
    "dicts": ["parsl_options", "MISC"],
}

SCHEME_MACE = {
    "required_global": [
        "GENERAL",
    ],
    "optional_global": ["ARCHITECTURE", "TRAINING", "MISC"],
    "required_general": [
        "name_exp",
    ],
    "optional_general": {
        "checkpoints_dir": "./checkpoints",
        "loss_dir": "./losses",
        "model_dir": "./model",
        "default_dtype": "float32",
        "compute_stress": False,
        "seed": 42,
    },
    "required_architecture": [],
    "optional_architecture": {
        "model": "ScaleShiftMACE",
        "scaling": "rms_forces_scaling",
        "r_max": 5.0,
        "num_channels": 128,
        "max_L": 1,
        "max_ell": 3,
        "radial_type": "bessel",
        "num_radial_basis": 8,
        "num_cutoff_basis": 5,
        "interaction": "RealAgnosticResidualInteractionBlock",
        "num_interactions": 2,
        "correlation": 3,
        "radial_MLP": [64, 64, 64],
        "gate": "silu",
        "MLP_irreps": "16x0e",
        "interaction_first": "RealAgnosticResidualInteractionBlock",
        "compute_avg_num_neighbors": True,
        "atomic_energies": None,
    },
    "required_training": [],
    "optional_training": {
        "batch_size": 5,
        "valid_batch_size": 5,
        "weight_decay": 5.0e-07,
        "lr": 0.01,
        "amsgrad": True,
        "optimizer": "adam",
        "scheduler": "ReduceLROnPlateau",
        "lr_scheduler_gamma": 0.9993,
        "lr_factor": 0.8,
        "scheduler_patience": 5,
        "swa": False,
        "ema": True,
        "ema_decay": 0.99,
        "loss": "weighted",
        "energy_weight": 1.0,
        "forces_weight": 1000.0,
        "virials_weight": 1.0,
        "stress_weight": 1.0,
        "config_type_weights": {"Default": 1.0},
        "clip_grad": 10.0,
    },
    "optional_misc": {
        "device": "cpu",
        "keep_checkpoints": False,
        "restart_latest": False,
        "error_table": "PerAtomMAE",
        "log_level": "INFO",
    },
}

SCHEME_MACE_DTYPES = {
    "strings": [
        "name_exp",
        "checkpoints_dir",
        "loss_dir",
        "model_dir",
        "default_dtype",
        "model",
        "scaling",
        "radial_type",
        "interaction",
        "gate",
        "interaction_first",
        "optimizer",
        "scheduler",
        "loss",
        "device",
        "error_table",
        "log_level",
        "MLP_irreps",
    ],
    "floats": [
        "r_max",
        "weight_decay",
        "lr",
        "lr_scheduler_gamma",
        "lr_factor",
        "ema_decay",
        "energy_weight",
        "forces_weight",
        "virials_weight",
        "stress_weight",
        "clip_grad",
    ],
    "ints": [
        "seed",
        "num_channels",
        "max_L",
        "max_ell",
        "num_radial_basis",
        "num_cutoff_basis",
        "num_interactions",
        "correlation",
        "batch_size",
        "valid_batch_size",
        "scheduler_patience",
    ],
    "bools": [
        "compute_stress",
        "compute_avg_num_neighbors",
        "amsgrad",
        "swa",
        "ema",
        "keep_checkpoints",
        "restart_latest",
    ],
    "lists": [
        "radial_MLP",
    ],
    "dicts": ["config_type_weights"],
    "optional_lists": ["atomic_energies"],
    "optional_strings": [],
}


def check_dtypes(
    settings: dict,
    scheme: dict,
    scheme_dtype: dict,
    scheme_key: str,
) -> dict:
    for k in scheme[scheme_key]:
        if k in scheme_dtype["floats"]:
            if settings[k] is np.inf:
                continue
            try:
                settings[k] = float(settings[k])
            except ValueError:
                raise ValueError(f"The value of {k} must be a float!")
        elif k in scheme_dtype["ints"]:
            if settings[k] is np.inf:
                continue
            try:
                settings[k] = int(settings[k])
            except ValueError:
                raise ValueError(f"The value of `{k}` must be an integer!")
        elif k in scheme_dtype["strings"]:
            assert isinstance(
                settings[k], str
            ), f"The value of `{k}` must be a string!"
        elif k in scheme_dtype["optional_strings"]:
            assert settings[k] is None or isinstance(
                settings[k], str
            ), f"The value of `{k}` must be a string or None!"
        elif k in scheme_dtype["bools"]:
            assert isinstance(
                settings[k], bool
            ), f"The value of `{k}` must be a bool!"
        elif k in scheme_dtype["dicts"]:
            assert isinstance(
                settings[k], dict
            ), f"The value of `{k}` must be a dict!"
        elif k in scheme_dtype["lists"]:
            assert isinstance(
                settings[k], list
            ), f"The value of `{k}` must be a list!"
        elif k in scheme_dtype["optional_lists"]:
            assert settings[k] is None or isinstance(
                settings[k], list
            ), f"The value of `{k}` must be a list or None!"
    return settings


def check_aimsPAX_settings(settings: dict, procedure: str = "full") -> dict:
    """
    Takes the user active learning settings and checks if they are
    valid. If not, it raises an error.
    Replaces missing optional settings with default values.

    Args:
        dict (dict): The active learning settings to check.
        procedure (str): The procedure for which the settings are checked.
            Can be "initial-ds", "al" or "full". Defaults to "full".

    Returns:
        dict: The checked and updated active learning settings.
    """

    for k in settings.keys():
        # check global structure
        required_global = k in SCHEME["required_global"]
        required_at_least_global = k in SCHEME["at_least_one_required_global"]
        optional_global = k in SCHEME["optional_global"]
        check = required_global or required_at_least_global or optional_global
        assert (
            check
        ), f"The keyword `{k}` is not valid in the global input structure!"

    aims_lib_path_provided_idg = False
    aims_lib_path_provided_al = False

    if procedure == "initial-ds" or procedure == "full":
        assert (
            settings.get("INITIAL_DATASET_GENERATION") is not None
        ), "The `INITIAL_DATASET_GENERATION` settings are not provided!"
        idg_settings = settings["INITIAL_DATASET_GENERATION"]

        # check idg structure
        for k, v in idg_settings.items():
            required_idg = k in SCHEME["required_idg"]
            optional_idg = k in SCHEME["optional_idg"]
            at_least_one_idg = k in SCHEME["idg_least_one_required"]
            check = required_idg or optional_idg or at_least_one_idg
            assert (
                check
            ), f"The keyword `{k}` is not valid in the INITIAL_DATASET_GENERATION input structure!"

        # check if all required keywords are present
        for k in SCHEME["required_idg"]:
            if k not in idg_settings:
                raise ValueError(
                    f"The keyword `{k}` is required in the INITIAL_DATASET_GENERATION settings!"
                )
        # check if at least one of the required keywords is present
        if not any(
            k in idg_settings for k in SCHEME["idg_least_one_required"]
        ):
            raise ValueError(
                "At least one of the following keywords is required in the "
                "`INITIAL_DATASET_GENERATION` settings: "
                f"`{SCHEME['idg_least_one_required']}`!"
            )

        # check if optional keys, values are missing and put defaults
        for k in SCHEME["optional_idg"]:
            if k not in idg_settings:
                idg_settings[k] = SCHEME["optional_idg"][k]

        idg_settings = check_dtypes(
            settings=idg_settings,
            scheme_key="required_idg",
            scheme=SCHEME,
            scheme_dtype=SCHEME_DTYPES,
        )
        idg_settings = check_dtypes(
            settings=idg_settings,
            scheme_key="optional_idg",
            scheme=SCHEME,
            scheme_dtype=SCHEME_DTYPES,
        )
        settings["INITIAL_DATASET_GENERATION"] = idg_settings
        aims_lib_path_provided_idg = idg_settings.get("aims_lib_path", False)

    if procedure == "al" or procedure == "full":
        assert (
            settings.get("ACTIVE_LEARNING") is not None
        ), "The `ACTIVE_LEARNING` settings are not provided!"
        al_settings = settings["ACTIVE_LEARNING"]
        # check al structure
        for k, v in al_settings.items():
            required_al = k in SCHEME["required_al"]
            optional_al = k in SCHEME["optional_al"]
            check = required_al or optional_al
            assert (
                check
            ), f"The keyword `{k}` is not valid in the ACTIVE_LEARNING input structure!"
        # check if all required keywords are present
        for k in SCHEME["required_al"]:
            if k not in al_settings:
                raise ValueError(
                    f"The keyword `{k}` is required in the `ACTIVE_LEARNING` settings!"
                )
        # check if at least one of the required keywords is present
        if not any(k in al_settings for k in SCHEME["al_least_one_required"]):
            raise ValueError(
                "At least one of the following keywords is required in the "
                "`ACTIVE_LEARNING` settings: "
                f"`{SCHEME['al_least_one_required']}`!"
            )
        # check if optional keys, values are missing and put defaults
        for k in SCHEME["optional_al"]:
            if k not in al_settings:
                al_settings[k] = SCHEME["optional_al"][k]
        al_settings = check_dtypes(
            settings=al_settings,
            scheme_key="required_al",
            scheme=SCHEME,
            scheme_dtype=SCHEME_DTYPES,
        )
        al_settings = check_dtypes(
            settings=al_settings,
            scheme_key="optional_al",
            scheme=SCHEME,
            scheme_dtype=SCHEME_DTYPES,
        )
        settings["ACTIVE_LEARNING"] = al_settings
        aims_lib_path_provided_al = al_settings.get("aims_lib_path", False)

    cluster_settings = settings.get("CLUSTER", False)
    if cluster_settings:
        # check structure
        for k, v in cluster_settings.items():
            required_cluster = k in SCHEME["required_cluster"]
            optional_cluster = k in SCHEME["optional_cluster"]
            check = required_cluster or optional_cluster
            assert (
                check
            ), f"The keyword `{k}` is not valid in the CLUSTER input structure!"
        # check if all required keywords are present
        for k in SCHEME["required_cluster"]:
            if k not in cluster_settings:
                raise ValueError(
                    f"The keyword `{k}` is required in the `CLUSTER` settings!"
                )
        # check if optional keys, values are missing and put defaults
        for k in SCHEME["optional_cluster"]:
            if k not in cluster_settings:
                cluster_settings[k] = SCHEME["optional_cluster"][k]

        # check required parsl options
        parsl_options = cluster_settings.get("parsl_options", {})
        for k in SCHEME["required_parsl_options"]:
            if k not in parsl_options:
                raise ValueError(
                    f"The keyword `{k}` is required in the `parsl_options` settings!"
                )
        # check parsl options dtypes
        parsl_options = check_dtypes(
            settings=parsl_options,
            scheme_key="required_parsl_options",
            scheme=SCHEME,
            scheme_dtype=SCHEME_DTYPES,
        )
        cluster_settings["parsl_options"] = parsl_options
        # check cluster options dtypes
        cluster_settings = check_dtypes(
            settings=cluster_settings,
            scheme_key="required_cluster",
            scheme=SCHEME,
            scheme_dtype=SCHEME_DTYPES,
        )
        settings["CLUSTER"] = cluster_settings

    if not aims_lib_path_provided_idg and not aims_lib_path_provided_al:
        assert cluster_settings, (
            "The `aims_lib_path` is not provided in the "
            "`INITIAL_DATASET_GENERATION` or `ACTIVE_LEARNING` settings! "
        )

    # check MD settings
    assert (
        settings.get("MD") is not None
    ), "The `MD` settings are not provided! "
    md_settings = settings.get("MD")
    # check that all required MD keywords are present
    for k in SCHEME["required_md"]:
        if k not in md_settings:
            raise ValueError(
                f"The keyword `{k}` is required in the `MD` settings!"
            )

    if md_settings["stat_ensemble"] not in ["NVT", "NPT"]:
        raise ValueError(
            f"The `stat_ensemble` must be either `NVT` or `NPT`, "
            f"but got `{md_settings['stat_ensemble']}`!"
        )
    if md_settings["stat_ensemble"] == "NPT":
        # check if optional npt keys, values are missing and put defaults
        for k in SCHEME["optional_npt"]:
            if k not in md_settings:
                md_settings[k] = SCHEME["optional_npt"][k]

    elif md_settings["stat_ensemble"] == "NVT":
        # check if optional nvt keys, values are missing and put defaults
        for k in SCHEME["optional_nvt"]:
            if k not in md_settings:
                md_settings[k] = SCHEME["optional_nvt"][k]

    else:
        raise ValueError(
            f"The `stat_ensemble` must be either `NVT` or `NPT`, "
            f"but got `{md_settings['stat_ensemble']}`!"
        )
    # check if the optional keywords have values of the correct type
    md_scheme_key = (
        "optional_nvt"
        if md_settings["stat_ensemble"] == "NVT"
        else "optional_npt"
    )
    md_settings = check_dtypes(
        settings=md_settings,
        scheme_key=md_scheme_key,
        scheme=SCHEME,
        scheme_dtype=SCHEME_DTYPES,
    )
    settings["MD"] = md_settings

    misc_settings = settings.get("MISC", {})
    # check if optional misc keys, values are missing and put defaults
    for k in SCHEME["optional_misc"]:
        if k not in misc_settings:
            misc_settings[k] = SCHEME["optional_misc"][k]
            logging.info(
                f"The keyword `{k}` is not provided in the `MISC` settings! "
                f"Setting it to the default value: {SCHEME['optional_misc'][k]}"
            )
    settings["MISC"] = misc_settings

    return settings


def check_MACE_settings(settings: dict) -> dict:
    """
    Validates and fills default values for MACE training settings
    based on SCHEME_MACE and SCHEME_MACE_DTYPES.

    Args:
        settings (dict): The MACE settings to validate.

    Returns:
        dict: The validated and updated settings.
    """
    # Check top-level keys
    for k in settings.keys():
        required_global = k in SCHEME_MACE["required_global"]
        optional_global = k in SCHEME_MACE["optional_global"]
        if not (required_global or optional_global):
            raise ValueError(
                f"The keyword `{k}` is not valid in the global MACE input structure!"
            )

    # Check all required global sections are present
    for k in SCHEME_MACE["required_global"]:
        if k not in settings:
            raise ValueError(
                f"The section `{k}` is required in the MACE settings!"
            )

    # GENERAL block
    general_settings = settings.get("GENERAL", {})
    for k in SCHEME_MACE["required_general"]:
        if k not in general_settings:
            raise ValueError(
                f"The keyword `{k}` is required in the GENERAL settings!"
            )
    for k in SCHEME_MACE["optional_general"]:
        if k not in general_settings:
            general_settings[k] = SCHEME_MACE["optional_general"][k]
    general_settings = check_dtypes(
        settings=general_settings,
        scheme_key="required_general",
        scheme=SCHEME_MACE,
        scheme_dtype=SCHEME_MACE_DTYPES,
    )
    general_settings = check_dtypes(
        settings=general_settings,
        scheme_key="optional_general",
        scheme=SCHEME_MACE,
        scheme_dtype=SCHEME_MACE_DTYPES,
    )
    settings["GENERAL"] = general_settings

    # ARCHITECTURE block
    arch_settings = settings.get("ARCHITECTURE", {})
    for k in SCHEME_MACE["optional_architecture"]:
        if k not in arch_settings:
            arch_settings[k] = SCHEME_MACE["optional_architecture"][k]
    arch_settings = check_dtypes(
        settings=arch_settings,
        scheme_key="required_architecture",
        scheme=SCHEME_MACE,
        scheme_dtype=SCHEME_MACE_DTYPES,
    )
    arch_settings = check_dtypes(
        settings=arch_settings,
        scheme_key="optional_architecture",
        scheme=SCHEME_MACE,
        scheme_dtype=SCHEME_MACE_DTYPES,
    )
    settings["ARCHITECTURE"] = arch_settings

    # TRAINING block
    training_settings = settings.get("TRAINING", {})
    for k in SCHEME_MACE["optional_training"]:
        if k not in training_settings:
            training_settings[k] = SCHEME_MACE["optional_training"][k]
    training_settings = check_dtypes(
        settings=training_settings,
        scheme_key="required_training",
        scheme=SCHEME_MACE,
        scheme_dtype=SCHEME_MACE_DTYPES,
    )
    training_settings = check_dtypes(
        settings=training_settings,
        scheme_key="optional_training",
        scheme=SCHEME_MACE,
        scheme_dtype=SCHEME_MACE_DTYPES,
    )
    settings["TRAINING"] = training_settings

    # MISC block
    misc_settings = settings.get("MISC", {})
    for k in SCHEME_MACE["optional_misc"]:
        if k not in misc_settings:
            misc_settings[k] = SCHEME_MACE["optional_misc"][k]
    misc_settings = check_dtypes(
        settings=misc_settings,
        scheme_key="optional_misc",
        scheme=SCHEME_MACE,
        scheme_dtype=SCHEME_MACE_DTYPES,
    )
    settings["MISC"] = misc_settings

    return settings
