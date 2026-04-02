import os
from parsl.config import Config
from parsl.executors import WorkQueueExecutor, MPIExecutor
from parsl.executors import ThreadPoolExecutor as ParslThreadPoolExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher
from parsl import python_app
import numpy as np
import re
import logging
from ase.io import ParseError
from pathlib import Path
from typing import Optional


def prepare_parsl(
    cluster_settings: dict = None, output_dir: Path = Path(".")
) -> dict:
    """
    Prepare the PARSL configuration and settings for running calculations.
    Creates the config file, sets the calculation directory, defines if
    the calculation directories should be cleaned, and initializes
    the calculation index (is used to keep track of each calculations
    when creating the directory names). Also, checks if the launch string
    is provided in the cluster settings. The lauch string is used to run
    the DFT calculations on the cluster.

    Args:
        cluster_settings (dict, optional): _description_. Defaults to None.
        output_dir (Path, optional): Base output directory. Relative paths
            for calc_dir and parsl_info_dir are resolved against this.
            Defaults to Path(".").

    Raises:
        KeyError: Launch string not found in YAML file.

    Returns:
        dict: A dictionary containing the PARSL configuration,
              calculation directory, whether to clean directories,
              launch string, and calculation index.
    """
    assert cluster_settings is not None, (
        "Cluster settings not found. Please provide a YAML file "
        "with the cluster settings."
    )
    launch_str = cluster_settings.get("launch_str", "")

    config = create_parsl_config(
        cluster_settings=cluster_settings, output_dir=output_dir
    )
    # get the path to the directory where the calculations will be run
    # if none is provided, place it under output_dir
    _calc_dir = cluster_settings.get("calc_dir", None)
    if _calc_dir is None:
        calc_dir = output_dir / "ase_aims_calcs"
    else:
        calc_dir = Path(_calc_dir)
        if not calc_dir.is_absolute():
            calc_dir = output_dir / calc_dir
    clean_dirs = cluster_settings.get("clean_dirs", True)
    calc_idx = 0

    parsl_options = cluster_settings.get("parsl_options", {})
    _parsl_info = parsl_options.get("parsl_info_dir", None)
    if _parsl_info is None:
        parsl_info_dir = output_dir / "parsl_info"
    else:
        parsl_info_dir = Path(_parsl_info)
        if not parsl_info_dir.is_absolute():
            parsl_info_dir = output_dir / parsl_info_dir
    clean_parsl_dirs = parsl_options.get("clean_parsl_dirs", True)
    clean_task_dirs = parsl_options.get("clean_task_dirs", True)

    return {
        "config": config,
        "calc_dir": calc_dir,
        "clean_dirs": clean_dirs,
        "launch_str": launch_str,
        "calc_idx": calc_idx,
        "parsl_info_dir": parsl_info_dir,
        "clean_parsl_dirs": clean_parsl_dirs,
        "clean_task_dirs": clean_task_dirs,
    }


def create_parsl_config(
    cluster_settings: dict, output_dir: Path = Path(".")
) -> Config:
    """
    Reads in CLUSTER settings as a dict (provided in the yaml file).
    The information is then used to create a PARSL configuration object.
    The configuration includes the executor, provider, and other settings
    necessary for running calculations on the cluster.

    One block is capabale of running PARSL apps like the DFT calculations
    we are doing. PARSL will automatically scale the number of blocks
    based on the number of tasks submitted until the maximum number of
    blocks.

    The slurm_str includes all the info for the slurm scheduler, such as
    the partition, time limit etc. The worker_str is used to initialize
    modules or set environment variables on the worker nodes.

    Args:
        cluster_settings (dict): A dictionary containing parsl options.
        The dictionary should contain the following:
        - "project_name": Name of the project (default: "fhi_aims_dft")
        - "parsl_options": A dictionary with the following keys:
            - "nodes_per_block": Number of nodes per block (default: 1)
            - "init_blocks": Initial number of blocks (default: 1)
            - "min_blocks": Minimum number of blocks (default: 1)
            - "max_blocks": Maximum number of blocks (default: 1)
            - "parsl_info_dir": Directory for PARSL info
                        (default: "./parsl_info")
            - "run_dir": Directory for running calculations
                        (default: "./parsl_info/run_dir")
            - "function_dir": Directory for function files
                        (default: "./parsl_info/function_dir")
            - "port": Port for the PARSL server
                        (default: automatically chosen free port)
            - "label": Label for the executor (default: "workqueue")
        - "worker_str": Worker initialization string (required)
        - "slurm_str": Slurm options string (required)

    Raises:
        KeyError: Worker initialization string not found in YAML file.
        KeyError: Slurm options string not found in YAML file.
        KeyError: Partition not found in slurm options string.

    Returns:
        Config: A PARSL configuration object with the specified settings.
    """

    executor = cluster_settings.get("executor", "workqueue")

    if executor == "local":
        parsl_info_dir = output_dir / "parsl_info"
        if not os.path.exists(parsl_info_dir):
            os.makedirs(parsl_info_dir)
        run_dir = parsl_info_dir / "run_dir"
        config = Config(
            executors=[
                ParslThreadPoolExecutor(
                    label="local",
                    max_threads=cluster_settings.get("max_workers", 4),
                )
            ],
            run_dir=str(run_dir),
            app_cache=False,
            initialize_logging=False,
            retries=0,
        )
        return config

    parsl_options = cluster_settings["parsl_options"]

    nodes_per_block = parsl_options.get("nodes_per_block", 1)
    init_blocks = parsl_options.get("init_blocks", 1)
    min_blocks = parsl_options.get("min_blocks", 1)
    max_blocks = parsl_options.get("max_blocks", 1)
    _parsl_info = parsl_options.get("parsl_info_dir", None)
    if _parsl_info is None:
        parsl_info_dir = output_dir / "parsl_info"
    else:
        parsl_info_dir = Path(_parsl_info)
        if not parsl_info_dir.is_absolute():
            parsl_info_dir = output_dir / parsl_info_dir

    # create a parent folder for all the parsl stuff
    if not os.path.exists(parsl_info_dir):
        os.makedirs(parsl_info_dir)
    run_dir = parsl_options.get("run_dir", parsl_info_dir / "run_dir")
    function_dir = parsl_options.get(
        "function_dir", parsl_info_dir / "function_dir"
    )

    label = parsl_options.get("label", "workqueue")
    cmd_timeout = parsl_options.get("cmd_timeout", 10)
    full_debug = parsl_options.get("full_debug", False)

    try:
        worker_init_str = cluster_settings["worker_str"]
    except KeyError:
        exception_msg = "Worker init string not found in YAML file. Closing."
        raise KeyError(exception_msg)

    try:
        slurm_options_str = cluster_settings["slurm_str"]
    except KeyError:
        exception_msg = "Slurm options string not found in YAML file. Closing."
        raise KeyError(exception_msg)

    # Extract the cluster partition from the slurm options string
    match = re.search(
        r"(?:partition\s*=\s*|(?:^|\s)-p\s*(?:=\s*)?)([\w\-]+)",
        slurm_options_str,
        re.IGNORECASE,
    )
    if match:
        partition = match.group(1)
    else:
        logging.error("Partition not found. Closing.")
        raise KeyError("Partition not found in slurm options string.")

    if executor == "workqueue":
        config = Config(
            executors=[
                WorkQueueExecutor(
                    label=label,
                    port=0,
                    shared_fs=True,  # assumes shared file system
                    function_dir=str(function_dir),
                    autocategory=False,
                    full_debug=full_debug,
                    provider=SlurmProvider(
                        partition=partition,
                        nodes_per_block=nodes_per_block,
                        init_blocks=init_blocks,
                        min_blocks=min_blocks,
                        max_blocks=max_blocks,
                        scheduler_options=slurm_options_str,
                        worker_init=worker_init_str,
                        cmd_timeout=cmd_timeout,
                    ),
                )
            ],
            run_dir=str(run_dir),
            app_cache=False,
            initialize_logging=False,
            retries=3,
        )
    elif executor == "mpi":
        config = Config(
            executors=[
                MPIExecutor(
                    label=label,
                    mpi_launcher="srun",
                    max_workers_per_block=1,
                    provider=SlurmProvider(
                        partition=partition,
                        nodes_per_block=nodes_per_block,
                        init_blocks=init_blocks,
                        min_blocks=min_blocks,
                        max_blocks=max_blocks,
                        scheduler_options=slurm_options_str,
                        worker_init=worker_init_str,
                        launcher=SimpleLauncher(),
                        cmd_timeout=cmd_timeout,
                    ),
                )
            ],
            run_dir=str(run_dir),
            app_cache=False,
            initialize_logging=False,
            retries=3,
        )
    return config


def handle_parsl_logger(log_dir: Path = Path("./")):
    """
    Just sets up the logger for PARSL events
    """
    parsl_logger = logging.getLogger("parsl")
    parsl_handler = logging.FileHandler(log_dir)
    parsl_logger.handlers.clear()
    parsl_logger.addHandler(parsl_handler)
    parsl_logger.propagate = False


def cleanup_task_dir(future) -> None:
    """
    Remove the per-task function_data directory for a completed Parsl task.
    Called immediately after future.result() is retrieved. Silently skips
    if the path cannot be determined (e.g. local executor, non-WorkQueue).
    """
    import shutil
    import parsl

    try:
        # parsl_executor_task_id is set on the inner exec_fu, not on AppFuture
        task_record = getattr(future, "task_record", None)
        if task_record is None:
            return
        exec_fu = task_record.get("exec_fu")
        if exec_fu is None:
            return
        task_id = getattr(exec_fu, "parsl_executor_task_id", None)
        if task_id is None:
            return
        for executor in parsl.dfk().executors.values():
            fdd = getattr(executor, "function_data_dir", None)
            if fdd is None:
                continue
            task_dir = Path(fdd) / "{:04d}".format(task_id)
            if task_dir.exists():
                shutil.rmtree(task_dir)
                logging.debug(f"Removed Parsl task dir: {task_dir}")
    except Exception:
        pass  # Non-critical; never crash the workflow


def cleanup_parsl_dirs(parsl_info_dir: Path) -> None:
    """
    Remove Parsl run_dir and function_dir under parsl_info_dir.
    Called after parsl.dfk().cleanup() when clean_parsl_dirs is True.
    """
    import shutil

    for subdir in ("run_dir", "function_dir"):
        d = parsl_info_dir / subdir
        if d.exists():
            shutil.rmtree(d)
            logging.debug(f"Removed Parsl directory: {d}")


@python_app
def recalc_dft_parsl(
    positions: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
    pbc: bool,
    aims_settings: dict,
    ase_aims_command: str,
    directory: str = "./",
    properties: list = ["energy", "forces"],
    aims_output_file: str = "aims.out",
    **kwargs,
):
    """
    PARSL app that runs the DFT calculations using ASE AIMS calculator.
    The function needs all necessary modules as the PARSL worker is
    running completely independently of the main process.

    Args:
        positions (np.ndarray): Geometry of the system.
        species (np.ndarray): Elements in the system.
        cell (np.ndarray): Unit cell of the system.
        pbc (bool): Periodic boundary conditions.
        aims_settings (dict): FHI aims control settings.
        ase_aims_command (str, optional): Which exact command is used
                                        to run the ASE AIMS calculator.
        directory (str, optional): Where to run the calculation.
                                        Defaults to "./".
        properties (list, optional): Which properties to calculate.
                                    Defaults to ["energy", "forces"].

    Returns:
        dict: Results of the DFT calculation.
    """
    from ase.calculators.aims import Aims, AimsProfile
    from aims_PAX.tools.utilities.utilities import get_free_vols
    import os
    from ase import Atoms
    from ase.io import ParseError

    # Convert inputs to ASE Atoms object
    atoms = Atoms(
        positions=positions,
        symbols=species,
        cell=cell,
        pbc=pbc,
    )
    # create output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    prefix = os.environ.get("PARSL_MPI_PREFIX")

    if prefix is not None:
        command = prefix + " " + ase_aims_command
    else:
        command = ase_aims_command

    os.environ["ASE_AIMS_COMMAND"] = command
    os.environ["PARSL_RESOURCE_SPECIFICATION"] = str(kwargs.get("resource_specification", ""))

    for var in ["UCX_NUM_EPS",]:
        os.environ.pop(var, None)

    calc = Aims(
        profile=AimsProfile(command=os.environ["ASE_AIMS_COMMAND"]),
        directory=directory,
        **aims_settings,
    )
    try:
        calc.calculate(atoms=atoms, properties=properties, system_changes=None)
        results = calc.results
        if "hirshfeld_volumes" in results.keys():
            with open(os.path.join(directory, aims_output_file), "r") as f:
                aims_output = f.readlines()
            free_vols = get_free_vols(aims_output)
            hirshfeld_vols = results["hirshfeld_volumes"]
            hirshfeld_ratios = hirshfeld_vols / free_vols
            results["hirshfeld_ratios"] = hirshfeld_ratios
        return calc.results
    except ParseError as pe:
        return None
    except Exception as e:
        # Handle any other calculation failures (including subprocess errors)
        # This catches cases where the DFT calculation fails due to
        # unphysical structures
        import logging

        logging.warning(
            f"DFT calculation failed in directory {directory}: {str(e)}"
        )
        return None


@python_app
def recalc_teacher_model_parsl(
    positions: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
    pbc: bool,
    model_type: str,
    model_path: str = None,
    model_settings: dict = None,
    properties: list = ["energy", "forces"],
    **kwargs,
):
    """
    PARSL app that runs a teacher model for reference calculations.
    The calculator is constructed inside the app to avoid PyTorch
    pickling issues. The function needs all necessary modules as
    the PARSL worker is running completely independently of the
    main process.

    Args:
        positions (np.ndarray): Geometry of the system.
        species (np.ndarray): Elements in the system.
        cell (np.ndarray): Unit cell of the system.
        pbc (bool): Periodic boundary conditions.
        model_type (str): Type of the teacher model.
            One of "mace-mp", "mace", "so3lr", "so3krates".
        model_path (str, optional): Path to a trained model file.
            Required for "mace", "so3lr", "so3krates".
        model_settings (dict, optional): Additional model settings
            (e.g. mace_model size, device, dtype, etc.).
        properties (list, optional): Which properties to calculate.
                                    Defaults to ["energy", "forces"].
    Returns:
        dict: Results of the teacher model calculation, or None on failure.
    """
    from ase import Atoms
    import logging

    if model_settings is None:
        model_settings = {}

    atoms = Atoms(
        positions=positions,
        symbols=species,
        cell=cell,
        pbc=pbc,
    )

    device = model_settings.get("device", "cpu")
    default_dtype = model_settings.get("default_dtype", "float64")
    compute_stress = "stress" in properties

    try:
        if model_type == "mace-mp":
            from mace.calculators import mace_mp

            mace_model = model_settings.get("mace_model", "small")
            dispersion = model_settings.get("dispersion", False)
            damping = model_settings.get("damping", "bj")
            dispersion_xc = model_settings.get("dispersion_xc", "pbe")
            dispersion_cutoff = model_settings.get("dispersion_cutoff", 12.0)
            calc = mace_mp(
                model=mace_model,
                dispersion=dispersion,
                default_dtype=default_dtype,
                device=device,
                damping=damping,
                dispersion_xc=dispersion_xc,
                dispersion_cutoff=dispersion_cutoff,
            )
        elif model_type == "mace":
            from mace.calculators import MACECalculator

            calc = MACECalculator(
                model_paths=model_path,
                device=device,
                default_dtype=default_dtype,
            )
        elif model_type in ["so3lr", "so3krates"]:

            r_max_lr = model_settings.get("r_max_lr", None)
            dispersion_lr_damping = model_settings.get(
                "dispersion_lr_damping", None
            )

            from so3krates_torch.calculator.so3 import TorchkratesCalculator

            calc = TorchkratesCalculator(
                model_paths=model_path,
                compute_stress=compute_stress,
                device=device,
                default_dtype=default_dtype,
                r_max_lr=r_max_lr,
                dispersion_lr_damping=dispersion_lr_damping,
            )
        else:
            raise ValueError(
                f"Unknown teacher model type: {model_type}. "
                "Supported types: 'mace-mp', 'mace', 'so3lr', 'so3krates'."
            )

        calc.calculate(atoms=atoms, properties=properties)
        return calc.results
    except Exception as e:
        logging.warning(f"Teacher model calculation failed: {str(e)}")
        return None


@python_app
def evaluate_batch_parsl(
    model_path: str,
    hdf5_path: str,
    batch_indices: list,
    threshold: float,
    r_max: float,
    r_max_lr,
    error_type: str,
    energy_key: str,
    forces_key: str,
    z_table_zs: list,
    head_index: int = 0,
    multihead: bool = False,
    eval_batch_size: int = 32,
    device: str = "cpu",
    hdf5_paths=None,
    batch_items=None,
    **kwargs,
):
    """
    PARSL app that evaluates a batch of structures from a raw HDF5 dataset.
    Loads the model, loads structures by index, computes prediction error
    vs reference labels. Returns indices that exceed the threshold.

    Structures are processed in mini-batches via a DataLoader for
    efficiency. Gradients must remain enabled because SO3LR computes
    forces via autograd differentiation of the energy.

    Args:
        model_path: Path to saved model file on shared filesystem.
        hdf5_path: Path to raw HDF5 v2.0 dataset.
        batch_indices: Global HDF5 indices to evaluate.
        threshold: Error threshold for filtering.
        r_max: Short-range cutoff radius.
        r_max_lr: Long-range cutoff radius (None if not used).
        error_type: Which error to compute: "forces", "energy", or "both".
        energy_key: Key for energy in HDF5 (e.g. "REF_energy").
        forces_key: Key for forces in HDF5 (e.g. "REF_forces").
        z_table_zs: List of atomic numbers for the z_table.
        head_index: Which output head to use for multi-head models.
        multihead: Whether the model is a MultiHeadSO3LR model.
        eval_batch_size: Mini-batch size for the internal DataLoader.
        device: Device string for model inference ("cpu", "cuda", "cuda:0", …).

    Returns:
        dict with keys:
            "exceeding_indices": list of int — indices that exceeded threshold
            "batch_errors": list of float — error per structure
            "mean_batch_error": float — mean error over batch
            "head_index": int — the head_index used
    """
    import torch
    import numpy as np
    from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5
    from so3krates_torch.data.atomic_data import AtomicData
    from so3krates_torch.data.utils import (
        KeySpecification,
        config_from_atoms,
    )
    from so3krates_torch.tools.utils import AtomicNumberTable
    from so3krates_torch.tools import torch_geometric as so3_torch_geometric

    _device = torch.device(device)
    z_table = AtomicNumberTable(z_table_zs)
    key_spec = KeySpecification(
        info_keys={"energy": energy_key},
        arrays_keys={"forces": forces_key},
    )

    model = torch.load(model_path, map_location=_device)
    model.eval()
    if multihead and hasattr(model, "select_heads"):
        model.select_heads = True

    # -----------------------------------------------------------------------
    # Combined mode: single-head multi-dataset pool
    # -----------------------------------------------------------------------
    if hdf5_paths is not None and batch_items is not None:
        from collections import defaultdict

        _cutoff_lr = r_max_lr if r_max_lr is not None else r_max
        all_heads_combined = None  # combined mode is always single-head

        # Group batch_items by ds_idx for efficient HDF5 loading
        by_ds: dict = defaultdict(list)
        for item_idx, (ds_i, local_i) in enumerate(batch_items):
            by_ds[ds_i].append((item_idx, local_i))

        data_list_combined = [None] * len(batch_items)
        for ds_i, idx_pairs in by_ds.items():
            item_positions = [p[0] for p in idx_pairs]
            local_indices = [p[1] for p in idx_pairs]
            atoms_group = load_atoms_from_hdf5(
                hdf5_paths[ds_i], index=local_indices
            )
            if not isinstance(atoms_group, list):
                atoms_group = [atoms_group]
            for pos, local_i, atoms in zip(
                item_positions, local_indices, atoms_group
            ):
                cfg = config_from_atoms(atoms, key_specification=key_spec)
                data = AtomicData.from_config(
                    cfg,
                    z_table=z_table,
                    cutoff=r_max,
                    cutoff_lr=_cutoff_lr,
                    heads=all_heads_combined,
                )
                data.global_idx = torch.tensor(
                    [local_i], dtype=torch.long
                )
                data.ds_idx = torch.tensor([ds_i], dtype=torch.long)
                data_list_combined[pos] = data

        loader_combined = so3_torch_geometric.dataloader.DataLoader(
            dataset=data_list_combined,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

        exceeding_items = []
        batch_errors_combined = []

        for mini_batch in loader_combined:
            mini_batch = mini_batch.to(_device)
            n_structs = mini_batch.num_graphs

            with torch.enable_grad():
                output = model(mini_batch.to_dict())

            force_errors_c = None
            energy_errors_c = None

            if error_type in ("forces", "both"):
                ref_forces = mini_batch.forces
                pred_forces = output.get("forces")
                if ref_forces is not None and pred_forces is not None:
                    per_atom_abs = (
                        torch.abs(pred_forces - ref_forces)
                        .mean(dim=-1)
                        .detach()
                    )
                    force_errors_c = torch.zeros(
                        n_structs, device=_device
                    )
                    counts_c = torch.zeros(n_structs, device=_device)
                    force_errors_c.scatter_add_(
                        0, mini_batch.batch, per_atom_abs
                    )
                    counts_c.scatter_add_(
                        0,
                        mini_batch.batch,
                        torch.ones(
                            per_atom_abs.shape[0], device=_device
                        ),
                    )
                    force_errors_c = force_errors_c / counts_c.clamp(
                        min=1
                    )

            if error_type in ("energy", "both"):
                ref_energy = mini_batch.energy
                pred_energy = output.get("energy")
                if ref_energy is not None and pred_energy is not None:
                    n_atoms_c = torch.bincount(
                        mini_batch.batch, minlength=n_structs
                    ).float()
                    energy_errors_c = (
                        torch.abs(
                            pred_energy.squeeze(-1)
                            - ref_energy.squeeze(-1)
                        )
                        / n_atoms_c.clamp(min=1)
                    ).detach()

            errors_c = torch.zeros(n_structs, device=_device)
            n_terms_c = 0
            if force_errors_c is not None:
                errors_c += force_errors_c
                n_terms_c += 1
            if energy_errors_c is not None:
                errors_c += energy_errors_c
                n_terms_c += 1
            if n_terms_c > 0:
                errors_c = errors_c / n_terms_c

            errors_np_c = errors_c.cpu().numpy()
            local_idxs_c = mini_batch.global_idx.cpu().numpy()
            ds_idxs_c = mini_batch.ds_idx.cpu().numpy()

            for err, local_i, ds_i in zip(
                errors_np_c, local_idxs_c, ds_idxs_c
            ):
                err_float = float(err)
                batch_errors_combined.append(err_float)
                if err_float > threshold:
                    exceeding_items.append((int(ds_i), int(local_i)))

        mean_batch_error_combined = (
            float(np.mean(batch_errors_combined))
            if batch_errors_combined
            else 0.0
        )
        return {
            "exceeding_items": exceeding_items,
            "batch_errors": batch_errors_combined,
            "mean_batch_error": mean_batch_error_combined,
            "head_index": 0,
        }

    # -----------------------------------------------------------------------
    # Standard mode (single dataset or multi-head)
    # -----------------------------------------------------------------------
    atoms_list = load_atoms_from_hdf5(hdf5_path, index=batch_indices)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    all_heads = [f"head_{i}" for i in range(model.num_output_heads)] \
        if multihead else None

    # When r_max_lr is None, SO3LR still needs edge_index_lr (electrostatics
    # / dispersion are enabled by default). Fall back to r_max so the graph
    # is always built with a valid cutoff.
    _cutoff_lr = r_max_lr if r_max_lr is not None else r_max

    # Build AtomicData list, tagging each entry with its global HDF5 index
    data_list = []
    for local_i, atoms in enumerate(atoms_list):
        config = config_from_atoms(atoms, key_specification=key_spec)
        data = AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
            cutoff_lr=_cutoff_lr,
            heads=all_heads,
        )
        if multihead:
            data.head = torch.tensor([head_index], dtype=torch.long)
        data.global_idx = torch.tensor(
            [batch_indices[local_i]], dtype=torch.long
        )
        data_list.append(data)

    loader = so3_torch_geometric.dataloader.DataLoader(
        dataset=data_list,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    exceeding_indices = []
    batch_errors = []

    for mini_batch in loader:
        mini_batch = mini_batch.to(_device)
        n_structs = mini_batch.num_graphs

        output = model(mini_batch.to_dict())

        force_errors = None
        energy_errors = None

        if error_type in ("forces", "both"):
            ref_forces = mini_batch.forces
            pred_forces = output.get("forces")
            if ref_forces is not None and pred_forces is not None:
                per_atom_abs = (
                    torch.abs(pred_forces - ref_forces)
                    .mean(dim=-1)
                    .detach()
                )
                force_errors = torch.zeros(n_structs, device=_device)
                counts = torch.zeros(n_structs, device=_device)
                force_errors.scatter_add_(
                    0, mini_batch.batch, per_atom_abs
                )
                counts.scatter_add_(
                    0,
                    mini_batch.batch,
                    torch.ones(
                        per_atom_abs.shape[0], device=_device
                    ),
                )
                force_errors = force_errors / counts.clamp(min=1)

        if error_type in ("energy", "both"):
            ref_energy = mini_batch.energy
            pred_energy = output.get("energy")
            if ref_energy is not None and pred_energy is not None:
                n_atoms = torch.bincount(
                    mini_batch.batch, minlength=n_structs
                ).float()
                energy_errors = (
                    torch.abs(
                        pred_energy.squeeze(-1) - ref_energy.squeeze(-1)
                    )
                    / n_atoms.clamp(min=1)
                ).detach()

        errors = torch.zeros(n_structs, device=_device)
        n_terms = 0
        if force_errors is not None:
            errors += force_errors
            n_terms += 1
        if energy_errors is not None:
            errors += energy_errors
            n_terms += 1
        if n_terms > 0:
            errors = errors / n_terms

        errors_np = errors.cpu().numpy()
        global_idxs = mini_batch.global_idx.cpu().numpy()

        for err, gidx in zip(errors_np, global_idxs):
            err_float = float(err)
            batch_errors.append(err_float)
            if err_float > threshold:
                exceeding_indices.append(int(gidx))

    mean_batch_error = float(np.mean(batch_errors)) if batch_errors else 0.0

    return {
        "exceeding_indices": exceeding_indices,
        "batch_errors": batch_errors,
        "mean_batch_error": mean_batch_error,
        "head_index": head_index,
    }
