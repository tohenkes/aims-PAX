import os
import threading
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

from aims_PAX.settings.project import ClusterSettings


def _patch_mace_simplify_for_thread_safety():
    """Patch mace.tools.compile.simplify to be a no-op in non-main threads.

    MACE's simplify() calls torch.fx.symbolic_trace(), which patches
    nn.Module.__call__ globally at the class level for the duration of
    tracing. When this runs in a PARSL ThreadPoolExecutor thread concurrently
    with MLFF MD in the main thread, the main thread's model sees the patched
    __call__ and raises NameError: module is not installed as a submodule.

    Skipping simplify() in non-main threads is safe: it only prepares modules
    for torch.compile, which we do not use for teacher reference calculations.
    Remote PARSL workers are separate processes whose function runs in their
    own main thread, so simplify() still runs normally there.
    """
    try:
        import mace.tools.compile as _mace_compile

        _orig_simplify = _mace_compile.simplify

        def _thread_safe_simplify(module):
            if threading.current_thread() is threading.main_thread():
                return _orig_simplify(module)
            return module

        _mace_compile.simplify = _thread_safe_simplify
    except ImportError:
        pass


_patch_mace_simplify_for_thread_safety()


def prepare_parsl(
    cluster_settings: ClusterSettings = None, output_dir: Path = Path(".")
) -> dict:
    """
    Prepare the PARSL configuration and settings for running calculations.
    Creates the config file, sets the calculation directory, defines if
    the calculation directories should be cleaned, and initializes
    the calculation index (is used to keep track of each calculation
    when creating the directory names). Also, checks if the launch string
    is provided in the cluster settings. The launch string is used to run
    the DFT calculations on the cluster.

    Args:
        cluster_settings (ClusterSettings, optional): _description_. Defaults to None.
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
    launch_str = cluster_settings.launch_str

    config = create_parsl_config(
        cluster_settings=cluster_settings, output_dir=output_dir
    )
    # get the path to the directory where the calculations will be run
    # if none is provided use the current working directory
    calc_dir = cluster_settings.calc_dir
    clean_dirs = cluster_settings.clean_dirs
    calc_idx = 0

    return {
        "config": config,
        "calc_dir": calc_dir,
        "clean_dirs": clean_dirs,
        "launch_str": launch_str,
        "calc_idx": calc_idx,
    }


def create_parsl_config(cluster_settings: ClusterSettings, output_dir: Path = Path(".")) -> Config:
    """
    Reads in CLUSTER settings as a Pydantic model (provided in the yaml file).
    The information is then used to create a PARSL configuration object.
    The configuration includes the executor, provider, and other settings
    necessary for running calculations on the cluster.

    One block is capable of running PARSL apps like the DFT calculations
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

    executor = cluster_settings.executor

    if executor == "local":
        parsl_info_dir = output_dir / "parsl_info"
        if not os.path.exists(parsl_info_dir):
            os.makedirs(parsl_info_dir)
        run_dir = parsl_info_dir / "run_dir"
        config = Config(
            executors=[
                ParslThreadPoolExecutor(
                    label="local",
                    max_threads=cluster_settings.max_workers,
                )
            ],
            run_dir=str(run_dir),
            app_cache=False,
            initialize_logging=False,
            retries=0,
        )
        return config
    parsl_options = cluster_settings.parsl_options

    nodes_per_block = parsl_options.nodes_per_block
    init_blocks = parsl_options.init_blocks
    min_blocks = parsl_options.min_blocks
    max_blocks = parsl_options.max_blocks
    label = parsl_options.label

    run_dir = parsl_options.run_dir
    function_dir = parsl_options.function_dir

    worker_init_str = cluster_settings.worker_str
    slurm_options_str = cluster_settings.slurm_str

    # Extract the cluster partition from the slurm options string
    match = re.search(
        r"(?: --partition| -p) *= *([\w-]*)",
        slurm_options_str,
        re.IGNORECASE,
    )
    if match:
        partition = match.group(1)
    else:
        logging.error("Partition not found. Closing.")
        raise KeyError("Partition not found in slurm options string.")

    config = None
    if executor == "workqueue":
        config = Config(
            executors=[
                WorkQueueExecutor(
                    label=label,
                    port=0,
                    shared_fs=True,  # assumes shared file system
                    function_dir=str(function_dir),
                    autocategory=False,
                    provider=SlurmProvider(
                        partition=partition,
                        nodes_per_block=nodes_per_block,
                        init_blocks=init_blocks,
                        min_blocks=min_blocks,
                        max_blocks=max_blocks,
                        scheduler_options=slurm_options_str,
                        worker_init=worker_init_str,
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
    from aims_PAX.tools.utilities.utilities import (
        get_free_vols,
        get_hirshfeld_charges,
    )
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
        aims_output_path = os.path.join(directory, aims_output_file)
        if os.path.exists(aims_output_path):
            with open(aims_output_path, "r") as f:
                aims_output = f.readlines()
            if "hirshfeld_volumes" in results.keys():
                free_vols = get_free_vols(aims_output)
                hirshfeld_vols = results["hirshfeld_volumes"]
                results["hirshfeld_ratios"] = hirshfeld_vols / free_vols
            hirshfeld_charges = get_hirshfeld_charges(aims_output)
            if hirshfeld_charges:
                results["hirshfeld_charges"] = np.array(hirshfeld_charges)
        return results
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


_teacher_calc_cache: dict = {}
_teacher_calc_lock = threading.Lock()


def _build_teacher_calculator(
    model_type, model_path, model_settings, properties=None
):
    """Build a teacher ML calculator from the given settings.

    Separated from the PARSL app so it can be called in the main thread
    for pre-loading (see preload_teacher_calculator).
    """
    if properties is None:
        properties = ["energy", "forces"]
    device = model_settings.get("device", "cpu")
    default_dtype = model_settings.get("default_dtype", "float64")
    compute_stress = "stress" in properties

    if model_type == "mace-mp":
        from mace.calculators import mace_mp

        return mace_mp(
            model=model_settings.get("mace_model", "small"),
            dispersion=model_settings.get("dispersion", False),
            default_dtype=default_dtype,
            device=device,
            damping=model_settings.get("damping", "bj"),
            dispersion_xc=model_settings.get("dispersion_xc", "pbe"),
            dispersion_cutoff=model_settings.get(
                "dispersion_cutoff", 12.0
            ),
        )
    elif model_type == "mace":
        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=model_path,
            device=device,
            default_dtype=default_dtype,
        )
    elif model_type in ["so3lr", "so3krates"]:
        from so3krates_torch.calculator.so3 import TorchkratesCalculator

        return TorchkratesCalculator(
            model_paths=model_path,
            compute_stress=compute_stress,
            device=device,
            default_dtype=default_dtype,
            r_max_lr=model_settings.get("r_max_lr", None),
            dispersion_lr_damping=model_settings.get(
                "dispersion_lr_damping", None
            ),
        )
    else:
        raise ValueError(
            f"Unknown teacher model type: {model_type}. "
            "Supported types: 'mace-mp', 'mace', 'so3lr', 'so3krates'."
        )


def preload_teacher_calculator(
    model_type: str,
    model_path,
    model_settings: dict,
    properties=None,
) -> None:
    """Pre-create and cache the teacher calculator in the calling thread.

    Must be called from the main thread before the parallel AL loop starts.
    This ensures that torch.fx.symbolic_trace (triggered inside
    SymmetricContraction.__init__ and mace.tools.compile.simplify during
    MACECalculator construction) runs in the main thread rather than in a
    PARSL ThreadPoolExecutor thread.  Running symbolic_trace in a non-main
    thread while the main thread executes MLFF model forward passes causes a
    race condition: symbolic_trace temporarily patches nn.Module.__call__ at
    the class level, and the main thread's MLFF model hits the patched call,
    resulting in NameError: module is not installed as a submodule.
    """
    if model_settings is None:
        model_settings = {}
    cache_key = (
        model_type,
        str(model_path),
        str(sorted(model_settings.items())),
    )
    if cache_key not in _teacher_calc_cache:
        logging.info(
            f"Pre-loading teacher calculator ({model_type}) in main thread "
            "to prevent torch.fx thread-safety issues."
        )
        calc = _build_teacher_calculator(
            model_type, model_path, model_settings, properties
        )
        _teacher_calc_cache[cache_key] = calc


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

    For local PARSL (ThreadPoolExecutor), uses a pre-built cached calculator
    to avoid torch.fx.symbolic_trace thread-safety issues during model init.

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

    cache_key = (
        model_type,
        str(model_path),
        str(sorted(model_settings.items())),
    )
    # Local PARSL (ThreadPoolExecutor) runs in a non-main thread of the same
    # process.  Use the pre-built cached calculator so that no new
    # torch.fx.symbolic_trace calls happen in this thread.
    # Remote PARSL workers are separate processes whose function runs in their
    # own main thread; the cache is empty there, so they build a fresh
    # calculator normally (no race condition in a dedicated process).
    in_local_parsl_thread = (
        threading.current_thread() is not threading.main_thread()
    )
    if in_local_parsl_thread and cache_key in _teacher_calc_cache:
        try:
            with _teacher_calc_lock:
                calc = _teacher_calc_cache[cache_key]
                calc.calculate(atoms=atoms, properties=properties)
                return dict(calc.results)
        except Exception as e:
            logging.warning(
                f"Teacher model calculation failed (cached calc): {str(e)}"
            )
            return None

    # Fallback: build a fresh calculator (remote workers or cache miss).
    try:
        calc = _build_teacher_calculator(
            model_type, model_path, model_settings, properties
        )
        calc.calculate(atoms=atoms, properties=properties)
        return calc.results
    except Exception as e:
        logging.warning(f"Teacher model calculation failed: {str(e)}")
        return None
