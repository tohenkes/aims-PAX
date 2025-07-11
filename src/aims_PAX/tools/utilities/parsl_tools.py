import os
from parsl.config import Config
from parsl.executors import WorkQueueExecutor
from parsl.providers import SlurmProvider
from parsl import python_app
import numpy as np
import re
import logging
from ase.io import ParseError
from pathlib import Path
import socket


def get_free_port():
    """
    Select a free port for the PARSL server to use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # let OS pick an available port
        return s.getsockname()[1]


def prepare_parsl(
    cluster_settings: dict = None
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
    try:
        launch_str = cluster_settings["launch_str"]
    except KeyError:
        exception_msg = "Launch string not found in YAML file. Closing."
        raise KeyError(exception_msg)

    config = create_parsl_config(cluster_settings=cluster_settings)
    # get the path to the directory where the calculations will be run
    # if none is provided use the current working directory
    calc_dir = cluster_settings.get(
        "calc_dir", os.getcwd() + "/" + "ase_aims_calcs/"
    )
    calc_dir = Path(calc_dir)
    clean_dirs = cluster_settings.get("clean_dirs", True)
    calc_idx = 0

    return {
        "config": config,
        "calc_dir": calc_dir,
        "clean_dirs": clean_dirs,
        "launch_str": launch_str,
        "calc_idx": calc_idx,
    }


def create_parsl_config(
    cluster_settings: dict
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
    
    project_name = cluster_settings.get("project_name", "fhi_aims_dft")
    parsl_options = cluster_settings["parsl_options"]

    nodes_per_block = parsl_options.get("nodes_per_block", 1)
    init_blocks = parsl_options.get("init_blocks", 1)
    min_blocks = parsl_options.get("min_blocks", 1)
    max_blocks = parsl_options.get("max_blocks", 1)
    parsl_info_dir = Path(parsl_options.get("parsl_info_dir", "./parsl_info"))
    
    # create a parent folder for all the parsl stuff
    if not os.path.exists(parsl_info_dir):
        os.makedirs(parsl_info_dir)
    run_dir = parsl_options.get("run_dir", parsl_info_dir / "run_dir")
    function_dir = parsl_options.get(
        "function_dir", parsl_info_dir / "function_dir"
    )
    
    port = parsl_options.get(
        "port", get_free_port()
    )  # 0 means automatically choose a free port
    label = parsl_options.get("label", "workqueue")

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
        r"partition\s*=\s*(\S+)", slurm_options_str, re.IGNORECASE
    )
    if match:
        partition = match.group(1)
    else:
        logging.error("Partition not found. Closing.")
        raise KeyError("Partition not found in slurm options string.")

    config = Config(
        executors=[
            WorkQueueExecutor(
                label=label,
                port=port,
                project_name=project_name,
                shared_fs=True,  # assumes shared file system
                function_dir=str(function_dir),
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
def recalc_aims_parsl(
    positions: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
    pbc: bool,
    aims_settings: dict,
    ase_aims_command: str,
    directory: str = "./",
    properties: list = ["energy", "forces"],
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
    import os
    from ase import Atoms

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

    os.environ["ASE_AIMS_COMMAND"] = ase_aims_command

    calc = Aims(
        profile=AimsProfile(command=os.environ["ASE_AIMS_COMMAND"]),
        directory=directory,
        **aims_settings,
    )
    try:
        calc.calculate(
            atoms=atoms, 
            properties=properties, 
            system_changes=None
        )
        return calc.results
    except ParseError:
        return None
