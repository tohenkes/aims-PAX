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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # let OS pick an available port
        return s.getsockname()[1]
    
def prepare_parsl(
    cluster_settings: dict = None
):
    assert cluster_settings is not None, (
        "Cluster settings not found. Please provide a YAML file "
        "with the cluster settings."
    )
    try:
        launch_str = cluster_settings["launch_str"]
    except KeyError:
        exception_msg = "Launch string not found in YAML file. Closing."
        raise KeyError(exception_msg)

    config = create_parsl_config(
        cluster_settings=cluster_settings
    )
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
    cluster_settings
    ):
    
    # Extract the settings
    project_name = cluster_settings.get("project_name", "fhi_aims_dft")
    parsl_options = cluster_settings['parsl_options']

    nodes_per_block = parsl_options.get("nodes_per_block", 1)
    init_blocks = parsl_options.get("init_blocks", 1)
    min_blocks = parsl_options.get("min_blocks", 1)
    max_blocks = parsl_options.get("max_blocks", 1)
    parsl_info_dir = Path(parsl_options.get("parsl_info_dir", "./parsl_info"))
    # create a parent folder for all the parsl stuff
    if not os.path.exists(parsl_info_dir):
        os.makedirs(parsl_info_dir)
    run_dir = parsl_options.get("run_dir", parsl_info_dir / "run_dir")
    function_dir = parsl_options.get("function_dir", parsl_info_dir / "function_dir")
    port = parsl_options.get("port", get_free_port())  # 0 means automatically choose a free port
    label = parsl_options.get("label", "workqueue")

    try:
        worker_init_str = cluster_settings['worker_str']
    except KeyError:
        exception_msg = "Worker init string not found in YAML file. Closing."
        raise KeyError(exception_msg)

    try: 
        slurm_options_str = cluster_settings['slurm_str']
    except KeyError:
        exception_msg = "Slurm options string not found in YAML file. Closing."
        raise KeyError(exception_msg)

    match = re.search(r"partition\s*=\s*(\S+)", slurm_options_str, re.IGNORECASE)
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

def handle_parsl_logger(
        log_dir: Path = Path("./")
):
    parsl_logger = logging.getLogger("parsl")
    parsl_handler = logging.FileHandler(log_dir)
    parsl_logger.handlers.clear()
    parsl_logger.addHandler(parsl_handler)
    parsl_logger.propagate = False

@python_app
def parsl_test_app():
    import time
    time.sleep(1)
    with open("/home/users/u101418/al_aims/asi/aims_PAX/examples/parsl/parsl_test.txt", "a") as f:
        f.write("Hello from PARSL!\n")
    return 0

@python_app
def recalc_aims_parsl(
    positions: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
    pbc: bool,
    aims_settings: dict,
    directory: str = "./",
    properties: list = ["energy", "forces"],
    ase_aims_command: str = None,
):

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
        calc.calculate(atoms=atoms, properties=properties, system_changes=None)
        return calc.results
    except ParseError:
        return None
    