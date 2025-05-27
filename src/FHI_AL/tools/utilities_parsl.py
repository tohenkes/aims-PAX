from parsl.config import Config
from parsl.executors import WorkQueueExecutor
from parsl.providers import SlurmProvider
from parsl import python_app
import re
import logging

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
    port = parsl_options.get("port", 9000)
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
        ]
    )
    return config


@python_app
def parsl_test_app():
    import time
    time.sleep(1)
    with open("/home/users/u101418/al_aims/asi/FHI_AL/examples/parsl/parsl_test.txt", "a") as f:
        f.write("Hello from PARSL!\n")
    return 0

@python_app
def recalc_aims_parsl(
    atoms,
    aims_settings: dict,
    directory: str = "./",
    properties: list = ["energy", "forces"],
    ase_aims_command: str = None,
):

    from ase.calculators.aims import Aims, AimsProfile
    import os

    # create output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    os.environ["ASE_AIMS_COMMAND"] = ase_aims_command

    calc = Aims(
        profile=AimsProfile(command=os.environ["ASE_AIMS_COMMAND"]),
        directory=directory,
        **aims_settings,
    )

    calc.calculate(atoms=atoms, properties=properties, system_changes=None)
    return calc.results
