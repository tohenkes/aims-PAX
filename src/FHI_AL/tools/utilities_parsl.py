from parsl.config import Config
from parsl.executors import WorkQueueExecutor
from parsl.providers import SlurmProvider
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
