## Overview

In this example, we use *aims PAX* to create a dataset and model for a rather simple molecule: aspirin.

This is just to explain some settings, the outputs, and to help you run the code yourself. The job files and some of the `CLUSTER` settings are made with [Meluxina](https://docs.lxp.lu/) in mind, so you have to adjust them accordingly.

**Things to change:**

You'll have to adjust some of the settings in the example depending of your conda and HPC environment.

1. Activation of environment and slurm settings in `example_aimsPAX.sh`
2. Path to species defaults in `aimsPAX.yaml` under `INITIAL_DATASET_GENERATION` and `ACTIVE_LEARNING`
3. `slurm_str` and `worker_str` in `aimsPAX.yaml` under `CLUSTER`
4. `launc_str` in `aimsPAX.yaml` under `CLUSTER`
5. `calc_dir` in `aimsPAX.yaml` under `CLUSTER`

### DFT Settings
[FHI-aims](https://fhi-aims.org/) is used for running DFT, and inside the `control.in` file the functional, dispersion correction, and other settings like SCF convergence criteria are specified. For more details on what settings are possible, consult the [official manual](https://fhi-aims.org/uploads/documents/FHI-aims.250320_1.pdf).

In our example, we will use PBE as it is cheap to evaluate.

The species defaults for numerical settings and basis sets are defined in `aimsPAX.yaml` as we will see later on (this is contrary to what is usually done in FHI aims, where these defaults are part of the `control.in` file).

### Initial Dataset Generation Settings

Before starting with the active learning itself, we have to create an initial dataset and initial models. In *aims PAX*, the default approach is to sample geometries using a pretrained model such as [MACE-MP0]() and then relabel them with the target DFT method.

This is done automatically when running `aims-PAX` in the command line. We can also run it independently by using the `aims-PAX-initial-ds` command.

In the `aimsPAX.yaml` file, you can find the settings for IDG under `INITIAL_DATASET_GENERATION`.

A detailed account of all settings is given in the main readme of *aims PAX*, but we want to highlight some important ones here. 

First, we set `desired_acc` to `0.1` eV/A, which is the target force MAE we want to achieve on the validation set. By setting `desired_acc_scale_idg` to `3`, this target MAE is scaled to `0.3` eV/A. Once the model reaches `0.3` eV/A on the validation set, the IDG stops.

During the procedure, we use the `medium` version of `mace-mp0`, as specified under `foundational_model`, and `foundational_model_settings/mace_model` to generate `10` structures per ensemble member. This is what is meant in `n_points_per_sampling_step_idg`. These are then labelled using DFT, and the models are trained for `5` (`intermediate_epochs_idg`) epochs. During training, the validation MAE is monitored in order to check when the IDG is supposed to stop.

### MD Settings

Sampling is done using molecular dynamics (MD) with the settings specified in `MD`.

### Cluster Settings

Resource management settings, handled by [PARSL](), are defined in `CLUSTER` at the bottom of the settings file.

Under `parsl_options`, one can specify how many resources are used for DFT. Each calculation is done inside a `block`, so you can set how many nodes each DFT run gets. There you can also manage how many blocks are initialized at the start (`init_blocks`), how many are kept alive at least (`min_blocks`), and how many will be scheduled at most (`max_blocks`). When a DFT calculation is to be done, a Slurm job will be submitted if there are no running blocks; if there are, the calculation is done there (or put in a queue). So it's not the case that each DFT calculation needs to be submitted to Slurm individually by *aims-PAX*. 

As the DFT jobs run independently through Slurm, we must specify all necessary settings under `slurm_str` and `worker_str` to run a stand-alone job. In the former, all the Slurm-relevant variables have to be mentioned, such as partitions, etc., and in the latter the environment where you installed *aims-PAX* has to be sourced, and modules as well as environment variables have to be set. These can differ across different HPC environments.

Finally, under `launch_str` you must write the exact commands that are used to run FHI aims, and under `calc_dir` a path has to be given where the DFT calculations will be performed (it does not have to exist; it will be created if not). By default, after DFT is done, the subdirectory for a given calculation is removed (set `clean_dirs` to `False` here to change this).

### Active Learning Settings

In the active learning settings, we encounter some keywords we have seen before in the initial dataset generation settings. New ones are, for example, `epochs_per_worker` and `intermediate_epochs_al`. When a DFT calculation is done, the models are trained on the updated datasets for a total of `epochs_per_worker`. This is done in steps of `intermediate_epochs_al`. So the model is trained, then other trajectories are propagated, and training continues. This becomes clear when you take a look inside the logs while the workflow runs.

Another important setting is `num_trajectories`, where you specify how many trajectories `aims-PAX` samples new points from at the same time. From a purely computational performance perspective, the number of trajectories should be higher than the number of blocks (see **Cluster Settings**).

Using `c_x`, the threshold is either tightened (`c_x` < 0) or loosened (`c_x` > 0).

The `skip_step_mlff` determines after how many MD steps we check the uncertainty of the geometry. 

The keywords `converge_al` and `converge_best` determine if the model(s) should be converged on the final dataset and if only the best-performing model should be converged.

### MACE Settings

In the `mace.yaml`, the MLFF-specific settings are defined, such as its architecture and training details. In the present example, we use many default values, which is why it looks so concise. 

In `GENERAL`, we name our models and define the `seed`. This seed is not directly used for the models, but with it, `ensemble_size` seeds are sampled and used for the ensemble members.

In `ARCHITECTURE`, the structure of the MACE model itself is set. For example, we want the cutoff (`r_max`) to be 5 Å and the maximum degree of equivariant features (`max_L`) to be 1.

In `TRAINING`, parameters such as the learning rate (`lr`), size of training (`batch_size`) and validation (`valid_batch_size`) batches, and what loss function (`loss`) to use are defined.

Lastly, in `MISC` under `device`, we specify that we want to use a GPU for the MLFF (`cuda`).

### Running

In our example, we simply start the procedure by launching the job script `example_aimsPAX.sh`. Inside, `aims-PAX` is invoked, which runs the whole workflow, including initial dataset generation and active learning.

### Logs

During the workflow, you can follow its progress in the logs inside the `logs` directory. All logs show you **all** settings (specified and default) that are used in this specific run, along with other general information.

#### Initial Dataset Generation Log

During the initial dataset generation, you will see how the pretrained model is used to sample points, DFT calculations are launched, and how the model is trained step by step.

In this example, the desired accuracy is reached in 5 steps of sampling, DFT, and training.

#### Active Learning Log

The active learning log is a bit more detailed. You will see how the ensemble is used to propagate the different trajectories. Each time the progress of a trajectory is reported, the uncertainty is measured and compared to the threshold.

You will then see how some trajectories cross the threshold, DFT jobs are launched, and how the models are updated.

At the end, when the desired accuracy is reached (after roughly 80 points are sampled), the active learning ends. In our example, the best-performing model of the ensemble is converged on the final dataset. This means we continue to train the model until either the specified `max_convergence_epochs` are reached or the model has not improved for `convergence_patience` epochs.

### Outputs

The final model and datasets are saved in the `results` folder.  
