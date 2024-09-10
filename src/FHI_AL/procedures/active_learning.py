import os
from pathlib import Path
import ase.build
import torch
import numpy as np
from mace import tools
from FHI_AL.tools.custom_MACECalculator import MACECalculator
from FHI_AL.tools.utilities import (
    create_dataloader,
    ensemble_training_setups,
    ensemble_from_folder,
    setup_ensemble_dicts,
    update_model_auxiliaries,
    save_checkpoint,
    save_datasets,
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
    create_mace_dataset,
    ensemble_prediction,
    ensemble_prediction_v2,
    setup_mace_training,
    max_sd_2,
    avg_sd,
    atom_wise_sd,
    atom_wise_f_error,
    Z_from_geometry_in,
    list_files_in_directory,
    get_atomic_energies_from_ensemble,
    create_ztable,
    AIMSControlParser,
)
from FHI_AL.tools.train_epoch_mace import train_epoch, validate_epoch_ensemble
import ase
from ase.io import read
import logging
from mpi4py import MPI
from asi4py.asecalc import ASI_ASE_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units

WORLD_COMM = MPI.COMM_WORLD
WORLD_SIZE = WORLD_COMM.Get_size()
RANK = WORLD_COMM.Get_rank()

class PrepareALProcedure:
    """
    Class to prepare the active learning procedure. It handles all the input files,
    prepares the calculators, models, directories etc.
    """
    def __init__(
        self,
        mace_settings,
        al_settings,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in"
    ) -> None:
        
        logging.basicConfig(
            filename="AL.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        tools.setup_logger(
            level=mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=mace_settings["GENERAL"]["log_dir"],
        )

        if RANK == 0:
            logging.info("Initializing active learning procedure.")
        
        self.control_parser = AIMSControlParser()
        self.handle_al_settings(al_settings)
        self.handle_mace_settings(mace_settings)
        self.handle_aims_settings(path_to_control)
        np.random.seed(self.mace_settings["GENERAL"]["seed"])
        #TODO: will break when using multiple settings for different trajectories and 
        # it should be adapted to the way the initial dataset procecure treats md settings
        self.md_settings = al_settings["MD"]
        
        self.create_folders()
        
        #TODO: this would change with multiple species
        self.z = Z_from_geometry_in()
        self.z_table = create_ztable(self.z)
        self.n_atoms = len(self.z)
        
        if self.seeds_tags_dict is None:
            try:
                self.seeds_tags_dict = dict(
                    np.load(
                        self.dataset_dir / "seeds_tags_dict.npz",
                        allow_pickle=True
                    )
                )
            except:
                logging.error(
    f"Could not load seeds_tags_dict.npz. Either specify it in the active "
    f"learning settings or put it as seeds_tags_dict.npz in {self.dataset_dir}."
            )

        # TODO: remove hardcode
        self.use_scheduler = False

        if RANK == 0:
            self.ensemble = ensemble_from_folder(
                path_to_models="./model",
                device=self.device,
            )

            if self.atomic_energies_dict is not None:
                # if they have been explicitly specified the model should not update them
                # anymore
                self.update_atomic_energies = False
                self.ensemble_atomic_energies_dict = {}
                self.ensemble_atomic_energies = {}
                for tag in self.ensemble.keys():
                    self.ensemble_atomic_energies_dict[tag] = self.atomic_energies_dict
                    self.ensemble_atomic_energies[tag] = np.array(
                        [
                            self.ensemble_atomic_energies_dict[tag][z]
                            for z in self.atomic_energies_dict.keys()
                        ]
                    )
            else:
                try:
                    if RANK == 0:
                        logging.info("Loading atomic energies from existing ensemble.")
                    (
                        self.ensemble_atomic_energies, self.ensemble_atomic_energies_dict
                    ) = get_atomic_energies_from_ensemble(
                        ensemble=self.ensemble,
                        z=self.z,
                    )
                except:
                    if RANK == 0:
                        logging.info('Could not load atomic energies '
                                     'from ensemble members.')
                self.update_atomic_energies = True

            self.training_setups = ensemble_training_setups(
                ensemble=self.ensemble,
                mace_settings=self.mace_settings,
            )
            

        # TODO: make option to directly load the dataset from the
        # initial dataset object instead of reading from disk
        if RANK == 0:
            logging.info("Loading initial datasets.")
            self.ensemble_ase_sets = load_ensemble_sets_from_folder(
                ensemble=self.ensemble,
                path_to_folder=Path(self.al["dataset_dir"] + "/initial"),
            )
            self.ensemble_mace_sets = ase_to_mace_ensemble_sets(
                ensemble_ase_sets=self.ensemble_ase_sets,
                z_table=self.z_table,
                r_max=self.r_max,
                seed=self.seed,
            )
            self.train_dataset_len = len(
                    self.ensemble_ase_sets[list(self.ensemble.keys())[0]][
                        "train"
                    ]
                )
        if RANK != 0:
            self.train_dataset_len = None
        
        MPI.COMM_WORLD.Barrier()
        self.train_dataset_len = MPI.COMM_WORLD.bcast(self.train_dataset_len, root=0)
        MPI.COMM_WORLD.Barrier()
        
        # this initializes the FHI aims process
        self.aims_calculator = self.setup_aims_calc()
        
        self.trajectories = {
            trajectory: read(path_to_geometry) for trajectory in range(self.num_trajectories)
        }
        self.trajectory_training = {
            trajectory: "running"
            for trajectory in range(self.num_trajectories)
        }
        self.trajectory_MD_steps = {
            trajectory: 0 for trajectory in range(self.num_trajectories)
        }
        self.trajectory_epochs = {
            trajectory: 0 for trajectory in range(self.num_trajectories)
        }
        if self.analysis:
            # this saves the intervals between points that cross the uncertainty threshold
            self.t_intervals = {
                trajectory: [] for trajectory in range(self.num_trajectories)
            }
            # this saves uncertainty and true errors for each trajectory
            self.sanity_checks = {
                trajectory: {
                    "atom_wise_uncertainty": [],
                    "uncertainty_via_max": [],
                    "uncertainty_via_avg": [],
                    "max_error": [],
                    "mean_error": [],
                    "atom_wise_error": [],
                    "threshold": [],
                    } for trajectory in range(self.num_trajectories)
            }
            # this saves the validation losses for each trajectory
            self.collect_losses = {
                "epoch": [],
                "avg_losses": [],
                "ensemble_losses": [],
            }
            
            self.collect_thresholds = {
                trajectory: [] for trajectory in range(self.num_trajectories)
            }
            

        if RANK == 0:
            self.setup_mace_calc()
            for trajectory in self.trajectories.values():
                trajectory.calc = self.mace_calc

            #TODO: make this more flexible, to allow different drivers per trajectory
            # just make a dictionary of different md settings and pass them here
            self.md_drivers = {
                trajectory: self.setup_md_al(
                    atoms=self.trajectories[trajectory], md_settings=self.md_settings
                ) for trajectory in range(self.num_trajectories)
            }
            # this saves the state of the trajectories


        self.uncertainties = []  # for moving average
        
        

        
            
    def handle_mace_settings(self, mace_settings: dict):
        self.mace_settings = mace_settings
        self.seed = self.mace_settings["GENERAL"]["seed"]
        self.r_max = self.mace_settings["ARCHITECTURE"]["r_max"]
        self.set_batch_size = self.mace_settings["TRAINING"]["batch_size"]
        self.set_valid_batch_size = self.mace_settings["TRAINING"][
            "valid_batch_size"
        ]
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.device = self.mace_settings["MISC"]["device"]
        self.model_dir = self.mace_settings["GENERAL"]["model_dir"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
        self.device = self.mace_settings["MISC"]["device"]
        self.atomic_energies_dict = self.mace_settings["ARCHITECTURE"].get("atomic_energies", None)
    
    def handle_al_settings(self, al_settings):
        
        self.al = al_settings['ACTIVE_LEARNING']
        self.al_misc = al_settings.get('MISC', {})
        
        self.max_MD_steps = self.al["max_MD_steps"]
        self.max_epochs_worker = self.al["max_epochs_worker"]
        self.max_final_epochs = self.al["max_final_epochs"]
        self.desired_accuracy = self.al["desired_acc"]
        self.num_trajectories = self.al["num_trajectories"]
        self.skip_step = self.al["skip_step_mlff"]
        self.valid_skip = self.al["valid_skip"]
        self.sanity_skip = self.al["sanity_skip"]
        self.valid_ratio = self.al["valid_ratio"]
        self.max_set_size = self.al["max_set_size"]
        self.num_trajectories = self.al["num_trajectories"]
        self.c_x = self.al["c_x"]
        self.intermediate_epochs = self.al["intermediate_epochs"]
        self.dataset_dir = Path(self.al["dataset_dir"])
        self.patience = self.al["patience"]
        self.species_dir = self.al["species_dir"]
        self.ASI_path = self.al["aims_lib_path"]
        self.analysis = self.al.get("analysis", False)
        self.seeds_tags_dict = self.al.get("seeds_tags_dict", None)
        self.margin = self.al.get("margin", 0.001)
        
        self.save_data_len_interval = self.al_misc.get("save_data_len_interval", 10)
    
    def handle_aims_settings(
        self,
        path_to_control: str
        ):
        """
        Loads and parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.

        """
        
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings['compute_forces'] = True
        self.aims_settings['species_dir'] = self.species_dir
    
    def create_folders(self):
        """
        Create the folders for the final datasets.
        """
        (self.dataset_dir / "final" / "training").mkdir(
            parents=True, exist_ok=True
        )
        (self.dataset_dir / "final" / "validation").mkdir(
            parents=True, exist_ok=True
        )
        if self.analysis:
            os.makedirs("analysis", exist_ok=True)

    def setup_aims_calc(
            self,
            path_to_geometry: str = "./geometry.in"
            ):
        """
        Creates and returns the AIMS calculator for a given atoms object.

        Args:
            path_to_geometry (str, optional): Path to geometry file. Defaults to "./geometry.in".
        """
        
        def init_via_ase(asi):
            
            from ase.calculators.aims import Aims
            calc = Aims(**self.aims_settings)
            calc.write_input(asi.atoms)
            
        calculator = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            MPI.COMM_WORLD,
            read(path_to_geometry) # TODO: must be changed when we have multiple species and then we need multiaims
            )
        return calculator

    def setup_mace_calc(self):
        """
        Loads the models of the existing ensemble and creates the MACE calculator.
        """
        model_paths = list_files_in_directory(self.model_dir)
        self.models = [
            torch.load(f=model_path, map_location=self.device) for model_path in model_paths
        ]
        # the calculator needs to be updated consistently see below
        self.mace_calc = MACECalculator(
            models=self.models,
            device=self.device,
            default_dtype=self.dtype)
        
    def setup_md_al(
            self,
            atoms: ase.Atoms,
            md_settings: dict
            ):
        """
        Sets up the ASE molecular dynamics object for the atoms object using
        the MD settings.

        Args:
            atoms (ase.Atoms): Atoms object to be propagated.
            md_settings (dict): Dictionary containing the MD settings.

        Returns:
            ase.md.MolecularDynamics: ASE MD engine.
        """
        #TODO: make this more flexible
        if md_settings["stat_ensemble"].lower() == 'nvt':
            if md_settings['thermostat'].lower() == 'langevin':
                dyn = Langevin(
                    atoms,
                    timestep=md_settings['timestep'] * units.fs,
                    friction=md_settings['friction'] / units.fs,
                    temperature_K=md_settings['temperature'],
                    rng=np.random.RandomState(md_settings['seed'])
                )
        # make this optional and have the possibility for different initial temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=md_settings['temperature'])
    
        return dyn


class ALProcedure(PrepareALProcedure):
    """
    Class for the active learning procedure. It handles the training of the ensemble
    members, the molecular dynamics simulations, the sampling of points and the saving
    of the datasets.
    """
    def sanity_check(
    #TODO: put this somewhere else and its ugly
            self,
            sanity_prediction: np.ndarray,
            true_forces: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the force uncertainty and the maximum 
        force error for a given prediction and true forces.

        Args:
            sanity_prediction (np.ndarray): Ensemble prediction. [n_members, n_points, n_atoms, 3]
            true_forces (np.ndarray): True forces. [n_points, n_atoms, 3]

        Returns:
            tuple[np.ndarray, np.ndarray]: force uncertainty, true force error
        """
        uncertainty_via_max = max_sd_2(sanity_prediction)
        uncertainty_via_avg = avg_sd(sanity_prediction)
        atom_wise_uncertainty = atom_wise_sd(sanity_prediction)
        mean_sanity_prediction = sanity_prediction.mean(0).squeeze()
        difference = true_forces - mean_sanity_prediction
        diff_sq = difference**2
        diff_sq_mean = np.mean(diff_sq, axis=-1)
        
        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
        atom_wise_error = atom_wise_f_error(mean_sanity_prediction, true_forces)

        return (
            atom_wise_uncertainty,
            uncertainty_via_max,
            uncertainty_via_avg,
            atom_wise_error,
            max_error,
            mean_error
            )

    def waiting_task(
            self,
            idx: int
            ):
        """
        Currently only adds the current point to the training or validation set.

        Args:
            idx (int): Index of the trajectory worker.

        """

        # there is no waiting time here and if we do it sequentially there is not waiting either
        # thus the models directly continue training with the new point which could make quite the difference
        # same with adding a training point to each of the ensemble members which slows down things considerably
        # as we have to wait for enough training points to be acquired
        # if calculation is finished:

        if self.point_added % self.valid_ratio == 0:
            self.trajectory_training[idx] = "running"
            self.num_workers_waiting -= 1
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the validation set."
                )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            if RANK == 0:
                for tag in self.ensemble_ase_sets.keys():
                    self.ensemble_ase_sets[tag]["valid"] += [self.point]
                    self.ensemble_mace_sets[tag]["valid"] += self.mace_point

        else:
            self.trajectory_training[idx] = "training"
            self.num_workers_training += 1
            self.num_workers_waiting -= 1
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the training set."
                )
                # while the initial datasets are different for each ensemble member we add the new points to
                # all ensemble member datasets
                for tag in self.ensemble_ase_sets.keys():
                    self.ensemble_ase_sets[tag]["train"] += [self.point]
                    self.ensemble_mace_sets[tag]["train"] += self.mace_point
                self.train_dataset_len = len(self.ensemble_ase_sets[tag]["train"])
            
            MPI.COMM_WORLD.Barrier()
            self.train_dataset_len = MPI.COMM_WORLD.bcast(self.train_dataset_len, root=0)
            MPI.COMM_WORLD.Barrier()
            
            if self.train_dataset_len > self.max_set_size:
                return True
            if RANK == 0:
                logging.info(
                    f"Size of the training and validation set: {self.train_dataset_len}, {len(self.ensemble_ase_sets[tag]['valid'])}."
                )
        self.point_added += 1

    def training_task(
            self,
            idx: int
            ):
        """
        Creates the dataloader of the updated dataset, updates
        the average number of neighbors, shifts and scaling factor
        and trains the ensemble members. Saves the models and checkpoints.

        Args:
            idx (int): Index of the trajectory worker.
        """
        if RANK == 0:
            for _, (tag, model) in enumerate(self.ensemble.items()):

                (
                    self.ensemble_mace_sets[tag]["train_loader"],
                    self.ensemble_mace_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    self.ensemble_mace_sets[tag]["train"],
                    self.ensemble_mace_sets[tag]["valid"],
                    self.set_batch_size,
                    self.set_valid_batch_size,
                )
                # because the dataset size is dynamically changing we have to update the average number of neighbors
                # and shifts and the scaling factor for the models
                # usually they converge pretty fast
                update_model_auxiliaries(
                    model=model,
                    mace_sets=self.ensemble_mace_sets[tag],
                    atomic_energies=self.ensemble_atomic_energies[tag],
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.ensemble_atomic_energies_dict[tag],
                    dtype=self.dtype,
                    device=self.device,
                )

            logging.info(f"Trajectory worker {idx} is training.")
            # we train only for some epochs before we move to the next worker which may be running MD
            # all workers train on the same models with the respective training settings for
            # each ensemble member

            for _ in range(self.intermediate_epochs):
                for tag, model in self.ensemble.items():

                    if self.ensemble_reset_opt[tag]:
                        #TODO: reduce this to resetting the optimizer; reset EMA also????
                        logging.info(f'Resetting training setup for model {tag}.')
                        self.training_setups[tag] = setup_mace_training(
                                        settings=self.mace_settings,
                                        model=self.ensemble[tag],
                                        tag=tag,
                                        )
                        self.ensemble_reset_opt[tag] = False

                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_mace_sets[tag]["train_loader"],
                        loss_fn=self.training_setups[tag]["loss_fn"],
                        optimizer=self.training_setups[tag]["optimizer"],
                        lr_scheduler=self.training_setups[tag]['lr_scheduler'] if self.use_scheduler else None,  # no scheduler used here
                        epoch=self.trajectory_epochs[idx],
                        start_epoch=None,
                        valid_loss=None,
                        logger=logger,
                        device=self.training_setups[tag]["device"],
                        max_grad_norm=self.training_setups[tag]["max_grad_norm"],
                        output_args=self.training_setups[tag]["output_args"],
                        ema=self.training_setups[tag]["ema"],
                    )
                self.total_epoch += 1
                

                    
                if (
                    self.trajectory_epochs[idx] % self.valid_skip == 0
                    or self.trajectory_epochs[idx] == self.max_epochs_worker - 1
                ):
                    ensemble_valid_loss, valid_loss, metrics = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.training_setups,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=self.trajectory_epochs[idx],
                    )
                    if self.analysis:
                        self.collect_losses["epoch"].append(self.total_epoch)
                        self.collect_losses['avg_losses'].append(valid_loss)
                        self.collect_losses['ensemble_losses'].append(ensemble_valid_loss)

                    self.current_valid_error = metrics["mae_f"]

                    for tag in ensemble_valid_loss.keys():
                        if ensemble_valid_loss[tag] < self.ensemble_best_valid[tag]:
                            self.ensemble_best_valid[tag] = ensemble_valid_loss[tag]
                        else:
                            self.ensemble_no_improvement[tag] += 1
                        
                        if self.ensemble_no_improvement[tag] > self.max_epochs_worker:
                            logging.info(
                                f"No improvements for {self.max_epochs_worker} epochs "
                                f"(maximum epochs per worker) at ensemble member {tag}."
                            )
                            self.ensemble_reset_opt[tag] = True
                            self.ensemble_no_improvement[tag] = 0
                        
                    
                    save_checkpoint(
                        checkpoint_handler=self.training_setups[tag][
                            "checkpoint_handler"
                        ],
                        training_setup=self.training_setups[tag],
                        model=model,
                        epoch=self.trajectory_epochs[idx],
                        keep_last=True,
                    )
                # to here, can be made into a class
                #############
                self.trajectory_epochs[idx] += 1
        # update calculator
        MPI.COMM_WORLD.Barrier()
        self.current_valid_error = MPI.COMM_WORLD.bcast(self.current_valid_error, root=0)
        MPI.COMM_WORLD.Barrier()
        if RANK == 0:
            for trajectory in self.trajectories.values():
                trajectory.calc.models = [self.ensemble[tag] for tag in self.ensemble.keys()]
                # TH: do we need this? could make it faster
                #for model in trajectory.calc.models:
                    #for param in model.parameters():
                        #param.requires_grad = False

        MPI.COMM_WORLD.Barrier()
        self.total_epoch = MPI.COMM_WORLD.bcast(self.total_epoch, root=0)
        self.trajectory_epochs[idx] = MPI.COMM_WORLD.bcast(self.trajectory_epochs[idx], root=0)
        MPI.COMM_WORLD.Barrier()

        if self.trajectory_epochs[idx] == self.max_epochs_worker:
            self.trajectory_training[idx] = "running"
            self.num_workers_training -= 1
            self.trajectory_epochs[idx] = 0
            if RANK == 0:
                logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

    def running_task(
            self,
            idx: int
            ):
        """
        Runs the molecular dynamics simulation using the MLFF and
        checks the uncertainty. If the uncertainty is above the threshold
        the point is calculated using FHI aims and sent to the waiting task.

        Args:
            idx (int): Index of the trajectory worker.

        """

        current_MD_step = self.trajectory_MD_steps[idx]
        
        # kill the worker if the maximum number of MD steps is reached
        if (
            current_MD_step > self.max_MD_steps
            and self.trajectory_training[idx] == "running"
        ):
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} reached maximum MD steps and is killed."
                )
            self.num_MD_limits_reached += 1
            self.trajectory_training[idx] = "killed"
            return "killed"

        else:
            # TODO:
            # ideally we would first check the uncertainty, then optionally 
            # calculate the aims forces and use them to propagate
            # currently the mace forces are used even if the uncertainty is too high
            # but ase is weird and i don't want to change it so whatever. when we have our own 
            # MD engine we can adress this.
            if RANK == 0:
                self.md_drivers[idx].step()

            #MPI.COMM_WORLD.Barrier()
            #self.trajectories[idx] = MPI.COMM_WORLD.bcast(self.trajectories[idx], root=0)
            #MPI.COMM_WORLD.Barrier()

            self.trajectory_MD_steps[idx] += 1
            if current_MD_step % self.skip_step == 0:
                if RANK == 0:
                    logging.info(
                        f"Trajectory worker {idx} at MD step {current_MD_step}."
                    )
                    
                    self.point = self.trajectories[idx].copy()
                    prediction = self.trajectories[idx].calc.results["forces_comm"]
                
                    uncertainty = max_sd_2(prediction)
                    # compute moving average of uncertainty
                    self.uncertainties.append(uncertainty)
                    # TODO: Remove hardcode, make a function out of this
                    # limit the history to 400
                    if len(self.uncertainties) > 400:
                        self.uncertainties = self.uncertainties[-400:]
                    if len(self.uncertainties) > 10:
                        mov_avg_uncert = np.mean(self.uncertainties)
                        self.threshold = mov_avg_uncert * (1.0 + self.c_x)
                        
                        if self.analysis:
                            self.collect_thresholds[
                                idx
                                ].append(self.threshold)
            
                if RANK != 0:
                    uncertainty = None
                    prediction = None
                    self.point = None
                    
                MPI.COMM_WORLD.Barrier()
                self.threshold = MPI.COMM_WORLD.bcast(self.threshold, root=0)
                self.point = MPI.COMM_WORLD.bcast(self.point, root=0)
                uncertainty = MPI.COMM_WORLD.bcast(uncertainty, root=0)
                prediction = MPI.COMM_WORLD.bcast(prediction, root=0)
                MPI.COMM_WORLD.Barrier()
        
                
                if uncertainty > self.threshold:
                    if RANK == 0:
                        logging.info(
                            f"Uncertainty of point is beyond threshold {np.round(self.threshold,3)} at worker {idx}: {round(uncertainty.item(),3)}."
                        )
                    # for analysis
                    self.t_intervals[idx].append(current_MD_step)
                    # at the moment the process waits for the calculation to finish
                    # ideally it should calculate in the background and the other
                    # workers sample/train in the meantime
                    
                    MPI.COMM_WORLD.Barrier()
                    self.aims_calculator.calculate(self.point, properties=["energy","forces"])
                    if RANK == 0:
                        self.point.info['energy'] = self.aims_calculator.results['energy']
                        self.point.arrays['forces'] = self.aims_calculator.results['forces']
                        self.mace_point = create_mace_dataset(
                            data=[self.point],
                            z_table=self.z_table,
                            seed=None,
                            r_max=self.r_max,
                        )
                    # it sends the job and does not wait for the result but
                    # continues with the next worker. only if the job is done
                    # the worker is set to training mode
                    self.trajectory_training[idx] = "waiting"
                    self.num_workers_waiting += 1
                    
                    MPI.COMM_WORLD.Barrier()
                    self.waiting_task(idx)

                    
                    if RANK == 0:
                        logging.info(
                            f"Trajectory worker {idx} is waiting for job to finish."
                        )

            if (
                current_MD_step % self.sanity_skip == 0
            ):  
                # TODO:
                # should not be static but increase with time, based on how many uninterrupted
                #  MD steps have been taken or if all workes are running and currently
                # we are not doing anything with this. Maybe check correlation between real error
                # and uncertainty?
                if RANK == 0:
                    logging.info(f"Trajectory worker {idx} doing a sanity check.")
                
                if current_MD_step % self.skip_step == 0:
                    sanity_prediction = prediction
                else:
                    if RANK == 0:
                        sanity_prediction = ensemble_prediction(
                            models=list(self.ensemble.values()),
                            atoms_list=[self.point],
                            device=self.device,
                            dtype=self.mace_settings["GENERAL"]["default_dtype"],
                        )
                    MPI.COMM_WORLD.Barrier()
                    sanity_prediction = MPI.COMM_WORLD.bcast(sanity_prediction, root=0)
                    MPI.COMM_WORLD.Barrier()
                    
                #TODO: sometimes already calculated above so we should not calculate it again
                MPI.COMM_WORLD.Barrier()
                self.aims_calculator.calculate(self.point, properties=["energy","forces"])
                MPI.COMM_WORLD.Barrier()
                (
                    atom_wise_uncertainty,
                    uncertainty_via_max,
                    uncertainty_via_avg,
                    atom_wise_error,
                    max_error,
                    mean_error
                ) = self.sanity_check(
                    sanity_prediction=sanity_prediction,
                    true_forces = self.aims_calculator.results['forces']
                )
                self.sanity_checks[idx]['atom_wise_uncertainty'].append(atom_wise_uncertainty)
                self.sanity_checks[idx]['uncertainty_via_max'].append(uncertainty_via_max)
                self.sanity_checks[idx]['uncertainty_via_avg'].append(uncertainty_via_avg)
                self.sanity_checks[idx]['atom_wise_error'].append(atom_wise_error)
                self.sanity_checks[idx]['max_error'].append(max_error)
                self.sanity_checks[idx]['mean_error'].append(mean_error)
                self.sanity_checks[idx]['threshold'].append(self.threshold)
                
                self.check += 1

    def run(self):
        """
        Main function to run the active learning procedure. Initializes variables and
        controls the workers tasks.
        """

        if RANK == 0:
            logging.info("Starting active learning procedure.")
            self.ensemble_reset_opt = {
                tag: False for tag in self.ensemble.keys()
            }
            self.ensemble_no_improvement = {
                tag: 0 for tag in self.ensemble.keys()
            }
            self.ensemble_best_valid = {
                tag: np.inf for tag in self.ensemble.keys()
            }
            
        self.current_valid_error = np.inf
        self.threshold = np.inf
        self.point_added = 0  # counts how many points have been added to the training set to decide when to add a point to the validation set
        self.num_MD_limits_reached = 0
        self.num_workers_training = 0  # maybe useful lateron to give CPU some to work if all workers are training
        self.num_workers_waiting = 0
        self.total_epoch = 0
        self.check = 0

        MPI.COMM_WORLD.Barrier()
        
        while True: 
            for trajectory_idx, _ in enumerate(self.trajectories):

                if self.trajectory_training[trajectory_idx] == "waiting":

                    set_limit = self.waiting_task(trajectory_idx)
                    if set_limit: # stops the process if the maximum dataset size is reached
                        break

                if (
                    self.trajectory_training[trajectory_idx] == "training"
                ):  # and training_job: # the idea is to let a worker train only if new points
                    # have been added. e.g. it can happen that one worker is
                    # beyond its MD limit but there is no new point that has been added

                    self.training_task(trajectory_idx)

                if self.trajectory_training[trajectory_idx] == "running":

                    self.running_task(trajectory_idx)

                if (
                    self.num_workers_training == self.num_trajectories
                ):  
                    if RANK == 0:
                        logging.info(
                            "All workers are in training mode."
                        )  
                
                if self.num_workers_waiting == self.num_trajectories:
                    if RANK == 0:
                        logging.info("All workers are waiting for jobs to finish.")
            
            if self.num_MD_limits_reached == self.num_trajectories:
                if RANK == 0:
                    logging.info(
                        "All trajectories reached maximum MD steps. Training until convergence."
                    )
                break
            
            if self.train_dataset_len >= self.max_set_size:
                if RANK == 0:
                    logging.info(
                        "Maximum size of training set reached. Training until convergence."
                    )
                break
            
            
            if self.current_valid_error < self.desired_accuracy:
                if RANK == 0:
                    logging.info(
                        "Desired accuracy reached. Training until convergence."
                    )
                break

            if RANK == 0:
                if self.train_dataset_len % self.save_data_len_interval == 0:
                    save_datasets(
                        ensemble=self.ensemble,
                        ensemble_ase_sets=self.ensemble_ase_sets,
                        path=self.dataset_dir / "final",
                    )
                if self.analysis:
                    np.savez(
                        "analysis/sanity_checks.npz",
                        self.sanity_checks
                    )
                    np.savez(
                        "analysis/t_intervals.npz",
                        self.t_intervals
                    )
                    np.savez(
                        "analysis/al_losses.npz",
                        **self.collect_losses
                    )
                    np.savez(
                        "analysis/thresholds.npz",
                        self.collect_thresholds
                    )
       
        
        # turn keys which are ints into strings
        # save the datasets and the intervals for analysis
        if RANK == 0:
            save_datasets(
                ensemble=self.ensemble,
                ensemble_ase_sets=self.ensemble_ase_sets,
                path=self.dataset_dir / "final",
            )

    def converge(self):
        """
        Converges the ensemble on the acquired dataset. Trains the ensemble members
        until the validation loss does not improve anymore.
        """
        if RANK == 0:
            logging.info("Converging ensemble on acquired dataset.")
            for _, (tag, model) in enumerate(self.ensemble.items()):
                train_set = create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["train"],
                    z_table=self.z_table,
                    seed=self.seeds[tag],
                    r_max=self.r_max,
                )
                valid_set = create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["valid"],
                    z_table=self.z_table,
                    seed=self.seeds[tag],
                    r_max=self.r_max,
                )
                (
                    self.ensemble_ase_sets[tag]["train_loader"],
                    self.ensemble_ase_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    train_set,
                    valid_set,
                    self.set_batch_size,
                    self.set_valid_batch_size,
                )

                update_model_auxiliaries(
                    model=model,
                    mace_sets=self.ensemble_mace_sets[tag],
                    atomic_energies=self.ensemble_atomic_energies[tag],
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.ensemble_atomic_energies_dict[tag],
                    dtype=self.dtype,
                    device=self.device,
                )
                training_setups = {}
            # resetting optimizer and scheduler
            for tag in self.ensemble.keys():
                training_setups[tag] = setup_mace_training(
                    settings=self.mace_settings,
                    model=self.ensemble[tag],
                    tag=tag,
                )
            best_valid_loss = np.inf
            epoch = 0

            no_improvement = 0
            ensemble_valid_losses = {tag: np.inf for tag in self.ensemble.keys()}
            for j in range(self.max_final_epochs):
                # ensemble_loss = 0
                for tag, model in self.ensemble.items():
                    training_setup = training_setups[tag]
                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                        loss_fn=training_setup["loss_fn"],
                        optimizer=training_setup["optimizer"],
                        lr_scheduler=training_setup["lr_scheduler"],
                        valid_loss=ensemble_valid_losses[tag],
                        epoch=epoch,
                        start_epoch=0,
                        logger=logger,
                        device=training_setup["device"],
                        max_grad_norm=training_setup["max_grad_norm"],
                        output_args=training_setup["output_args"],
                        ema=training_setup["ema"],
                    )
                    # ensemble_loss += loss
                # ensemble_loss /= len(ensemble)

                if epoch % self.valid_skip == 0 or epoch == self.max_final_epochs - 1:
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        _,
                    ) = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=training_setups,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=epoch,
                    )

                    if best_valid_loss > valid_loss and (best_valid_loss - valid_loss) > self.margin:
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        no_improvement = 0
                        for tag, model in self.ensemble.items():
                            torch.save(
                                model,
                                Path(self.mace_settings["GENERAL"]["model_dir"])
                                / (tag + ".model"),
                            )
                            save_checkpoint(
                                checkpoint_handler=training_setups[tag][
                                    "checkpoint_handler_convergence"
                                ],
                                training_setup=training_setups[tag],
                                model=model,
                                epoch=epoch,
                                keep_last=True,
                            )
                    else:
                        no_improvement += 1

                epoch += 1
                if no_improvement > self.patience:
                    logging.info(
                        f"No improvements for {self.patience} epochs. Training converged. Best model (Epoch {best_epoch}) based on validation loss saved."
                    )
                    break
                if j == self.max_final_epochs - 1:
                    logging.info(
                        f"Maximum number of epochs reached. Best model (Epoch {best_epoch}) based on validation loss saved."
                    )
        MPI.COMM_WORLD.Barrier()    

# TODO: This is not done yet
class StandardMACEEnsembleProcedure:
    """
    Just a simple Class to train ensembles from scratch using existing
    dataset in order to compare the continuous learning procedure with
    learning from scratch.
    """
    def __init__(
        self, 
        mace_settings: dict,
        active_learning_settings: dict,
        train_set_dir: str = "data/final/training",
        valid_set_dir: str = "data/final/validation",
        ) -> None:
        self.handle_mace_settings(mace_settings)
        logging.basicConfig(
            filename="standard_ensemble.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level=self.mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=self.mace_settings["GENERAL"]["log_dir"],
        )
        self.create_folders()
        self.z = Z_from_geometry_in()

        if self.atomic_energies_dict is None:
            self.atomic_energies_dict = {
                z: 0 for z in np.unique(self.z)
            }
            self.update_atomic_energies = True

        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
        )

        np.random.seed(self.seed)
        #random.seed(self.seed)
        self.ensemble_seeds = np.random.randint(
            0, 1000, size=active_learning_settings["ACTIVE_LEARNING"]["ensemble_size"]
        )
    
        (
        self.seeds_tags_dict,
        self.ensemble,
        self.training_setups,
        ) = setup_ensemble_dicts(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            atomic_energies_dict=self.atomic_energies_dict,
            save_seeds_tags_dict=False
        )
        for tag in self.ensemble.keys():
            train_set = read(train_set_dir + f"/train_set_{tag}.xyz",index=":")
            valid_set = read(valid_set_dir + f"/valid_set_{tag}.xyz",index=":")
            logging.info(
                f"Training set {tag} contains {len(train_set)} structures."
            )
            self.ensemble_ase_sets = {
                tag: {"train": train_set, "valid": valid_set} for tag in self.ensemble.keys()
            }
        self.ensemble_mace_sets = {
            tag: {
                    "train": create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["train"],
                    z_table=self.z_table,
                    seed=self.seeds_tags_dict[tag],
                    r_max=self.r_max
                    ),
                    "valid": create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["valid"],
                    z_table=self.z_table,
                    seed=self.seeds_tags_dict[tag],
                    r_max=self.r_max
                    )
                } for tag in self.ensemble.keys()
        }
            
    def train(self):
        
        for _, (tag, model) in enumerate(self.ensemble.items()):
            (
                self.ensemble_mace_sets[tag]["train_loader"],
                self.ensemble_mace_sets[tag]["valid_loader"],
            ) = create_dataloader(
                self.ensemble_mace_sets[tag]["train"],
                self.ensemble_mace_sets[tag]["valid"],
                self.set_batch_size,
                self.set_valid_batch_size,
            )

            update_model_auxiliaries(
                model=model,
                mace_sets=self.ensemble_mace_sets[tag],
                atomic_energies=self.atomic_energies,
                scaling=self.scaling,
                update_atomic_energies=self.update_atomic_energies,
                z_table=self.z_table,
                atomic_energies_dict=self.atomic_energies_dict,
                dtype=self.dtype,
                device=self.device,
            )
            
        best_valid_loss = np.inf
        epoch = 0
        no_improvement = 0
        ensemble_valid_losses = {tag: np.inf for tag in self.ensemble.keys()}
        for j in range(self.max_num_epochs):
            # ensemble_loss = 0
            for tag, model in self.ensemble.items():
                training_setup = self.training_setups[tag]
                logger = tools.MetricsLogger(
                    directory=self.mace_settings["GENERAL"]["results_dir"],
                    tag=tag + "_standard_train",
                )
                train_epoch(
                    model=model,
                    train_loader=self.ensemble_mace_sets[tag]["train_loader"],
                    loss_fn=training_setup["loss_fn"],
                    optimizer=training_setup["optimizer"],
                    lr_scheduler=training_setup["lr_scheduler"],
                    valid_loss=ensemble_valid_losses[tag],
                    epoch=epoch,
                    start_epoch=0,
                    logger=logger,
                    device=training_setup["device"],
                    max_grad_norm=training_setup["max_grad_norm"],
                    output_args=training_setup["output_args"],
                    ema=training_setup["ema"],
                )
                # ensemble_loss += loss
            # ensemble_loss /= len(ensemble)

            if epoch % self.eval_interval == 0 or epoch == self.max_num_epochs - 1:
                (
                    ensemble_valid_losses,
                    valid_loss,
                    _,
                ) = validate_epoch_ensemble(
                    ensemble=self.ensemble,
                    training_setups=self.training_setups,
                    ensemble_set=self.ensemble_mace_sets,
                    logger=logger,
                    log_errors=self.mace_settings["MISC"]["error_table"],
                    epoch=epoch,
                )
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    no_improvement = 0
                    for tag, model in self.ensemble.items():
                        torch.save(
                            model,
                            self.standard_model_dir / (tag + ".model"),
                        )
                        save_checkpoint(
                            checkpoint_handler=self.training_setups[tag][
                                "checkpoint_handler_convergence"
                            ],
                            training_setup=self.training_setups[tag],
                            model=model,
                            epoch=epoch,
                            keep_last=True,
                        )
                else:
                    no_improvement += 1

            epoch += 1
            if no_improvement > self.patience:
                logging.info(
                    f"No improvements for {self.patience} epochs. Training converged. Best model (Epoch {best_epoch}) based on validation loss saved."
                )
                break
            if j == self.max_num_epochs - 1:
                logging.info(
                    f"Maximum number of epochs reached. Best model (Epoch {best_epoch}) based on validation loss saved."
                )

    def handle_mace_settings(self, mace_settings: dict):
        self.mace_settings = mace_settings
        self.seed = self.mace_settings["GENERAL"]["seed"]
        self.r_max = self.mace_settings["ARCHITECTURE"]["r_max"]
        self.set_batch_size = self.mace_settings["TRAINING"]["batch_size"]
        self.set_valid_batch_size = self.mace_settings["TRAINING"][
            "valid_batch_size"
        ]
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.model_dir = self.mace_settings["GENERAL"]["model_dir"]
        self.eval_interval = self.mace_settings['TRAINING']['eval_interval']
        self.max_num_epochs = self.mace_settings['TRAINING']['max_num_epochs']
        self.patience = self.mace_settings['TRAINING']['patience']
        self.device = self.mace_settings["MISC"]["device"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
        self.atomic_energies_dict = self.mace_settings["ARCHITECTURE"].get("atomic_energies_dict", None)

    
    def create_folders(self):
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.standard_model_dir = Path(self.model_dir) / "standard"
        self.standard_model_dir.mkdir(
            parents=True, exist_ok=True
        )

    def setup_mace_calc(self):
        """
        Loads the models of the existing ensemble and creates the MACE calculator.
        """
        model_paths = list_files_in_directory(Path(self.model_dir) / "standard")
        self.models = [
            torch.load(f=model_path, map_location=self.device) for model_path in model_paths
        ]
        # the calculator needs to be updated consistently see below
        self.mace_calc = MACECalculator(
            models=self.models,
            device=self.device,
            default_dtype=self.dtype)
        
    def sanity_check(
    #TODO: put this somewhere else and its ugly
            self,
            sanity_prediction: np.ndarray,
            true_forces: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the force uncertainty and the maximum 
        force error for a given prediction and true forces.

        Args:
            sanity_prediction (np.ndarray): Ensemble prediction. [n_members, n_points, n_atoms, 3]
            true_forces (np.ndarray): True forces. [n_points, n_atoms, 3]

        Returns:
            tuple[np.ndarray, np.ndarray]: force uncertainty, true force error
        """
        uncertainty_via_max = max_sd_2(sanity_prediction)
        uncertainty_via_avg = avg_sd(sanity_prediction)
        atom_wise_uncertainty = atom_wise_sd(sanity_prediction)
        mean_sanity_prediction = sanity_prediction.mean(0).squeeze()
        difference = true_forces - mean_sanity_prediction
        diff_sq = difference**2
        diff_sq_mean = np.mean(diff_sq, axis=-1)
        
        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
        atom_wise_error = atom_wise_f_error(mean_sanity_prediction, true_forces)

        return {
            "atom_wise_uncertainty": atom_wise_uncertainty,
            "uncertainty_via_max": uncertainty_via_max,
            "uncertainty_via_avg": uncertainty_via_avg,
            "max_error": max_error,
            "mean_error": mean_error,
            "atom_wise_error": atom_wise_error
        }

    def test(
        self,
        path_to_ds: str,
    ):
        """
        Tests the ensemble on a dataset.

        Args:
            path_to_ds (str): Path to the dataset.
        """
        test_set = read(path_to_ds, index=":")
        test_mace_ds = create_mace_dataset(
            data=test_set,
            z_table=self.z_table,
            seed=None,
            r_max=self.r_max,
        )
        logging.info("Created DS.")
        ensemble_energies, ensemble_forces = ensemble_prediction_v2(
            models=list(self.ensemble.values()),
            mace_ds=test_mace_ds,
            device=self.device,
            dtype=self.dtype,
            return_energies=True
        )
        reference_energies = np.array([atoms.info["energy"] for atoms in test_set])
        reference_forces = np.array([atoms.arrays["forces"] for atoms in test_set])
        check = self.sanity_check(ensemble_forces, reference_forces)
        np.savez("check.npz", **check)
