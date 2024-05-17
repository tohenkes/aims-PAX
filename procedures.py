import sys
import os
from pathlib import Path
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
from mace import tools
from FHI_AL.utilities import (
    create_dataloader,
    ensemble_training_setups,
    ensemble_from_folder,
    setup_ensemble_dicts,
    update_datasets,
    update_avg_neighs_shifts_scale,
    save_checkpoint,
    save_datasets,
    pre_trajectories_from_folder,
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
    create_mace_dataset,
    ensemble_prediction,
    setup_mace_training,
    max_sd_2
)
from FHI_AL.train_epoch_mace import train_epoch, validate_epoch_ensemble
from ase.io import read
import logging
import random


class PrepareInitialDatasetProcedure:
    def __init__(
        self,
        mace_settings,
        al_settings,
        path_to_trajectory: str = None,
        ensemble_seeds: np.array = None
    ):

        self.handle_mace_settings(mace_settings)
        self.handle_al_settings(al_settings)
        self.create_folders()
        self.get_atomic_energies()

        if ensemble_seeds is not None:
            self.ensemble_size = len(ensemble_seeds)
            self.ensemble_seeds = ensemble_seeds
        else:
            self.ensemble_seeds = np.random.randint(
                0, 1000, size=self.ensemble_size
            )
        (
            self.seeds_tags_dict,
            self.ensemble,
            self.training_setups,
        ) = setup_ensemble_dicts(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            al_settings=self.al_settings,
            atomic_energies_dict=self.atomic_energies_dict,
        )
        self.ensemble_mace_sets, self.ensemble_ase_sets = (
            {tag: {"train": [], "valid": []} for tag in self.ensemble.keys()},
            {tag: {"train": [], "valid": []} for tag in self.ensemble.keys()},
        )

        if path_to_trajectory:
            self.simulate_trajectory(path_to_trajectory)

        logging.basicConfig(
            filename="initial_dataset.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level=self.mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=self.mace_settings["GENERAL"]["log_dir"],
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

    def handle_al_settings(self, al_settings):

        self.al_settings = al_settings
        self.ensemble_size = self.al_settings["ensemble_size"]
        self.desired_acc = self.al_settings["desired_acc"]
        self.lamb = self.al_settings["lambda"]
        self.n_samples = self.al_settings["n_samples"]
        self.max_initial_epochs = self.al_settings["max_initial_epochs"]
        self.max_final_epochs = self.al_settings["max_final_epochs"]
        self.valid_skip = self.al_settings["valid_skip"]
        self.skip_step = self.al_settings["skip_step"]
        self.intermediate_epochs = self.al_settings["intermediate_epochs"]
        self.initial_valid_ratio = self.al_settings["initial_valid_ratio"]

    def create_folders(self):
        self.dataset_dir = Path(self.al_settings["dataset_dir"])
        (self.dataset_dir / "initial" / "training").mkdir(
            parents=True, exist_ok=True
        )
        (self.dataset_dir / "initial" / "validation").mkdir(
            parents=True, exist_ok=True
        )
        os.makedirs("model", exist_ok=True)

    def get_atomic_energies(self):
        #TODO: remove hardocde!!!
        self.atomic_energies_dict = {1: -12.482766945, 6: -1027.170068545}
        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
        )

    def simulate_trajectory(self, path_to_trajectory: str):
        self.trajectory = read(path_to_trajectory, index=":")

class InitalDatasetProcedure(PrepareInitialDatasetProcedure):
    def run(self):

        current_valid = np.inf
        step = 0
        epoch = 0
        self.point_added = 0
        while (
            self.desired_acc <= (current_valid * self.lamb**-1)
            and epoch < self.max_initial_epochs
        ):

            logging.info(f"Sampling new points at step {step}.")
            sampled_points = self.trajectory[:: self.skip_step][
                self.n_samples
                * self.ensemble_size
                * step : self.n_samples
                * self.ensemble_size
                * (step + 1)
            ]
            random.shuffle(sampled_points)
            self.point_added += len(sampled_points)

            for number, (tag, model) in enumerate(self.ensemble.items()):

                member_points = sampled_points[
                    self.n_samples * number : self.n_samples * (number + 1)
                ]

                (
                    self.ensemble_ase_sets[tag],
                    self.ensemble_mace_sets[tag],
                ) = update_datasets(
                    new_points=member_points,
                    mace_set=self.ensemble_mace_sets[tag],
                    ase_set=self.ensemble_ase_sets[tag],
                    valid_split=self.initial_valid_ratio,
                    z_table=self.z_table,
                    seed=self.seed,
                    r_max=self.r_max,
                )

                batch_size = (
                    1
                    if len(self.ensemble_mace_sets[tag]["train"])
                    < self.set_batch_size
                    else self.set_batch_size
                )
                valid_batch_size = (
                    1
                    if len(self.ensemble_mace_sets[tag]["valid"])
                    < self.set_valid_batch_size
                    else self.set_valid_batch_size
                )

                (
                    self.ensemble_mace_sets[tag]["train_loader"],
                    self.ensemble_mace_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    self.ensemble_mace_sets[tag]["train"],
                    self.ensemble_mace_sets[tag]["valid"],
                    batch_size,
                    valid_batch_size,
                )

                update_avg_neighs_shifts_scale(
                    model=model,
                    train_loader=self.ensemble_mace_sets[tag]["train_loader"],
                    atomic_energies=self.atomic_energies,
                    scaling=self.scaling,
                )

                step += 1
                logging.info(
                    f"Training set size for '{tag}': {len(self.ensemble_mace_sets[tag]['train'])}; Validation set size: {len(self.ensemble_mace_sets[tag]['valid'])}."
                )

            logging.info("Training.")
            ensemble_valid_losses = {
                tag: np.inf for tag in self.ensemble.keys()
            }
            for i in range(self.intermediate_epochs):
                for tag, model in self.ensemble.items():

                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_mace_sets[tag][
                            "train_loader"
                        ],
                        loss_fn=self.training_setups[tag]["loss_fn"],
                        optimizer=self.training_setups[tag]["optimizer"],
                        lr_scheduler=self.training_setups[tag]["lr_scheduler"],
                        epoch=epoch,
                        start_epoch=0,
                        valid_loss=ensemble_valid_losses[tag],
                        logger=logger,
                        device=self.training_setups[tag]["device"],
                        max_grad_norm=self.training_setups[tag][
                            "max_grad_norm"
                        ],
                        output_args=self.training_setups[tag]["output_args"],
                        ema=self.training_setups[tag]["ema"],
                    )

                if (
                    epoch % self.valid_skip == 0
                    or (epoch + 1) % self.valid_skip == 0
                ):
                    (
                        ensemble_valid_losses,
                        _,
                        metrics,
                    ) = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        ema=self.training_setups[tag]["ema"],
                        loss_fn=self.training_setups[tag]["loss_fn"],
                        valid_loader=self.ensemble_mace_sets[tag][
                            "valid_loader"
                        ],
                        output_args=self.training_setups[tag]["output_args"],
                        device=self.training_setups[tag]["device"],
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=epoch,
                    )
                    current_valid = metrics["mae_f"]

                    if self.desired_acc >= (current_valid * self.lamb**-1):
                        logging.info(
                            f"Accuracy criterion reached at step {step}."
                        )
                        logging.info(
                            f"Criterion: {self.desired_acc * self.lamb}; Current accuracy: {current_valid}."
                        )
                        for tag, model in self.ensemble.items():
                            torch.save(
                                model,
                                Path(
                                    self.mace_settings["GENERAL"]["model_dir"]
                                )
                                / (tag + ".model"),
                            )

                            save_checkpoint(
                                checkpoint_handler=self.training_setups[tag][
                                    "checkpoint_handler"
                                ],
                                training_setup=self.training_setups[tag],
                                model=model,
                                epoch=epoch,
                                keep_last=True,
                            )
                        break
                    else:
                        for tag, model in self.ensemble.items():
                            save_checkpoint(
                                checkpoint_handler=self.training_setups[tag][
                                    "checkpoint_handler"
                                ],
                                training_setup=self.training_setups[tag],
                                model=model,
                                epoch=epoch,
                                keep_last=False,
                            )
                epoch += 1

            if (
                epoch == self.max_initial_epochs
            ):  # TODO: change to a different variable (shares with al-algo right now)
                logging.info(f"Maximum number of epochs reached.")
        save_datasets(
            self.ensemble,
            self.ensemble_ase_sets,
            path=self.dataset_dir / "initial",
            initial=True,
        )

    def converge(self):

        for _, (tag, model) in enumerate(self.ensemble.items()):

            (
                self.ensemble_ase_sets[tag]["train_loader"],
                self.ensemble_ase_sets[tag]["valid_loader"],
            ) = create_dataloader(
                self.ensemble_mace_sets[tag]["train"],
                self.ensemble_mace_sets[tag]["valid"],
                self.set_batch_size,
                self.set_valid_batch_size,
            )

            update_avg_neighs_shifts_scale(
                model=model,
                train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                atomic_energies=self.atomic_energies,
                scaling=self.scaling,
            )

        # TODO: reset or not?
        self.training_setups_convergence = {}
        for tag in self.ensemble.keys():
            self.training_setups_convergence[tag] = setup_mace_training(
                settings=self.mace_settings,
                model=self.ensemble[tag],
                tag=tag,
            )
        best_valid_loss = np.inf
        epoch = 0
        patience = self.al_settings["patience"]
        no_improvement = 0
        ensemble_valid_losses = {tag: np.inf for tag in self.ensemble.keys()}
        for j in range(self.max_final_epochs):
            # ensemble_loss = 0
            for tag, model in self.ensemble.items():
                logger = tools.MetricsLogger(
                    directory=self.mace_settings["GENERAL"]["results_dir"],
                    tag=tag + "_train",
                )
                train_epoch(
                    model=model,
                    train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                    loss_fn=self.training_setups_convergence[tag]["loss_fn"],
                    optimizer=self.training_setups_convergence[tag][
                        "optimizer"
                    ],
                    lr_scheduler=self.training_setups_convergence[tag][
                        "lr_scheduler"
                    ],
                    valid_loss=ensemble_valid_losses[tag],
                    epoch=epoch,
                    start_epoch=0,
                    logger=logger,
                    device=self.training_setups_convergence[tag]["device"],
                    max_grad_norm=self.training_setups_convergence[tag][
                        "max_grad_norm"
                    ],
                    output_args=self.training_setups_convergence[tag][
                        "output_args"
                    ],
                    ema=self.training_setups_convergence[tag]["ema"],
                )
                # ensemble_loss += loss
            # ensemble_loss /= len(ensemble)

            if (
                epoch % self.valid_skip == 0
                or epoch == self.max_final_epochs - 1
            ):
                ensemble_valid_losses, valid_loss, _ = validate_epoch_ensemble(
                    ensemble=self.ensemble,
                    ema=self.training_setups_convergence[tag]["ema"],
                    loss_fn=self.training_setups_convergence[tag]["loss_fn"],
                    valid_loader=self.ensemble_ase_sets[tag]["valid_loader"],
                    output_args=self.training_setups_convergence[tag][
                        "output_args"
                    ],
                    device=self.training_setups_convergence[tag]["device"],
                    logger=logger,
                    log_errors=self.mace_settings["MISC"]["error_table"],
                    epoch=epoch,
                )
                if best_valid_loss > valid_loss and (best_valid_loss - valid_loss) > 0.01:
                    best_valid_loss = valid_loss
                    no_improvement = 0
                    for tag, model in self.ensemble.items():
                        torch.save(
                            model,
                            Path(self.mace_settings["GENERAL"]["model_dir"])
                            / (tag + ".model"),
                        )
                        save_checkpoint(
                            checkpoint_handler=self.training_setups_convergence[
                                tag
                            ][
                                "checkpoint_handler_convergence"
                            ],
                            training_setup=self.training_setups_convergence[
                                tag
                            ],
                            model=model,
                            epoch=epoch,
                            keep_last=True,
                        )
                else:
                    no_improvement += 1

            epoch += 1
            if no_improvement > patience:
                logging.info(
                    f"No improvements for {patience} epochs. Training converged. Best model based on validation loss saved"
                )
                break
            if j == self.max_final_epochs - 1:
                logging.info(
                    "Maximum number of epochs reached. Best model based on validation loss saved"
                )

class PrepareALProcedure:
    def __init__(
        self,
        mace_settings,
        al_settings,
        path_to_trajectories: str = None,
        analysis: bool = False,
    ) -> None:
        self.handle_al_settings(al_settings)
        self.handle_mace_settings(mace_settings)
        self.create_folders()
        self.get_atomic_energies()
        self.seeds = dict(
            np.load(
                self.dataset_dir / "seeds_tags_dict.npz", allow_pickle=True
            )
        )
        self.use_scheduler = False
        logging.basicConfig(
            filename="AL.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level=self.mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=self.mace_settings["GENERAL"]["log_dir"],
        )

        logging.info("Initializing active learning procedure.")
        self.ensemble = ensemble_from_folder(
            path_to_models="./model",
            device=self.device,
        )
        self.training_setups = ensemble_training_setups(
            ensemble=self.ensemble,
            mace_settings=self.mace_settings,
        )
        if path_to_trajectories is not None:
            self.simulate_trajectories(path_to_trajectories)

        logging.info("Loading initial datasets.")
        self.ensemble_ase_sets = load_ensemble_sets_from_folder(
            ensemble=self.ensemble,
            path_to_folder=al_settings["dataset_dir"] + "/initial",
        )
        self.ensemble_mace_sets = ase_to_mace_ensemble_sets(
            ensemble_ase_sets=self.ensemble_ase_sets,
            z_table=self.z_table,
            r_max=self.r_max,
            seed=self.seeds,
        )

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
        self.uncertainties = []  # for moving average
        self.t_intervals = {
            trajectory: [] for trajectory in range(self.num_trajectories)
        }
        self.sanity_checks = {
            trajectory: [] for trajectory in range(self.num_trajectories)
        }
        self.sanity_checks_valid = {}
        self.analysis = analysis

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
    
    def handle_al_settings(self, al_settings):
        self.al_settings = al_settings
        self.max_MD_steps = al_settings["max_MD_steps"]
        self.max_epochs_worker = al_settings["max_epochs_worker"]
        self.max_final_epochs = al_settings["max_final_epochs"]
        self.desired_accuracy = al_settings["desired_acc"]
        self.num_trajectories = al_settings["num_trajectories"]
        self.skip_step = al_settings["skip_step"]
        self.valid_skip = al_settings["valid_skip"]
        self.sanity_skip = al_settings["sanity_skip"]
        self.valid_ratio = al_settings["valid_ratio"]
        self.max_set_size = al_settings["max_set_size"]
        self.num_trajectories = al_settings["num_trajectories"]
        self.c_x = al_settings["c_x"]
        self.intermediate_epochs = al_settings["intermediate_epochs"]
        self.dataset_dir = Path(al_settings["dataset_dir"])
        self.patience = al_settings["patience"]

    def create_folders(self):
        (self.dataset_dir / "final" / "training").mkdir(
            parents=True, exist_ok=True
        )
        (self.dataset_dir / "final" / "validation").mkdir(
            parents=True, exist_ok=True
        )

    def get_atomic_energies(self):
        self.atomic_energies_dict = {1: -12.482766945, 6: -1027.170068545}
        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
        )

    def simulate_trajectories(self, path_to_trajectories: str):
        self.trajectories = pre_trajectories_from_folder(
            path=path_to_trajectories,
            num_trajectories=self.num_trajectories,
        )

class ALProcedure(PrepareALProcedure):
    def sanity_check(self, sanity_prediction):
        sanity_uncertainty = max_sd_2(sanity_prediction)
        mean_sanity_prediction = sanity_prediction.mean(0).squeeze()
        difference = self.point.get_forces() - mean_sanity_prediction
        diff_sq = difference**2
        diff_sq_mean = np.mean(diff_sq, axis=-1)
        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        return sanity_uncertainty, max_error

    def waiting_task(self, idx):
        # if calculation is finished:
        # there is no waiting time here and if we do it sequentially there is not waiting either
        # thus the models directly continue training with the new point which could make quite the difference
        # same with adding a training point to each of the ensemble members which slows down things considerably
        # as we have to wait for enough training points to be acquired
        if self.point_added % self.valid_ratio == 0:
            self.trajectory_training[idx] = "running"
            self.num_workers_waiting -= 1
            logging.info(
                f"Trajectory worker {idx} is adding a point to the validation set."
            )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            for tag in self.ensemble_ase_sets.keys():
                self.ensemble_ase_sets[tag]["valid"] += [self.point]
                self.ensemble_mace_sets[tag]["valid"] += self.mace_point

            if self.analysis:
                sanity_prediction = ensemble_prediction(
                    models=list(self.ensemble.values()),
                    atoms_list=self.ensemble_ase_sets[tag]["valid"],
                    device=self.device,
                    dtype=self.mace_settings["GENERAL"]["default_dtype"],
                )
                self.sanity_check(sanity_prediction)

        else:
            self.trajectory_training[idx] = "training"
            self.num_workers_training += 1
            self.num_workers_waiting -= 1
            logging.info(
                f"Trajectory worker {idx} is adding a point to the training set."
            )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            for tag in self.ensemble_ase_sets.keys():
                self.ensemble_ase_sets[tag]["train"] += [self.point]
                self.ensemble_mace_sets[tag]["train"] += self.mace_point
            if len(self.ensemble_ase_sets[tag]["train"]) > self.max_set_size:
                return True
            logging.info(
                f"Size of the training and validation set: {len(self.ensemble_ase_sets[tag]['train'])}, {len(self.ensemble_ase_sets[tag]['valid'])}."
            )
        self.point_added += 1

    def training_task(self, idx):
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
            update_avg_neighs_shifts_scale(
                model=model,
                train_loader=self.ensemble_mace_sets[tag]["train_loader"],
                atomic_energies=self.atomic_energies,
                scaling=self.scaling,
            )

        logging.info(f"Trajectory worker {idx} is training.")
        # we train only for some epochs before we move to the next worker which may be running MD
        # all workers train on the same models with the respective training settings for
        # each ensemble member
        for _ in range(self.intermediate_epochs):
            for tag, model in self.ensemble.items():
                # from here
                #############

                # training_setups[tag] = setup_mace_training(
                #                settings=mace_settings,
                #                model=ensemble[tag],
                #                tag=tag,
                #                )

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
                _, _, metrics = validate_epoch_ensemble(
                    ensemble=self.ensemble,
                    ema=self.training_setups[tag]["ema"],
                    loss_fn=self.training_setups[tag]["loss_fn"],
                    valid_loader=self.ensemble_mace_sets[tag]["valid_loader"],
                    output_args=self.training_setups[tag]["output_args"],
                    device=self.training_setups[tag]["device"],
                    logger=logger,
                    log_errors=self.mace_settings["MISC"]["error_table"],
                    epoch=self.trajectory_epochs[idx],
                )
                self.current_valid = metrics["mae_f"]

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

        if self.trajectory_epochs[idx] == self.max_epochs_worker:
            self.trajectory_training[idx] = "running"
            self.num_workers_training -= 1
            self.trajectory_epochs[idx] = 0
            logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

        return None

    def running_task(self, idx):
        current_MD_step = self.trajectory_MD_steps[idx]
        if (
            current_MD_step > self.max_MD_steps
            and self.trajectory_training[idx] == "running"
        ):
            logging.info(
                f"Trajectory worker {idx} reached maximum MD steps and is killed."
            )
            self.num_MD_limits_reached += 1
            self.trajectory_training[idx] = "killed"
            return "killed"

        else:

            # before the worker takes another MD step it checks
            # if a job has been finished
            # if it is finished it adds the point to the training set
            # and sets the worker to training mode without taking another MD step
            # if the job has not finished the worker is skipped

            # this means that there should ideally exist a queue for the
            # jobs that are sent to the FHI-aims calculation

            ###########################
            # Here would be the MD step
            ###########################

            self.point = self.trajectories[idx][current_MD_step]
            self.mace_point = create_mace_dataset(
                data=[self.point],
                z_table=self.z_table,
                seed=None,
                r_max=self.r_max,
            )
            self.trajectory_MD_steps[idx] += 1
            if current_MD_step % self.skip_step == 0:
                logging.info(
                    f"Trajectory worker {idx} at step {current_MD_step}."
                )
                prediction = ensemble_prediction(
                    models=list(self.ensemble.values()),
                    atoms_list=[self.point],
                    device=self.device,
                    dtype=self.mace_settings["GENERAL"]["default_dtype"],
                )
                uncertainty = max_sd_2(prediction)
                # compute moving average of uncertainty
                self.uncertainties.append(uncertainty)
                # limit the history to 400 TODO: make this a parameter
                if len(self.uncertainties) > 400:
                    self.uncertainties = self.uncertainties[-400:]
                if len(self.uncertainties) > 10:
                    mov_avg_uncert = np.mean(self.uncertainties)
                    self.threshold = mov_avg_uncert * (1.0 + self.c_x)

                if uncertainty > self.threshold:
                    logging.info(
                        f"Uncertainty of point is beyond threshold {np.round(self.threshold,3)} at worker {idx}: {round(uncertainty.item(),3)}."
                    )
                    #################################
                    # Here would be the FHI-aims call
                    #################################

                    # it sends the job and does not wait for the result but
                    # continues with the next worker. only if the job is done
                    # the worker is set to training mode
                    self.trajectory_training[idx] = "waiting"
                    self.num_workers_waiting += 1

                    # for analysis
                    self.t_intervals[idx].append(current_MD_step)
                    logging.info(
                        f"Trajectory worker {idx} is waiting for job to finish."
                    )

            if (
                current_MD_step % self.sanity_skip == 0
            ):  # should not be static but increase with time, based on how many uninterrupted MD steps have been taken or if all workes are running
                logging.info(f"Trajectory worker {idx} doing a sanity check.")
                if current_MD_step % self.skip_step == 0:
                    sanity_prediction = prediction
                    sanity_uncertainty = uncertainty
                else:
                    sanity_prediction = ensemble_prediction(
                        models=list(self.ensemble.values()),
                        atoms_list=[self.point],
                        device=self.device,
                        dtype=self.mace_settings["GENERAL"]["default_dtype"],
                    )
                    sanity_uncertainty = max_sd_2(sanity_prediction)

                sanity_uncertainty, max_error = self.sanity_check(
                    sanity_prediction=sanity_prediction
                )
                self.sanity_checks[idx].append((sanity_uncertainty, max_error))
                self.check += 1


    def run(self):
        logging.info("Starting active learning procedure.")
        self.current_valid = np.inf
        self.threshold = np.inf
        self.point_added = 0  # counts how many points have been added to the training set to decide when to add a point to the validation set
        self.num_MD_limits_reached = 0
        self.num_workers_training = 0  # maybe useful lateron to give CPU some to work if all workers are training
        self.num_workers_waiting = 0
        self.total_epoch = 0
        self.check = 0
        while True:
            for trajectory_idx, _ in enumerate(self.trajectories):
                # workers wait for the "FHI-aims" calculation to finish
                # if the calculation is finished the worker adds the point to the training or validation set
                # based on the point_added counter and a ratio that is set in the active_learning_settings.yaml
                if self.trajectory_training[trajectory_idx] == "waiting":

                    set_limit = self.waiting_task(trajectory_idx)
                    if set_limit:
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
                ):  # and cpu == 'idle':
                    logging.info(
                        "All workers are in training mode."
                    )  # Sampling random points from existing trajectories.")
                if self.num_workers_waiting == self.num_trajectories:
                    logging.info("All workers are waiting for jobs to finish.")

            if self.num_MD_limits_reached == self.num_trajectories:
                logging.info(
                    "All trajectories reached maximum MD steps. Training until convergence."
                )
                break
            if (
                len(
                    self.ensemble_ase_sets[list(self.ensemble.keys())[0]][
                        "train"
                    ]
                )
                >= self.max_set_size
            ):
                logging.info(
                    "Maximum size of training set reached. Training until convergence."
                )
                break
            if self.current_valid < self.desired_accuracy:
                logging.info(
                    "Desired accuracy reached. Training until convergence."
                )
                break

        # turn keys which are ints into strings
        # save the datasets and the intervals for analysis
        save_datasets(
            ensemble=self.ensemble,
            ensemble_ase_sets=self.ensemble_ase_sets,
            path=self.dataset_dir / "final",
        )


    def converge(self):
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

            update_avg_neighs_shifts_scale(
                model=model,
                train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                atomic_energies=self.atomic_energies,
                scaling=self.scaling,
            )
            training_setups = {}
        # reseting optimizer and scheduler
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
                    ema=training_setup["ema"],
                    loss_fn=training_setup["loss_fn"],
                    valid_loader=self.ensemble_ase_sets[tag]["valid_loader"],
                    output_args=training_setup["output_args"],
                    device=training_setup["device"],
                    logger=logger,
                    log_errors=self.mace_settings["MISC"]["error_table"],
                    epoch=epoch,
                )
                if best_valid_loss > valid_loss and (best_valid_loss - valid_loss) > 0.01:
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
    def evaluate_ensemble(
        self,
        ase_atoms_list
    ):
        tag = list(self.ensemble.keys())[0]
        
        test_set = create_mace_dataset(
            data=ase_atoms_list,
            z_table=self.z_table,
            seed=self.seeds[tag],
            r_max=self.r_max,
        )

        test_dataloader = tools.torch_geometric.dataloader.DataLoader(
            dataset=test_set,
            batch_size=self.set_batch_size,
            shuffle=False,
            drop_last=False,
        )

        _, _, metrics = validate_epoch_ensemble(
            ensemble=self.ensemble,
            ema=self.training_setups[tag]["ema"],
            loss_fn=self.training_setups[tag]["loss_fn"],
            valid_loader=test_dataloader,
            output_args=self.training_setups[tag]["output_args"],
            device=self.training_setups[tag]["device"],
            logger=None,
            log_errors=None,
            epoch=0,
        )
        return metrics

class StandardMACEEnsembleProcedure:
    def __init__(
        self, 
        mace_settings: dict,
        dataset_dir_train: str = None,
        dataset_dir_valid: str = None,
        num_members: int = None,
        ensemble_ase_sets: dict = None,
        seeds: list = None,
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
        self.get_atomic_energies()
        
        if seeds is None:
            self.ensemble_seeds = np.random.randint(
                        0, 1000, size=num_members
                            )
        else:
            self.ensemble_seeds = np.array(seeds)    
    
        (
        self.seeds_tags_dict,
        self.ensemble,
        self.training_setups,
        ) = setup_ensemble_dicts(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            al_settings=None,
            atomic_energies_dict=self.atomic_energies_dict,
            save_seeds_tags_dict=False
        )
        if ensemble_ase_sets is not None:
            self.ensemble_ase_sets = ensemble_ase_sets
        else:
            train_set = read(dataset_dir_train)
            valid_set = read(dataset_dir_valid)
            self.ensemble_ase_sets = {
                tag: {"train": train_set, "valid": valid_set} for tag in self.ensemble.keys()
            }
            
    def train(self):
        
        for _, (tag, model) in enumerate(self.ensemble.items()):
            train_set = create_mace_dataset(
                data=self.ensemble_ase_sets[tag]["train"],
                z_table=self.z_table,
                seed=self.seeds_tags_dict[tag],
                r_max=self.r_max,
            )
            valid_set = create_mace_dataset(
                data=self.ensemble_ase_sets[tag]["valid"],
                z_table=self.z_table,
                seed=self.seeds_tags_dict[tag],
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

            update_avg_neighs_shifts_scale(
                model=model,
                train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                atomic_energies=self.atomic_energies,
                scaling=self.scaling,
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

            if epoch % self.eval_interval == 0:
                (
                    ensemble_valid_losses,
                    valid_loss,
                    _,
                ) = validate_epoch_ensemble(
                    ensemble=self.ensemble,
                    ema=training_setup["ema"],
                    loss_fn=training_setup["loss_fn"],
                    valid_loader=self.ensemble_ase_sets[tag]["valid_loader"],
                    output_args=training_setup["output_args"],
                    device=training_setup["device"],
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
       
    def evaluate_ensemble(
        self,
        ase_atoms_list
        ):
            tag = list(self.ensemble.keys())[0]
            
            test_set = create_mace_dataset(
                data=ase_atoms_list,
                z_table=self.z_table,
                seed=self.seeds_tags_dict[tag],
                r_max=self.r_max,
            )

            test_dataloader = tools.torch_geometric.dataloader.DataLoader(
                dataset=test_set,
                batch_size=self.set_batch_size,
                shuffle=False,
                drop_last=False,
            )

            _, _, metrics = validate_epoch_ensemble(
                ensemble=self.ensemble,
                ema=self.training_setups[tag]["ema"],
                loss_fn=self.training_setups[tag]["loss_fn"],
                valid_loader=test_dataloader,
                output_args=self.training_setups[tag]["output_args"],
                device=self.training_setups[tag]["device"],
                logger=None,
                log_errors=None,
                epoch=0,
            )
            return metrics
    
    def get_atomic_energies(self):
        #TODO: remove hardocde!!!
        self.atomic_energies_dict = {1: -12.482766945, 6: -1027.170068545}
        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
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
        self.eval_interval = self.mace_settings['MISC']['eval_interval']
        self.max_num_epochs = self.mace_settings['TRAINING']['max_num_epochs']
        self.patience = self.mace_settings['TRAINING']['patience']
    
    def create_folders(self):
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.standard_model_dir = Path(self.model_dir) / "standard"
        self.standard_model_dir.mkdir(
            parents=True, exist_ok=True
        )
        