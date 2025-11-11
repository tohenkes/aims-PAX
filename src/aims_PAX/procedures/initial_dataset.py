import numpy as np
import time
import shutil
from .preparation import PrepareInitialDatasetProcedure
from mace import tools
from mace.calculators import mace_mp
from typing import Optional
from so3krates_torch.calculator.so3 import SO3LRCalculator
from aims_PAX.tools.utilities.data_handling import (
    create_dataloader,
    update_datasets,
    save_datasets,
)
from aims_PAX.tools.utilities.utilities import (
    update_model_auxiliaries,
    save_checkpoint,
    save_ensemble,
    log_yaml_block,
)
from aims_PAX.tools.utilities.parsl_utils import (
    recalc_dft_parsl,
    handle_parsl_logger,
    prepare_parsl,
)
from aims_PAX.tools.train_epoch_mace import (
    train_epoch,
    validate_epoch_ensemble,
)
import ase
import logging
import random
import sys

try:
    import asi4py
except Exception as e:
    asi4py = None
try:
    import parsl
except ImportError:
    parsl = None


sys.stdout.flush()


class InitialDatasetProcedure(PrepareInitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    This is the base class for the serial, parallel, and PARSL version of
    this workflow.
    TODO: change training so it uses the code from AL
    """

    def _sample_points(self):
        """
        Dummy function to sample points. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError

    def _train(self) -> bool:
        """
        Trains the model(s) on the sampled points and updates the
        average number of neighbors, shifts and the scaling factor
        for each ensemble member.

        Returns:
            bool: Returns True if the maximum number of epochs is reached.
        """
        if self.rank == 0:
            random.shuffle(self.sampled_points)
            # each ensemble member collects their respective points
            for number, (tag, model) in enumerate(self.ensemble.items()):

                member_points = self.sampled_points[
                    len(self.atoms)
                    * self.n_points_per_sampling_step_idg
                    * number : len(self.atoms)
                    * self.n_points_per_sampling_step_idg
                    * (number + 1)
                ]

                (
                    self.ensemble_ase_sets[tag],
                    self.ensemble_mace_sets[tag],
                ) = update_datasets(
                    new_points=member_points,
                    mace_set=self.ensemble_mace_sets[tag],
                    ase_set=self.ensemble_ase_sets[tag],
                    valid_split=self.valid_ratio,
                    z_table=self.z_table,
                    seed=self.seed,
                    r_max=self.r_max,
                    key_specification=self.key_specification
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
                # because we are continously training the model we
                # have to update the average number of neighbors, shifts
                # and the scaling factor continously as well
                update_model_auxiliaries(
                    model=model,
                    mace_sets=self.ensemble_mace_sets[tag],
                    atomic_energies_list=self.ensemble_atomic_energies[tag],
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.ensemble_atomic_energies_dict[
                        tag
                    ],
                    dtype=self.dtype,
                    device=self.device,
                )
                logging.info(
                    f"Training set size for '{tag}': "
                    f"{len(self.ensemble_mace_sets[tag]['train'])}; Validation"
                    f" set size: {len(self.ensemble_mace_sets[tag]['valid'])}."
                )

            logging.info("Training.")
            ensemble_valid_losses = {
                tag: np.inf for tag in self.ensemble.keys()
            }
            for _ in range(self.intermediate_epochs):
                # each member gets trained individually
                for tag, model in self.ensemble.items():

                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["loss_dir"],
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
                        epoch=self.epoch,
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
                # the validation errors are averages over the ensemble members
                if (
                    self.epoch % self.valid_skip == 0
                    or (self.epoch + 1) % self.valid_skip == 0
                ):
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        metrics,
                    ) = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.training_setups,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=self.epoch,
                    )
                    self.current_valid = metrics["mae_f"]

                    if self.analysis:
                        self._handle_analysis(
                            valid_loss=valid_loss,
                            ensemble_valid_losses=ensemble_valid_losses,
                        )

                    for tag, model in self.ensemble.items():
                        save_checkpoint(
                            checkpoint_handler=self.training_setups[tag][
                                "checkpoint_handler"
                            ],
                            training_setup=self.training_setups[tag],
                            model=model,
                            epoch=self.epoch,
                            keep_last=False,
                        )
                        save_datasets(
                            self.ensemble,
                            self.ensemble_ase_sets,
                            path=self.dataset_dir / "initial",
                            initial=True,
                        )
                    if self.create_restart:
                        self._update_restart_dict()
                        np.save(
                            "restart/initial_ds/initial_ds_restart.npy",
                            self.init_ds_restart_dict,
                        )
                    if (
                        self.desired_acc * self.desired_acc_scale_idg
                        >= self.current_valid
                    ):
                        logging.info(
                            f"Accuracy criterion reached at step {self.step}."
                        )
                        logging.info(
                            f"Criterion: {self.desired_acc * self.desired_acc_scale_idg}; Current accuracy: {self.current_valid}."
                        )

                        break

                self.epoch += 1

            if (
                self.epoch == self.max_initial_epochs
            ):  # TODO: change to a different variable (shares with al-algo right now)
                logging.info("Maximum number of epochs reached.")
                return True

    def _sample_and_train(self):
        """
        Combines the sampling of points and the training of the ensemble members
        in one method (easier for overwritting for derived classes).
        """
        if self.rank == 0:
            logging.info(f"Sampling new points at step {self.step}.")
        self.sampled_points = []
        # in case SCF fails to converge no point is returned
        while len(self.sampled_points) == 0:
            self.sampled_points = self._sample_points()

        self.step += 1
        self._train()

    def _setup_calcs(self):
        """
        Dummy function to set up the calculators. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError

    def _close_aims(self):
        """
        Dummy function to close the AIMS calculators. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError

    def _handle_analysis(
        self,
        valid_loss: float,
        ensemble_valid_losses: dict,
        save_path: str = "analysis/initial_losses.npz",
    ):
        """
        Collects number of epochs, average validation loss and
        per ensemble member validation losses and saves in a
        npz file.

        Args:
            valid_loss (float): Averaged validation loss over the ensemble.
            ensemble_valid_losses (dict): Per ensemble member
                                                    validation losses.
            save_path (str, optional): Path to save the analysis data.
                    Defaults to "analysis/initial_losses.npz".
        """
        self.collect_losses["epoch"].append(self.epoch)
        self.collect_losses["avg_losses"].append(valid_loss)
        self.collect_losses["ensemble_losses"].append(ensemble_valid_losses)
        np.savez(save_path, **self.collect_losses)

    def run(self):
        """
        Main function to run the initial dataset generation procedure.
        It samples points and trains the ensemble members until the
        stopping criterion is met.

        """

        # initializing md and FHI aims
        for idx, atoms in self.trajectories.items():
            dyn = self.setup_md(
                atoms,
                md_settings=self.md_settings[idx]
            )
            self.md_drivers[idx] = dyn

        if self.rank == 0:
            logging.info(
                f'Using following settings for MDs:'
            )
            log_yaml_block(
                "MD_SETTINGS",
                self.md_settings
            )

        self._setup_calcs()

        self.current_valid = np.inf
        # criterion for initial dataset is multiple of the desired accuracy
        # TODO: add maximum initial dataset len criterion
        while (
            self.desired_acc * self.desired_acc_scale_idg <= self.current_valid
            and self.epoch < self.max_initial_epochs
        ):

            self._sample_and_train()
            # only one worker is doing the training right now,
            # so we have to broadcast the criterion so they
            # don't get stuck in the while loop
            self.comm_handler.barrier()
            self.current_valid = self.comm_handler.bcast(
                self.current_valid, root=0
            )
            self.epoch = self.comm_handler.bcast(self.epoch, root=0)
            self.comm_handler.barrier()

        if self.rank == 0:

            save_ensemble(
                ensemble=self.ensemble,
                training_setups=self.training_setups,
                mace_settings=self.mace_settings,
            )

            if self.create_restart:
                self._update_restart_dict()
                self.init_ds_restart_dict["initial_ds_done"] = True
                np.save(
                    "restart/initial_ds/initial_ds_restart.npy",
                    self.init_ds_restart_dict,
                )
        self.logger.handlers.clear()
        self._close_aims()
        return 0


class InitialDatasetAIMD(InitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses ab initio MD to sample points. Runs serially.

    """

    def _sample_points(self) -> list:
        """
        Samples geometries solely using AIMD.

        Returns:
            list: List of ASE Atoms objects.
        """

        sampled = []
        for idx in self.trajectories.keys():
            sampled.extend(
                [
                    self._run_MD(
                        atoms=self.trajectories[idx], dyn=self.md_drivers[idx]
                    )
                    for _ in range(
                        self.ensemble_size
                        * self.n_points_per_sampling_step_idg
                    )
                ]
            )

        return sampled

    def _setup_calcs(self):
        """
        Sets up the calculators for the initial dataset generation.
        In this case it sets up the AIMS calculators for AIMD.
        """
        if len(self.atoms) > 1:
            raise NotImplementedError(
                "Initital dataset generation with AIMD is not "
                "implemented for multiple geometries."
            )
        for idx in self.trajectories.keys():
            self.trajectories[idx].calc = self._setup_aims_calculator(
                self.trajectories[idx]
            )

    def _close_aims(self):
        """
        Kills the AIMS calculators.
        """
        self.atoms.calc.close()


class InitialDatasetFoundational(InitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses a "foundational" model to sample points. These are then recomputed
    using DFT. Runs serially.
    """

    def _setup_foundational(
        self,
        model_choice: str,
        foundational_model_settings: dict
    ):
        """
        Creates the foundational model for sampling.

        

        Returns:
            ase.Calculator: ASE calculator object.
        """

        if model_choice == 'mace-mp':
            mace_model = foundational_model_settings['mace_model']
            return mace_mp(
                model=mace_model,
                dispersion=False,
                default_dtype=self.dtype,
                device=self.device
            )
        elif model_choice == 'so3lr':
            r_max_lr = foundational_model_settings['r_max_lr']
            dispersion_lr_damping = foundational_model_settings[
                'dispersion_lr_damping'
            ]
            return SO3LRCalculator(
                r_max_lr=r_max_lr,
                dispersion_energy_cutoff_lr_damping=dispersion_lr_damping,
                compute_stress=self.compute_stress,
                device=self.device,
                default_dtype=self.dtype,
                key_specification=self.key_specification
            )
        else:
            raise ValueError(
                f"Unknown foundational model choice: {model_choice}"
            )

    def _recalc_dft(self, current_point: ase.Atoms) -> ase.Atoms:
        """
        Recalculates the energies and forces of the current point using
        the AIMS calculator. If the SCF is converged, it saves the energy
        and forces (and stress) in the MACE readable format.
        If not, it returns None.

        Args:
            current_point (ase.Atoms): System to recompute.

        Returns:
            ase.Atoms: Atoms object containing generated DFT data.
        """
        self.aims_calc.calculate(current_point, properties=self.properties)
        if self.aims_calc.asi.is_scf_converged:
            current_point.info["REF_energy"] = self.aims_calc.results["energy"]
            current_point.arrays["REF_forces"] = self.aims_calc.results[
                "forces"
            ]
            if self.compute_stress:
                current_point.info["REF_stress"] = self.aims_calc.results[
                    "stress"
                ]
            return current_point
        else:
            if self.rank == 0:
                logging.info("SCF not converged.")
            return None

    def _md_w_foundational(
        self,
    ):
        """
        Samples points using the foundational model.
        For each geometry n points are sampled with
        n = self.n_points_per_sampling_step_idg * self.ensemble_size.
        Thus, the total number of points sampled are n * n_geometries.
        """

        self.comm_handler.barrier()
        self.sampled_points = {idx: [] for idx in self.trajectories.keys()}
        if self.rank == 0:
            for idx in self.trajectories.keys():
                dyn = self.md_drivers[idx]
                atoms = self.trajectories[idx]
                for _ in range(
                    self.ensemble_size * self.n_points_per_sampling_step_idg
                ):
                    current_point = self._run_MD(atoms, dyn)
                    self.sampled_points[idx].append(current_point)
            total_points_sampled = sum(len(points) for points in self.sampled_points.values())
            logging.info(
                f"Sampled {total_points_sampled} points using foundational model."
            )
        self.comm_handler.barrier()
        self.sampled_points = self.comm_handler.bcast(
            self.sampled_points, root=0
        )
        self.comm_handler.barrier()

    def _sample_points(self) -> list:
        """
        Samples geometries using foundational model and recalculates
        the energies and forces with DFT.

        Returns:
            list: List of ASE Atoms objects.
        """
        self._md_w_foundational()
        if self.rank == 0:
            logging.info("Recalculating energies and forces with DFT.")
        recalculated_points = []
        for idx in self.trajectories.keys():
            for atoms in self.sampled_points[idx]:
                temp = self._recalc_dft(atoms)
                if temp is not None:
                    recalculated_points.append(temp)
        return recalculated_points

    def _setup_calcs(self):
        """
        Sets up the calculators for the initial dataset generation.
        In this case it sets up the AIMS calculators for recalculating
        the energies and forces and the foundational model for MD.
        """
        if len(self.atoms) > 1:
            raise NotImplementedError(
                "Initial dataset generation with foundational model "
                " without using PARSL for multiple geometries."
            )
        self.aims_calc = self._setup_aims_calculator(self.atoms[0])
        if self.rank == 0:
            logging.info(
                f"Initial dataset generation with foundational model: {self.foundational_model}."
            )
            foundational_calc = self._setup_foundational()
            for idx in self.trajectories.keys():
                self.trajectories[idx].calc = foundational_calc

    def _close_aims(self):
        """
        Kills the AIMS calculator.
        """
        self.aims_calc.close()


class InitialDatasetFoundationalParallel(InitialDatasetFoundational):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses a "foundational" model to sample points. These are then recomputed
    using DFT. Runs in parallel using MPI. The MD using the foundational
    model is propagted while DFT is being run.

    !!! WARNING: Not recommended (especially for large systems) as the
        model can generate strange geometries when run to long, which
        can happen when DFT calculations take too much time. Speedup is
        modest as DFT calculations are processed one at a time. Ideally
        use the PARSL version. !!!
    """

    def __init__(
        self,
        mace_settings: dict,
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        # this is necessary because of the way the MPI communicator is split
        super().__init__(
            mace_settings=mace_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
        if self.rank == 0:
            logging.warning(
                "Not recommended (especially for large systems) as the "
                "model can generate strange geometries when run to long, which"
                "can happen when DFT calculations take too much time. Speedup "
                "is modest as DFT calculations are processed one at a time. "
                "Ideally use the PARSL version."
            )
        # one for ML and one for DFT
        if self.rank == 0:
            self.color = 0
        else:
            self.color = 1

        self.comm = self.comm_handler.comm.Split(
            color=self.color, key=self.rank
        )

    def _close_aims(self):
        # this is just to overwrite the function in the parent class
        # due to the communicators we are closing it inside the
        # sample_and_train function
        return None

    def _setup_aims_calculator(
        self,
        atoms: ase.Atoms,
    ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object. Uses the AIMS
        settings from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.
            pbc (bool, optional): If periodic boundry conditions are required
            or not. Defaults to False.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """
        aims_settings = self.aims_settings.copy()
        # only one communictor initializes aims
        if self.color == 1:
            self.properties = ["energy", "forces"]
            if self.compute_stress:
                self.properties.append("stress")

            def init_via_ase(asi):
                from ase.calculators.aims import Aims, AimsProfile

                aims_settings["profile"] = AimsProfile(
                    command="asi-doesnt-need-command"
                )
                calc = Aims(**aims_settings)
                calc.write_inputfiles(asi.atoms, properties=self.properties)

            calc = asi4py.asecalc.ASI_ASE_calculator(
                self.ASI_path, init_via_ase, self.comm, atoms
            )
            return calc
        else:
            return None

    def _sample_and_train(self) -> list:
        """
        Samples points using the foundational models, computes
        DFT data in parallel and trains the ensemble members.
        Contains all the MPI communications.

        Returns:
            list: List of ASE Atoms objects with the sampled points
                and their DFT data.
        """
        self.sampled_points = []

        # TODO: add stress
        temp_sampled_geometries = []
        temp_sampled_forces = []
        temp_sampled_energies = []
        self.req_geometries, self.req_energies, self.req_forces = (
            None,
            None,
            None,
        )

        self.req = None  # handling data communication
        self.criterion_req = (
            None  # handling the communication regarding stopping
        )
        current_point = None
        recieved_points = None
        criterion_met = False
        self.atoms_dummy = self.atoms.copy()

        if self.rank == 0:
            logging.info("Starting sampling and training using parallel mode.")

        while not criterion_met:
            if self.color == 0:
                current_point = self._run_MD(self.atoms, self.dyn)
                # TODO: also send cell and pbc
                geometry = current_point.get_positions()
                # using isend to create a queue of messages
                sample_send = self.comm_handler.comm.Isend(
                    geometry, dest=1, tag=96
                )
                sample_send.Wait()

                # creating requests if there are none
                if (
                    self.req_geometries is None
                    and self.req_energies is None
                    and self.req_forces is None
                ):
                    # creates buffers in memory for receiving data
                    buf_geometries, buf_energies, buf_forces = (
                        np.zeros(
                            (
                                self.n_points_per_sampling_step_idg
                                * self.ensemble_size,
                                len(self.atoms),
                                3,
                            ),
                            dtype=float,
                        ),
                        np.zeros(
                            self.n_points_per_sampling_step_idg
                            * self.ensemble_size,
                            dtype=float,
                        ),
                        np.zeros(
                            (
                                self.n_points_per_sampling_step_idg
                                * self.ensemble_size,
                                len(self.atoms),
                                3,
                            ),
                            dtype=float,
                        ),
                    )
                    # non-blocking recieve for data
                    self.req_geometries = self.comm_handler.comm.Irecv(
                        buf=buf_geometries, source=1, tag=2210
                    )
                    self.req_energies = self.comm_handler.comm.Irecv(
                        buf=buf_energies, source=1, tag=2211
                    )
                    self.req_forces = self.comm_handler.comm.Irecv(
                        buf=buf_forces, source=1, tag=2212
                    )

                else:
                    # listening for data
                    status_geometries = (
                        self.req_geometries.Test()
                    )  # non-blocking recieve
                    status_energies = (
                        self.req_energies.Test()
                    )  # non-blocking recieve
                    status_forces = (
                        self.req_forces.Test()
                    )  # non-blocking recieve

                    if status_energies and status_forces and status_geometries:
                        recieved_points = []

                        for i in range(
                            self.n_points_per_sampling_step_idg
                            * self.ensemble_size
                        ):
                            temp = self.atoms_dummy.copy()
                            temp.set_positions(buf_geometries[i])
                            temp.info["REF_energy"] = buf_energies[i]
                            temp.arrays["REF_forces"] = buf_forces[i]
                            recieved_points.append(temp)

                        self.req_geometries = None
                        self.req_energies = None
                        self.req_forces = None

                        # checking if the criterion is met
                        criterion_met = (
                            self.desired_acc * self.desired_acc_scale_idg
                            >= self.current_valid
                            or self.epoch >= self.max_initial_epochs
                        )
                        if criterion_met:
                            # instructs the DFT worker to stop when
                            # the criterion is met
                            for dest in range(
                                1, self.comm_handler.comm.Get_size()
                            ):
                                self.criterion_send = (
                                    self.comm_handler.comm.isend(
                                        None, dest=dest, tag=2305
                                    )
                                )
                                self.criterion_send.Wait()
                            break
                        logging.info(
                            "Recieved points from DFT worker; training."
                        )
                        self.sampled_points.extend(recieved_points)
                        recieved_points = None
                        self._train()
                        logging.info("Training done, going back to sampling.")
            # DFT workers
            if self.color == 1:
                current_geometry = None
                # recieving the criterion
                if self.criterion_req is None:
                    self.criterion_req = self.comm_handler.comm.irecv(
                        source=0, tag=2305
                    )
                criterion_met = self.criterion_req.Test()
                if criterion_met:
                    break

                # recieving sampled point to recompute
                if self.rank == 1:
                    if self.req is None:
                        buffer = np.zeros(
                            self.atoms_dummy.get_positions().shape, dtype=float
                        )
                        self.req = self.comm_handler.comm.Irecv(
                            buf=buffer, source=0, tag=96
                        )
                    self.req.wait()  # blocking recieve
                    current_geometry = buffer.copy()

                self.req = None
                self.comm.Barrier()
                current_geometry = self.comm.bcast(current_geometry, root=0)
                self.comm.Barrier()

                current_point = self.atoms_dummy.copy()
                current_point.set_positions(current_geometry)

                dft_result = self._recalc_dft(current_point)

                # one rank sends data back
                if self.rank == 1:
                    energies, forces = (
                        dft_result.info["REF_energy"],
                        dft_result.arrays["REF_forces"],
                    )
                    # TODO: add stress
                    if dft_result is not None:
                        temp_sampled_geometries.append(current_geometry)
                        temp_sampled_energies.append(energies)
                        temp_sampled_forces.append(forces)

                        # if enough are computed send them to training worker
                    if (
                        len(temp_sampled_energies)
                        % (
                            self.n_points_per_sampling_step_idg
                            * self.ensemble_size
                        )
                        == 0
                        and len(temp_sampled_energies) != 0
                    ):
                        logging.info(
                            f"Computed {len(temp_sampled_energies)} points "
                            "with DFT and sending them to training worker."
                        )

                        # TODO: create loop or package data in one
                        self.req_send = self.comm_handler.comm.Isend(
                            np.array(temp_sampled_geometries), dest=0, tag=2210
                        )
                        self.req_send.Wait()

                        self.req_send = self.comm_handler.comm.Isend(
                            np.array(temp_sampled_energies), dest=0, tag=2211
                        )
                        self.req_send.Wait()

                        self.req_send = self.comm_handler.comm.Isend(
                            np.array(temp_sampled_forces), dest=0, tag=2212
                        )
                        self.req_send.Wait()

                        temp_sampled_geometries = []
                        temp_sampled_energies = []
                        temp_sampled_forces = []

        self.comm_handler.barrier()
        self.current_valid = self.comm_handler.bcast(
            self.current_valid, root=0
        )
        self.epoch = self.comm_handler.bcast(self.epoch, root=0)
        self.comm_handler.barrier()

        if self.color == 1:
            self.aims_calc.close()
        self.comm.Free()


class InitialDatasetPARSL(InitialDatasetFoundational):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses a "foundational" model to sample points. These are then recomputed
    using DFT. Uses PARSL and can run DFT in parallel on multiple nodes.
    """

    def __init__(
        self,
        mace_settings: dict,
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        close_parsl: bool = True,
    ):

        super().__init__(
            mace_settings=mace_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            use_mpi=False,
        )
        self.close_parsl = close_parsl

        if parsl is None:
            raise ImportError(
                "Parsl is not installed. Please install parsl"
                " to use this feature."
            )

        if self.rank == 0:
            logging.info("Setting up PARSL for initial dataset generation.")
            # TODO: create function to check if all
            # necessary settings are provided and fall back to
            # defaults if not
            parsl_setup_dict = prepare_parsl(
                cluster_settings=self.cluster_settings
            )
            self.config = parsl_setup_dict["config"]
            self.calc_dir = parsl_setup_dict["calc_dir"]
            self.clean_dirs = parsl_setup_dict["clean_dirs"]
            self.launch_str = parsl_setup_dict["launch_str"]
            self.calc_idx = parsl_setup_dict["calc_idx"]
            handle_parsl_logger(
                log_dir=self.log_dir / "parsl_initial_dataset.log",
            )
            logging.info("Using following settings for the HPC environment:")
            log_yaml_block("CLUSTER:", self.cluster_settings)
            self.comm_handler.barrier()

    def _sample_points(self) -> list:
        """
        Samples geometries using foundational model and recalculates
        the energies and forces with DFT using PARSL.

        Returns:
            list: List of ASE Atoms objects with the sampled points
                and their DFT data.
        """
        self._md_w_foundational()
        recalculated_points = []
        if self.rank == 0:
            logging.info("Recalculating energies and forces with DFT.")
            job_results = {idx: {} for idx in self.sampled_points.keys()}
            calc_launched = 0
            # loop over different systems
            for idx in self.sampled_points.keys():
                # loop over geometries of different systems
                for i, atoms in enumerate(self.sampled_points[idx]):
                    self.calc_idx += 1
                    # launches a parsl app and returns a future
                    # that can be used to get the result later
                    directory = self.calc_dir / f"initial_calc_{self.calc_idx}"
                    # if there is only one entry in aims_settings the same
                    # settings are used for all systems
                    system_idx = idx if len(self.aims_settings) > 1 else 0
                    temp_result = recalc_dft_parsl(
                        positions=atoms.get_positions(),
                        species=atoms.get_chemical_symbols(),
                        cell=atoms.get_cell(),
                        pbc=atoms.pbc,
                        aims_settings=self.aims_settings[system_idx],
                        directory=directory,
                        properties=self.properties,
                        ase_aims_command=self.launch_str,
                    )
                    job_results[system_idx][i] = temp_result
                    calc_launched += 1

            while calc_launched > 0:
                for idx in job_results.keys():
                    for i in list(job_results[idx].keys()):
                        result = job_results[idx][i]
                        if result.done():
                            temp = result.result()
                            if temp is None:
                                logging.warning(
                                    f"SCF not converged for point {i}. Skipping."
                                )
                                del job_results[idx][i]
                                calc_launched -= 1
                                continue
                            current_point = self.sampled_points[idx][i]
                            current_point.info["REF_energy"] = temp["energy"]
                            current_point.arrays["REF_forces"] = temp["forces"]
                            if self.compute_stress:
                                current_point.info["REF_stress"] = temp["stress"]
                            recalculated_points.append(current_point)

                            if (
                                len(recalculated_points)
                                % self.idg_progress_dft_update
                            ) == 0 or (
                                len(recalculated_points)
                                == len(self.sampled_points)
                            ):
                                logging.info(
                                    f"Recalculated {len(recalculated_points)} points."
                                )
                            del job_results[idx][i]
                            calc_launched -= 1
                    time.sleep(0.5)

            if self.clean_dirs:
                try:
                    for calc_dir in self.calc_dir.glob("initial_calc_*"):
                        shutil.rmtree(calc_dir)
                except Exception as e:
                    logging.error(
                        f"Error while cleaning directories: {e}. "
                        "Please check the directories manually."
                    )

        return recalculated_points

    def run(self):
        if self.rank == 0:
            parsl.load(self.config)
        super().run()
        if self.rank == 0:
            if self.clean_dirs:
                try:
                    shutil.rmtree(self.calc_dir)
                except Exception as e:
                    logging.error(
                        f"Error while cleaning directories: {e}. "
                        "Please check the directories manually."
                    )

    def _setup_calcs(
        self,
    ) -> None:
        if self.rank == 0:
            logging.info(
                "Initial dataset generation with foundational "
                f"model: {self.foundational_model}."
            )
            foundational_calc = self._setup_foundational(
                model_choice=self.foundational_model,
                foundational_model_settings=self.foundational_model_settings,
            )
            for idx in self.trajectories.keys():
                self.trajectories[idx].calc = foundational_calc

    def _close_aims(self):
        if self.close_parsl:
            logging.info("Closing PARSL.")
            parsl.dfk().cleanup()
        else:
            logging.info(
                "Not closing PARSL. Please close it manually if needed."
            )
