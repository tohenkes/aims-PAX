import logging
import torch
import numpy as np
from typing import (
    Tuple,
    List,
    Union
)
from mace import data as mace_data
from mace.tools import torch_geometric, torch_tools, utils
from mace.data.utils import (
    load_from_xyz,
)
from mace.tools.utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)
from torchmetrics import Metric


def evaluate_model(
    atoms_list: list,
    model: str,
    batch_size: int,
    device: str,
    compute_stress: bool = False,
    dtype: str = "float64",
) -> torch.tensor:
    """
    Evaluate a MACE model on a list of ASE atoms objects.
    This only handles atoms list with a single species. TODO: address this

    Args:
        atoms_list (list): List of ASE atoms objects.
        model (str): MACE model to evaluate.
        batch_size (int): Batch size for evaluation.
        device (str): Device to evaluate the model on.
        compute_stress (bool, optional): Compute stress or not.
                                        Defaults to False.
        dtype (str, optional): Data type of model. Defaults to "float64".

    Returns:
        torch.tensor: _description_
    """
    torch_tools.set_default_dtype(dtype)

    # Load data and prepare input
    configs = [mace_data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace_data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    stresses_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    # TODO: This only works for predicting a single molecule not different ones in one set
    forces_array = np.stack(forces_collection).reshape(len(energies), -1, 3)
    assert len(atoms_list) == len(energies) == len(forces_array)
    if compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

        return energies, forces_array, stresses
    else:
        return energies, forces_array


def ensemble_prediction(
    models: list,
    atoms_list: list,
    device: str,
    dtype: str = "float64",
    batch_size: int = 1,
    return_energies: bool = False,
) -> np.array:
    """
    Predict forces for a list of ASE atoms objects using an ensemble of models.
    !!! Does not reduce the energies or forces to a single value. !!!
    TODO: currently only works for atoms list with a single species.

    Args:
        models (list): List of models.
        atoms_list (list): List of ASE atoms objects.
        device (str): Device to evaluate the models on.
        dtype (str, optional): Dtype of models. Defaults to "float64".
        batch_size (int, optional): Batch size of evaluation. Defaults to 1.
        return_energies (bool, optional): Whether to return energies or not.
                                            Defaults to False.

    Returns:
        np.array: Forces [n_models, n_mols, n_atoms, xyz]
        Optionally:
        np.array: (
            Energies [n_models, n_mols],
            Forces [n_models, n_mols, n_atoms, xyz]
        )

    """
    all_forces = []
    all_energies = []
    i = 0
    for model in models:
        E, F = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        all_forces.append(F)
        all_energies.append(E)
        i += 1

    all_forces = np.stack(all_forces).reshape(
        (len(models), len(atoms_list), -1, 3)
    )

    all_energies = np.stack(all_energies).reshape(
        (len(models), len(atoms_list))
    )

    if return_energies:
        return all_energies, all_forces
    return all_forces


def evaluate_model_mace_ds(
    mace_ds: list,
    model: str,
    batch_size: int,
    device: str,
    compute_stress: bool = False,
    dtype: str = "float64",
) -> torch.tensor:
    """
    Evaluate a MACE model on a MACE style dataset.
    This only handles atoms list with a single species.

    Args:
        atoms_list (list): List of ASE atoms objects.
        model (str): MACE model to evaluate.
        batch_size (int): Batch size for evaluation.
        device (str): Device to evaluate the model on.
        compute_stress (bool, optional): Compute stress or not.
                                            Defaults to False.
        dtype (str, optional): Data type of model. Defaults to "float64".

    Returns:
        torch.tensor: _description_
    """
    torch_tools.set_default_dtype(dtype)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=mace_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    stresses_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    # TODO: This only works for predicting a single molecule not different 
    # ones in one set
    forces_array = np.stack(forces_collection).reshape(len(energies), -1, 3)
    assert len(mace_ds) == len(energies) == len(forces_array)
    if compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(mace_ds) == stresses.shape[0]

        return energies, forces_array, stresses
    else:
        return energies, forces_array


def ensemble_prediction_mace_ds(
    models: list,
    mace_ds: list,
    device: str,
    dtype: str = "float64",
    batch_size: int = 1,
    return_energies: bool = False,
) -> np.array:
    """
    Predict forces for a MACE style dataset using an ensemble of models.
    !!! Does not reduce the energies or forces to a single value. !!!
    TODO: currently only works for atoms list with a single species.

    Args:
        models (list): List of models.
        atoms_list (list): List of ASE atoms objects.
        device (str): Device to evaluate the models on.
        dtype (str, optional): Dtype of models. Defaults to "float64".
        batch_size (int, optional): Batch size of evaluation. Defaults to 1.
        return_energies (bool, optional): Whether to return energies or not. 
                                            Defaults to False.

    Returns:
        np.array: Forces [n_models, n_mols, n_atoms, xyz]
        Optionally:
        np.array: Energies [n_models, n_mols], 
                    Forces [n_models, n_mols, n_atoms, xyz]

    """
    all_forces = []
    all_energies = []
    i = 0
    for model in models:
        E, F = evaluate_model_mace_ds(
            mace_ds=mace_ds,
            model=model,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        all_forces.append(F)
        all_energies.append(E)
        i += 1

    all_forces = np.stack(all_forces).reshape(
        (len(models), len(mace_ds), -1, 3)
    )

    all_energies = np.stack(all_energies).reshape((len(models), len(mace_ds)))

    if return_energies:
        return all_energies, all_forces
    return all_forces


class MACEEval(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "num_data", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_stress_per_atom", default=[], dist_reduce_fx="cat"
        )
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_virials_per_atom", default=[], dist_reduce_fx="cat"
        )
        self.add_state(
            "Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")

    def update(self, batch, output):  # pylint: disable=arguments-differ
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1.0
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"])
                / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1.0
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
        if output.get("stress") is not None and batch.stress is not None:
            self.stress_computed += 1.0
            self.delta_stress.append(batch.stress - output["stress"])
            self.delta_stress_per_atom.append(
                (batch.stress - output["stress"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.virials_computed += 1.0
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.Mus_computed += 1.0
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )

    def convert(
        self, delta: Union[torch.Tensor, List[torch.Tensor]]
    ) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return torch_tools.to_numpy(delta)

    def compute(self):
        aux = {}
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["max_e"] = compute_max_error(delta_es)
            aux["max_e_per_atom"] = compute_max_error(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["max_f"] = compute_max_error(delta_fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            delta_stress_per_atom = self.convert(self.delta_stress_per_atom)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)

        return aux


def test_model(
    model: torch.nn.Module,
    data_loader: torch_geometric.DataLoader,
    output_args: dict,
    device: str,
    return_predictions: bool = False,
) -> dict:
    """
    Function to test a MACE model on a set of configurations.

    Args:
        model (torch.nn.Module): Model to test.
        data_loader (torch_geometric.DataLoader): DataLoader for the test set.
        output_args (dict): Dictionary of output arguments for the model.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").
        return_predictions (bool, optional): Whether to return the predictions
            made during the test. Defaults to False.

    Returns:
        dict: Dictionary with the computed metrics.
    """
    for param in model.parameters():
        param.requires_grad = False

    metrics = MACEEval().to(device)
    start_time = time.time()
    if return_predictions:
        predictions = {}
        if output_args.get("energy", False):
            predictions["energy"] = []
        if output_args.get("forces", False):
            predictions["forces"] = []
        if output_args.get("stress", False):
            predictions["stress"] = []
        if output_args.get("virials", False):
            predictions["virials"] = []
        if output_args.get("dipole", False):
            predictions["dipole"] = []

    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        aux = metrics(batch, output)

        if return_predictions:
            if output_args.get("energy", False):
                predictions["energy"].append(output["energy"])
            if output_args.get("forces", False):
                predictions["forces"].append(output["forces"])
            if output_args.get("stress", False):
                predictions["stress"].append(output["stress"])
            if output_args.get("virials", False):
                predictions["virials"].append(output["virials"])
            if output_args.get("dipole", False):
                predictions["dipole"].append(output["dipole"])

    aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True
    if return_predictions:
        for key in predictions.keys():
            predictions[key] = (
                torch.cat(predictions[key], dim=0).detach().cpu()
            )
        aux["predictions"] = predictions
    return aux


def test_ensemble(
    ensemble: dict,
    batch_size: int,
    output_args: dict,
    device: str,
    path_to_data: str = None,
    atoms_list: list = None,
    logger: MetricsLogger = None,
    log_errors: str = "PerAtomMAE",
    return_predictions: bool = False,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[dict, dict]:
    """
    Function taken from MACE code and adapted to work with ensembles.
    Tests an ensemble of MACE models on a set of configurations.
    
    Either `atoms_list` or `path_to_data` must be provided. So it's data
    is either loaded or the atoms list is transformed into a MACE
    compatible format.

    Args:
        ensemble (dict): Dictionary of MACE models.
        batch_size (int): Batch size for testing the models.
        output_args (dict): Dictionary of output arguments for the models.
        device (str): Device to run the models on (e.g., "cpu" or "cuda").
        path_to_data (str, optional): Path to the data file in ASE readable 
                                        format.
        atoms_list (list, optional): . Defaults to None.
        logger (MetricsLogger, optional): Logger object for eval.
                                            Defaults to None.
        log_errors (str, optional): What error to log.
                                            Defaults to "PerAtomMAE".
        return_predictions (bool, optional): Whether to return the predictions
                                    made during the test. Defaults to False.
        energy_key (str, optional): How energy is defined in the ase.Atoms.
                                        Defaults to "REF_energy".
        forces_key (str, optional): How forces are defined in the ase.Atoms.
                                        Defaults to "REF_forces".
        stress_key (str, optional): How stress is defined in the ase.Atoms.
                                        Defaults to "REF_stress".
        virials_key (str, optional): How virials are defined in the ase.Atoms.
                                        Defaults to "virials".
        dipole_key (str, optional): How dipoles are defined in the ase.Atoms.
                                        Defaults to "dipoles".
        charges_key (str, optional): How charges are defined in the ase.Atoms.
                                        Defaults to "charges".
        head_key (str, optional): Which output head to test. 
                                        Defaults to "head".

    Raises:
        ValueError: Raises an error if neither `atoms_list` nor `path_to_data`
                    is provided.

    Returns:
        Tuple[dict, dict]: Avgerage metrics and ensemble metrics.
    """
    if atoms_list is not None:
        configs = [
            mace_data.config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                dipole_key=dipole_key,
                virials_key=virials_key,
                charges_key=charges_key,
                head_key=head_key,
            )
            for atoms in atoms_list
        ]
    elif path_to_data is not None:
        _, configs = load_from_xyz(
            file_path=path_to_data,
            config_type_weights=None,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            head_key=head_key,
        )

    else:
        raise ValueError("Either atoms_list or path_to_data must be provided")

    z_table = utils.AtomicNumberTable(
        [int(z) for z in ensemble[list(ensemble.keys())[0]].atomic_numbers]
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace_data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=float(ensemble[list(ensemble.keys())[0]].r_max),
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    ensemble_metrics = {}
    for tag, model in ensemble.items():
        metrics = test_model(
            model=model,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
            return_predictions=return_predictions,
        )
        ensemble_metrics[tag] = metrics

    avg_ensemble_metrics = {}
    for key in ensemble_metrics[list(ensemble_metrics.keys())[0]].keys():
        if key not in ["mode", "epoch", "predictions"]:
            avg_ensemble_metrics[key] = np.mean(
                [m[key] for m in ensemble_metrics.values()]
            )
        if return_predictions:
            avg_ensemble_metrics["predictions"] = {
                key: np.mean(
                    [m["predictions"][key] for m in ensemble_metrics.values()],
                    axis=0,
                )
                for key in ensemble_metrics[list(ensemble_metrics.keys())[0]][
                    "predictions"
                ].keys()
            }
    if logger is not None:
        logger.log(avg_ensemble_metrics)
        if log_errors == "PerAtomRMSE":
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f}"
                "meV / A"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and avg_ensemble_metrics["rmse_stress_per_atom"] is not None
        ):
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_stress = avg_ensemble_metrics["rmse_stress_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} "
                f"meV / A, RMSE_stress_per_atom={error_stress:.1f} meV / A^3"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and avg_ensemble_metrics["rmse_virials_per_atom"] is not None
        ):
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_virials = avg_ensemble_metrics["rmse_virials_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f}"
                f"meV / A, RMSE_virials_per_atom={error_virials:.1f} meV"
            )
        elif log_errors == "TotalRMSE":
            error_e = avg_ensemble_metrics["rmse_e"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            logging.info(
                f"RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "PerAtomMAE":
            error_e = avg_ensemble_metrics["mae_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["mae_f"] * 1e3
            logging.info(
                f"MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f}"
                "meV / A"
            )
        elif log_errors == "TotalMAE":
            error_e = avg_ensemble_metrics["mae_e"] * 1e3
            error_f = avg_ensemble_metrics["mae_f"] * 1e3
            logging.info(
                f"MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "DipoleRMSE":
            error_mu = avg_ensemble_metrics["rmse_mu_per_atom"] * 1e3
            logging.info(f"RMSE_MU_per_atom={error_mu:.2f} mDebye")
        elif log_errors == "EnergyDipoleRMSE":
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_mu = avg_ensemble_metrics["rmse_mu_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f}"
                f"meV / A, RMSE_Mu_per_atom={error_mu:.2f} mDebye"
            )

    return (avg_ensemble_metrics, ensemble_metrics)

