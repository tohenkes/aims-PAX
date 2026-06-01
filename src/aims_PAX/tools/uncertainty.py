import numpy as np
from so3krates_torch.calculator.so3 import MultiHeadSO3LRCalculator


class HandleUncertainty:
    def __init__(self, uncertainty_type):
        self.uncertainty_type = uncertainty_type

    def ensemble_sd(
        self,
        ensemble_prediction: np.array,
    ) -> np.array:
        """
        Compute the standard deviation of the ensemble prediction.

        Args:
            ensemble_prediction (np.array): Ensemble prediction of forces:
                            [n_ensemble_members, n_mols, n_atoms, xyz].

        Returns:
            np.array: Standard deviation of atomic forces per molecule:
                                            [n_mols].
        """
        # average prediction over ensemble of models
        pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
        diff_sq = (ensemble_prediction - pred_av) ** 2.0
        diff_sq_mean = np.mean(diff_sq, axis=(0, -1))
        sd = np.sqrt(diff_sq_mean)
        return sd

    def max_atomic_sd(
        self,
        ensemble_sd_atomic: np.array,
    ):
        """
        Compute the maximum atomic standard deviation of the
        ensemble prediction.

        Args:
            ensemble_sd_atomic (np.array): Atomic standard deviation of forces:
                                            [n_mols, n_atoms].

        Returns:
            np.array: Maximum atomic standard deviation of atomic forces per
                            molecule: [n_mols].
        """
        max_sd = np.max(ensemble_sd_atomic, axis=-1)
        return max_sd

    def mean_atomic_sd(
        self,
        ensemble_sd_atomic: np.array,
    ):
        """
        Compute the mean atomic standard deviation of the ensemble prediction.

        Args:
            ensemble_sd_atomic (np.array): Atomic standard deviation of forces:
                                            [n_mols, n_atoms].

        Returns:
            np.array: Mean atomic standard deviation of atomic forces per
                                    molecule: [n_mols].
        """
        mean_sd = np.mean(ensemble_sd_atomic, axis=-1)
        return mean_sd

    def __call__(self, ensemble_prediction):
        ensemble_sd_atomic = self.ensemble_sd(ensemble_prediction)
        if self.uncertainty_type == "ensemble_sd":
            return ensemble_sd_atomic
        elif self.uncertainty_type == "max_atomic_sd":
            return self.max_atomic_sd(ensemble_sd_atomic)
        elif self.uncertainty_type == "mean_atomic_sd":
            return self.mean_atomic_sd(ensemble_sd_atomic)
        else:
            raise ValueError("Uncertainty type not recognized.")


class MolForceUncertainty(HandleUncertainty):
    def __init__(self, mol_idxs, uncertainty_type):
        super().__init__(uncertainty_type)
        self.mol_idxs = mol_idxs
        self.uncertainty_type = uncertainty_type
        if uncertainty_type == "max_atomic_sd":
            self.uncertainty = self.max_atomic_sd
        elif uncertainty_type == "mean_atomic_sd":
            self.uncertainty = self.mean_atomic_sd

        # [n_ensemble_members, n_mols, n_atoms, xyz]
        # [n_ensemble_members, n_mols, xzy]

    def get_global_uncertainty(
        self,
        ensemble_prediction: np.array,
    ):
        # [n_mols]
        self.global_uncerstainty = self.uncertainty(
            self.ensemble_sd(ensemble_prediction)
        )
        # [n_mols, 1]
        self.global_uncerstainty = np.array(self.global_uncerstainty).reshape(
            -1, 1
        )

    def compute_mol_forces_ensemble(
        self, ensemble_prediction: np.array, select_idxs
    ) -> np.array:

        if ensemble_prediction.ndim == 4:
            mol_forces = np.empty(
                (
                    ensemble_prediction.shape[0],
                    ensemble_prediction.shape[1],
                    len(select_idxs),
                    3,
                )
            )
        elif ensemble_prediction.ndim == 3:
            mol_forces = np.empty(
                (ensemble_prediction.shape[0], len(select_idxs), 3)
            )

        for idx, mol in enumerate(select_idxs):
            if ensemble_prediction.ndim == 4:
                per_mol = ensemble_prediction[:, :, mol, :].sum(axis=-2)
                mol_forces[:, :, idx, :] = per_mol
            elif ensemble_prediction.ndim == 3:
                per_mol = ensemble_prediction[:, mol, :].sum(axis=-2)
                mol_forces[:, idx, :] = per_mol
            else:
                raise ValueError(
                    "Unexpected number of dimensions in ensemble_prediction"
                )
        return mol_forces

    def get_intermol_uncertainty(self, ensemble_prediction: np.array):
        # [n_ensemble_members, n_mols, len(select_idxs), xyz]
        self.mol_forces = self.compute_mol_forces_ensemble(
            ensemble_prediction, self.mol_idxs
        )
        # [n_mols, len(select_idxs)]
        self.inter_mol_uncertainty = self.ensemble_sd(self.mol_forces).reshape(
            -1, len(self.mol_idxs)
        )
        return self.inter_mol_uncertainty.squeeze()

    def __call__(self, ensemble_prediction):
        self.get_global_uncertainty(ensemble_prediction)
        self.get_intermol_uncertainty(ensemble_prediction)

        combined_uncertainty = np.concatenate(
            (self.global_uncerstainty, self.inter_mol_uncertainty), axis=-1
        )
        return combined_uncertainty.squeeze()


class UDDCalculator(MultiHeadSO3LRCalculator):

    def __init__(
            self,
            A: float,
            B: float = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.A = A
        self.B = B

    def _apply_bias_potential_linear(
            self,
            num_atoms: int
    ):
        energy_deviation_ = self.results["energies"] - self.results["energy"]
        sigma_E_2 = 0.5 * np.sum(
            energy_deviation_ ** 2.0
        )
        
        E_bias = -self.A * sigma_E_2 / (self.num_heads * num_atoms)

        self.results["energy_bias"] = E_bias
        self.results["energy"] += E_bias

        force_deviation = self.results["forces_comm"] - self.results["forces"][None, :, :]
        f_bias = -self.A / (self.num_heads * num_atoms) * np.sum(
                energy_deviation_[:, :, None] * force_deviation,
                axis=0
            )
        self.results["forces_bias"] = f_bias
        self.results["forces"] += f_bias

    def _process_results(self, ret_tensors, out, multi_output=False, num_atoms=None):
        super()._process_results(ret_tensors, out, multi_output)
        self._apply_bias_potential_linear(num_atoms)



def get_threshold(
    uncertainties: np.array, c_x: float = 0.0, max_len: int = 400
) -> np.array:
    """
    Computes the threshold for active learning by
    computing the average of past uncertainties and
    scales it by a factor 1 + c_x.
    I.e. using c_x < 0 will result in a tighter threshold,
    while c_x > 0 will result in a looser threshold.

    Args:
        uncertainties (np.array): History of uncertainties.
        c_x (float, optional): Scaling factor. Defaults to 0.0.
        max_len (int, optional): How many past uncertainties are
                used to compute the threshold. Defaults to 400.

    Returns:
        np.array: Threhshold for active learning.
    """
    uncertainties = np.array(uncertainties)[-max_len:]
    avg_uncertainty = np.mean(uncertainties, axis=0)
    threshold = avg_uncertainty * (1.0 + c_x)
    return threshold
