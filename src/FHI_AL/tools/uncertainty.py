import numpy as np
import logging

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
            ensemble_prediction (np.array): Ensemble prediction of forces: [n_ensemble_members, n_mols, n_atoms, xyz].

        Returns:
            np.array: Standard deviation of atomic forces per molecule: [n_mols].
        """
        # average prediction over ensemble of models
        pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
        diff_sq = (ensemble_prediction - pred_av) ** 2.
        diff_sq_mean = np.mean(diff_sq, axis=(0,-1))
        sd = np.sqrt(diff_sq_mean)
        return sd
    
    def max_atomic_sd(
            self,
            ensemble_sd_atomic: np.array,
    ):
        """
        Compute the maximum atomic standard deviation of the ensemble prediction.

        Args:
            ensemble_sd_atomic (np.array): Atomic standard deviation of forces: [n_mols, n_atoms].

        Returns:
            np.array: Maximum atomic standard deviation of atomic forces per molecule: [n_mols].
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
            ensemble_sd_atomic (np.array): Atomic standard deviation of forces: [n_mols, n_atoms].

        Returns:
            np.array: Mean atomic standard deviation of atomic forces per molecule: [n_mols].
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
    def __init__(
            self,
            mol_idxs,
            uncertainty_type
            ):
        super().__init__(uncertainty_type)
        self.mol_idxs = mol_idxs
        self.uncertainty_type = uncertainty_type
        if uncertainty_type == "max_atomic_sd":
            self.uncertainty = self.max_atomic_sd
        elif uncertainty_type == "mean_atomic_sd":
            self.uncertainty = self.mean_atomic_sd

        #[n_ensemble_members, n_mols, n_atoms, xyz]
        #[n_ensemble_members, n_mols, xzy]
    def get_global_uncertainty(
            self,
            ensemble_prediction: np.array,):
        #[n_mols]
        self.global_uncerstainty = self.uncertainty(
            self.ensemble_sd(ensemble_prediction)
            )
        #[n_mols, 1]
        self.global_uncerstainty = np.array(self.global_uncerstainty).reshape(-1,1)
        
    def compute_mol_forces(
            self,
            ensemble_prediction: np.array,
            select_idxs
    )-> np.array:
        
        if ensemble_prediction.ndim == 4:
            self.mol_forces = np.empty(
                (
                ensemble_prediction.shape[0],
                ensemble_prediction.shape[1],
                len(select_idxs),
                3
                )
            )
        elif ensemble_prediction.ndim == 3:
            self.mol_forces = np.empty(
                (
                ensemble_prediction.shape[0],
                len(select_idxs),
                3
                )
            )
        
        for idx, mol in enumerate(select_idxs):
            if ensemble_prediction.ndim == 4:
                per_mol = ensemble_prediction[:,:,mol,:].sum(axis=-2)
                self.mol_forces[:,:,idx,:] = per_mol
            elif ensemble_prediction.ndim == 3:
                per_mol = ensemble_prediction[:,mol,:].sum(axis=-2)
                self.mol_forces[:,idx,:] = per_mol
            else:
                raise ValueError("Unexpected number of dimensions in ensemble_prediction")
        return self.mol_forces
               
    def get_intermol_uncertainty(
            self,
            ensemble_prediction: np.array
            ):
        # [n_ensemble_members, n_mols, len(select_idxs), xyz]
        self.compute_mol_forces(ensemble_prediction, self.mol_idxs)
        # [n_mols, len(select_idxs)] 
        self.inter_mol_uncertainty = self.ensemble_sd(
            self.mol_forces
            ).reshape(-1, len(self.mol_idxs))
        return self.inter_mol_uncertainty.squeeze()
        
        
    def __call__(self, ensemble_prediction):
        self.get_global_uncertainty(ensemble_prediction)
        self.get_intermol_uncertainty(ensemble_prediction)

        combined_uncertainty = np.concatenate(
            (self.global_uncerstainty, self.inter_mol_uncertainty),
            axis=-1
        )
        return combined_uncertainty.squeeze()
        
def get_threshold(
        uncertainties: np.array,
        c_x: float = 0.,
        max_len: int = 400
):
    uncertainties = np.array(uncertainties)[-max_len:]
    avg_uncertainty = np.mean(uncertainties, axis=0)
    threshold = avg_uncertainty * (1.0 + c_x)    
    return threshold
    
