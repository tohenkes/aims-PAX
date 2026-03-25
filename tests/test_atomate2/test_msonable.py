"""
This module contains tests for the `msonable` module.
"""

import torch
import numpy as np
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator

from mace.modules.models import MACE
from aims_PAX.atomate2.msonable.mace import MSONScaleShiftMACE


def test_msonable_mace(clean_dir, si, data_dir):
    """Test MSONable MACE potential"""

    def get_data_from_model(model: MACE, s: Atoms):
        """Returns energy and forces calculated from the mace model using ASE"""
        s.calc =  MACECalculator(
            models=[model],
            device="cpu")
        # Get energy and forces from the original model
        return {
            "energy": s.get_potential_energy(),
            "forces": s.get_forces()
        }

    # Load model
    model_path = data_dir / "models" / "Si.model"
    original_model = torch.load(model_path, map_location="cpu", weights_only=False)
    original_model.to("cpu")

    # Load structure, then get energy and forces from the original model
    atoms = read(si)
    data_orig = get_data_from_model(original_model, atoms)
    # Make the model serializable, then serialize and deserialize it
    msonable_model = MSONScaleShiftMACE.from_parent(original_model)
    model_dict = MSONScaleShiftMACE.as_dict(msonable_model)
    deserialized_model = MSONScaleShiftMACE.from_dict(model_dict)
    deserialized_model.to("cpu")

    # Get energy and forces from the deserialized model
    data_deserialized = get_data_from_model(original_model, atoms)

    # Check if energy and forces are the same
    np.testing.assert_allclose(data_orig["energy"], data_deserialized["energy"], atol=1e-6)
    np.testing.assert_allclose(data_orig["forces"], data_deserialized["forces"], atol=1e-6)
