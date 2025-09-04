import logging
import os
from typing import Union
from .input_checks import check_aimsPAX_settings, check_MACE_settings
from ase.io import read
from yaml import safe_load
import ase


def read_input_files(
    path_to_mace_settings: str = "./mace.yaml",
    path_to_aimsPAX_settings: str = "./aimsPAX.yaml",
    procedure: str = "full",
) -> tuple:
    """
    Reads the input files for MACE and aimsPAX settings.
    Checks the settings and returns the checked settings
    and paths to control and geometry files.

    Args:
        path_to_mace_settings (str, optional): Path to the MACE settings file.
                                        Defaults to "./mace.yaml".
        path_to_aimsPAX_settings (str, optional): Path to the AIMS PAX
                                settings file. Defaults to "./aimsPAX.yaml".
        procedure (str, optional): The procedure for which the settings are
            checked. Can be "initial-ds", "al" or "full". Defaults to "full".

    Returns:
        tuple: A tuple containing the checked MACE settings, checked AIMS PAX
            settings, path to control file, and path to geometry file.

    Returns:
        tuple: A tuple containing the checked MACE settings, checked AIMS PAX
        settings, path to control file, and path to geometry file.
    """
    with open(path_to_mace_settings, "r") as file:
        mace_settings = safe_load(file)
    with open(path_to_aimsPAX_settings, "r") as file:
        aimsPAX_settings = safe_load(file)

    aimsPAX_settings = check_aimsPAX_settings(
        aimsPAX_settings, procedure=procedure
    )

    mace_settings = check_MACE_settings(mace_settings)

    path_to_control = aimsPAX_settings["MISC"].get(
        "path_to_control", "./control.in"
    )
    path_to_geometry = aimsPAX_settings["MISC"].get(
        "path_to_geometry", "./geometry.in"
    )

    return (
        mace_settings,
        aimsPAX_settings,
        path_to_control,
        path_to_geometry,
    )


def read_geometry(
    geometry_source: Union[str, dict[int, str]],
    log: bool = False
) -> dict[int, ase.Atoms]:
    """
    Reads geometry data from various sources and returns a dictionary of ASE Atoms objects.
    
    Args:
        geometry_source (Union[str, dict]): Either a path to a file/directory or a 
            dictionary mapping integers to file paths.
            - If string and directory: reads all ASE-readable files in the directory
            - If string and file: reads the single file
            - If dict: reads files specified by the dictionary values
        log (bool, optional): Whether to log information about loaded geometries. 
            Defaults to False.

    Returns:
        Dict[int, ase.Atoms]: Dictionary mapping integer indices to ASE Atoms objects.

    Raises:
        ValueError: If no valid ASE readable files are found in a directory.
        TypeError: If geometry_source is neither string nor dictionary, or if 
            dictionary values are not strings or keys are not integers.
        Exception: If individual files cannot be read by ASE.
    """

    if isinstance(geometry_source, str):
        if os.path.isdir(geometry_source):
            atoms_dict = {}
            for i, filename in enumerate(sorted(os.listdir(geometry_source))):
                if log:
                    logging.info(
                        f"Geometry {i}: {filename.split('.')[0]} is at index {i}."
                    )
                complete_path = os.path.join(geometry_source, filename)
                if os.path.isfile(complete_path):
                    try:
                        atoms = read(complete_path)
                        atoms_dict[i] = atoms
                    except Exception as e:
                        logging.error(
                            f"File {filename} is not a valid ASE readable file: {e}"
                        )
            if not atoms_dict:
                raise ValueError("No valid ASE readable files found.")
            return atoms_dict
        else:
            atoms_dict = {0: read(geometry_source)}

    elif isinstance(geometry_source, dict):
        atoms_dict = {}
        for key, value in geometry_source.items():
            assert type(value) is str, "All entries in the geometry dictionary must be strings."
            assert type(key) is int, "All keys in the geometry dictionary must be integers."
            try:
                atoms = read(value)
                atoms_dict[key] = atoms
                if log:
                    logging.info(
                        f"Geometry {key}: {value.split('.')[0]} is at index {key}."
                    )
            except Exception as e:
                logging.error(f"Failed to read geometry file {value}: {e}")
    else:
        raise TypeError("The geometry source must be a string or a dictionary.")
    
    return atoms_dict

