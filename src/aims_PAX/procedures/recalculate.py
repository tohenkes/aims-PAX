from mace import tools
from aims_PAX.tools.utilities.utilities import AIMSControlParser
from ase.io import read, write
import logging
from yaml import safe_load
from aims_PAX.tools.utilities.parsl_utils import (
    recalc_dft_parsl,
    prepare_parsl,
    handle_parsl_logger,
)
import shutil
import time

try:
    import parsl
except ImportError:
    parsl = None


class ReCalculatorPARSL:
    def __init__(
        self,
        path_to_data: str,
        basis_dir: str = None,
        path_to_control: str = "./control.in",
        path_to_settings: str = "./aimsPAX.yaml",
        start_idx: int = 0,
        end_idx: int = None,
        save_interval: int = 10,
        properties: list = ["energy", "forces"],
    ):
        if parsl is None:
            raise ImportError(
                "Parsl is not installed. Please install parsl"
                " to use this feature."
            )
        # load settings
        try:
            with open(path_to_settings, "r") as file:
                al_settings = safe_load(file)
                self.cluster_settings = al_settings.get(
                    "CLUSTER",
                    None
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find {path_to_settings}. "
                "Please provide a valid path."
            )
        if self.cluster_settings is None:
            raise ValueError(
                "No cluster settings found in the provided settings file."
            )
            
        if path_to_control is None:
            raise FileNotFoundError("No control.in provided.")

        logging.basicConfig(
            filename="recalculate.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level="INFO",
            #    tag=tag,
            directory=".",
        )
        self.save_interval = save_interval
        self.control_parser = AIMSControlParser()

        if basis_dir is not None:
            self.basis_dir = basis_dir
        else:
            try:
                with open("./aimsPAX.yaml", "r") as file:
                    al_settings = safe_load(file)
                    self.basis_dir = al_settings["ACTIVE_LEARNING"][
                        "species_dir"
                    ]
            except FileNotFoundError:
                logging.error(
                    "No basis_dir provided or found in active_learning_settings.yaml."
                )

        self._handle_aims_settings(path_to_control)
        idx_str = (
            f"{start_idx}:" if end_idx is None else f"{start_idx}:{end_idx}"
        )

        logging.info(f"Loading data from {path_to_data}.")
        self.data = read(path_to_data, index=idx_str)
        logging.info(f"Found {len(self.data)} points in the dataset.")
        logging.info(f"Found cell: {self.data[0].get_cell()}")
        logging.info("Setting up calculator.")
        logging.info("Setting up PARSL for recalculation.")
        
        parsl_setup_dict = prepare_parsl(
            cluster_settings=self.cluster_settings
        )
        self.config = parsl_setup_dict["config"]
        self.calc_dir = parsl_setup_dict["calc_dir"]
        self.clean_dirs = parsl_setup_dict["clean_dirs"]
        self.launch_str = parsl_setup_dict["launch_str"]
        self.calc_idx = parsl_setup_dict["calc_idx"]
        handle_parsl_logger(
            log_dir="./parsl_recalculation.log",
        )
        parsl.load(self.config)
        self.properties = properties

    def __call__(self):
        """
        Recalculates the dataset using PARSL.
        """
        recalculated_points = []
        job_results = {}
        calcs_done = 0
        for i, atoms in enumerate(self.data):
            self.calc_idx += 1
            temp = recalc_dft_parsl(
                positions=atoms.get_positions(),
                species=atoms.get_chemical_symbols(),
                cell=atoms.get_cell(),
                pbc=atoms.get_pbc(),
                aims_settings=self.aims_settings,
                directory=self.calc_dir / f"calc_{self.calc_idx}",
                properties=self.properties,
                ase_aims_command=self.launch_str,
            )
            job_results[i] = temp
        while len(job_results) > 0:
            for i in list(job_results.keys()):
                result = job_results[i]
                if result.done():
                    temp = result.result()
                    calcs_done += 1
                    if temp is None:
                        logging.warning(
                            f"SCF not converged for point {i}. Skipping."
                        )
                        del job_results[i]
                        continue
                    current_point = self.data[i].copy()
                    current_point.info["energy"] = temp["energy"]
                    current_point.arrays["forces"] = temp["forces"]
                    if 'stress' in self.properties:
                        current_point.info["stress"] = temp["stress"]
                    recalculated_points.append(current_point)
                    del job_results[i]
                    if calcs_done % self.save_interval == 0:
                        logging.info(
                            f"Recalculated point {i+1} with reference method."
                        )
                        write(
                            "recalculated_data_parsl.xyz",
                            recalculated_points,
                            append=True
                        )
                        del recalculated_points[:]
                        
            time.sleep(0.5)
        if len(recalculated_points) > 0:
            write("recalculated_data_parsl.xyz", recalculated_points, append=True)
        if self.clean_dirs:
            try:
                for calc_dir in self.calc_dir.glob("calc_*"):
                    shutil.rmtree(calc_dir)
            except Exception as e:
                logging.error(
                    f"Error while cleaning directories: {e}. "
                    "Please check the directories manually."
                )

        logging.info("Closing PARSL.")
        parsl.dfk().cleanup()
        exit()

    def _handle_aims_settings(self, path_to_control: str):
        """
        Parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """

        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings["species_dir"] = self.basis_dir
        self.aims_settings["compute_forces"] = True