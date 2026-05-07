"""Command line utilities for working with datasets."""
import argparse
import random
import logging
import warnings

from ase.io import read

from aims_PAX.atomate2.cli import get_args_parser
from aims_PAX.tools.utilities.data_handling import save_datasets
from aims_PAX.tools.utilities.input_utils import read_input_files
from aims_PAX.tools.utilities.utilities import get_seeds, create_seeds_tags_dict

# silence torch jit warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def to_chunks(lst,
              n,
              dropout_args: dict[str, bool | float],
              to_keep: list[str]):
    """
    Divides a list into n approximately equal chunks.

    This function takes a list and splits it into n smaller lists (chunks),
    where the sizes of the chunks are as equal as possible. If the length
    of the list is not perfectly divisible by n, the larger chunks are
    given to the earlier positions in the resulting list of chunks.

    Args:
        dropout_args: a dictionary containing dropout settings.
        lst: The list to be divided into chunks.
        n: The number of chunks to divide the list into.
        to_keep: A list of configuration types to keep in the dataset.

    Returns:
        A list of n chunks, each being a sublist of the input list. The
        size of the chunks will differ by at most one.

    Raises:
        ValueError: If n is less than or equal to 0.
    """
    if dropout_args["dropout_on"]:
        configs_to_keep = [a for a in lst if a.info.get('config_type', '') in to_keep]
        configs_to_drop = [a for a in lst if a.info.get('config_type', '') not in to_keep]
        if configs_to_keep:
            logger.info(f"Keeping {len(configs_to_keep)} configs of type {to_keep}")
        chunk_len = int(len(lst) * (1 - dropout_args["dropout_ratio"]) - len(configs_to_keep))
        chunks = [random.choices(configs_to_drop, k=chunk_len) + configs_to_keep for _ in range(n)]
    else:
        chunks = []
        k, remainder = divmod(len(lst), n)
        start = 0
        for i in range(n):
            end = start + k + (1 if i < remainder else 0)
            chunks.append(lst[start:end])
            start = end
    return chunks


def split_by_ratio(lst, ratio):
    split_idx = int(len(lst) * ratio)
    return lst[:split_idx], lst[split_idx:]


def split_dataset(
        path_to_dataset: str,
        dropout_args: dict[str, bool | float],
        to_keep: list[str],
        path_to_aimspax_settings: str,
        path_to_model_settings: str
):
    """
    Splits a dataset into multiple training and validation subsets based on ensemble
    size and other configuration settings provided. This function organizes and
    shuffles the data, applies a split ratio, assigns appropriate tags, and saves
    the resulting subsets in the defined directories.

    Args:
        path_to_dataset (str): Path to the original dataset, formatted as "extxyz".
        dropout_args (dict[str, bool | float]): Dictionary specifying dropout settings.
        to_keep (list[str]): A list of configuration types to keep in the dataset.
        path_to_aimspax_settings (str): Path to the AIMSPAX configuration settings file.
        path_to_model_settings (str): Path to the model configuration settings file.

    Raises:
        FileNotFoundError: If any of the provided paths to settings or dataset is invalid.
        KeyError: If the required keys are missing in the configuration settings.
        ValueError: If the dataset cannot be read or if invalid split ratios are provided.
        Exception: For any other error that may arise during dataset processing or file IO.

    Returns:
        None
    """

    # prepare directories
    model_settings, project_settings, _, _ = read_input_files(
        path_to_model_settings,
        path_to_aimspax_settings,
        procedure="full",
    )
    # create seeds and related tags
    ensemble_size = project_settings.INITIAL_DATASET_GENERATION.ensemble_size
    logger.info(f"Splitting dataset {path_to_dataset} into {ensemble_size} chunks")
    ensemble_seeds = get_seeds(model_settings.GENERAL.seed,
                               ensemble_size)
    seeds_tags_dict = create_seeds_tags_dict(
        seeds=ensemble_seeds,
        model_settings=model_settings,
        dataset_dir=project_settings.MISC.dataset_dir,
    )
    tags: list[str] = list(seeds_tags_dict.keys())
    logger.info(f"Tags used: {tags}")
    # read and shuffle dataset
    if dropout_args["dropout_on"]:
        logger.info(f"Using random dropout with ratio {dropout_args['dropout_ratio']}.")

    dataset = list(read(path_to_dataset, format="extxyz", index=":"))
    random.shuffle(dataset)
    # split dataset using ensemble size
    logger.info(f"Full dataset length: {len(dataset)}.")
    chunks = to_chunks(dataset, ensemble_size, dropout_args, to_keep)
    # split each chunk into training and validation sets
    ase_sets = {}
    logger.info(f"Test / Train ratio used: {project_settings.INITIAL_DATASET_GENERATION.valid_ratio}")
    for tag, chunk in zip(tags, chunks):
        logger.info(f" - Tag: {tag}")
        logger.info(f"   Chunk size: {len(chunk)}")
        test, train = split_by_ratio(chunk, project_settings.INITIAL_DATASET_GENERATION.valid_ratio)
        logger.info(f"   Train size: {len(train)}")
        logger.info(f"   Test size: {len(test)}")
        ase_sets[tag] = {"train": train, "valid": test}
    # in the next line only tags are needed from the ensemble
    # so we can mimic ensemble dict
    save_datasets(
        seeds_tags_dict,
        ase_sets,
        project_settings.MISC.dataset_dir / "initial",
        initial=True,
        save_combined_initial=True
    )
    logger.info(f"Datasets saved successfully at {project_settings.MISC.dataset_dir / 'initial'}.")


def main():
    """The main function for the command line execution."""
    parser = get_args_parser(description="Aims-PAX split datasets")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset XYZ file to split"
    )
    parser.add_argument(
        "--dropout",
        action="store_true",
        default=False,
        help="Whether to split datasets with random dropout"
    )
    parser.add_argument(
        "--dropout-ratio",
        type=validated_ratio,
        default=0.1,
        help="Random dropout ratio. Should be between 0 and 1.",
    )
    parser.add_argument(
        "--keep-isolated-atoms",
        action="store_true",
        help="Whether to keep isolated atoms (should have `info.config_type = 'IsolatedAtom'`) in all datasets"
    )
    parser.add_argument(
        "--keep-dimers",
        action="store_true",
        help="Whether to keep dimers (should have `info.config_type = 'dimer'`) in all datasets"
    )
    args = parser.parse_args()
    to_keep = []
    if args.keep_isolated_atoms:
        to_keep.append("IsolatedAtom")
    if args.keep_dimers:
        to_keep.append("dimer")
    dropout = {
        "dropout_on": args.dropout,
        "dropout_ratio": args.dropout_ratio
    }
    split_dataset(args.dataset, dropout, to_keep, args.aimsPAX_settings, args.model_settings)


def validated_ratio(value: str) -> float:
    """Validator for a dropout ratio."""
    ratio = float(value)
    if not 0 < ratio < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return ratio

if __name__ == "__main__":
    main()