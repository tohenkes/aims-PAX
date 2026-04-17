"""Command line utilities for working with datasets."""
import argparse
import random

import numpy as np
from ase.io import read

from aims_PAX.atomate2.cli import get_args_parser
from aims_PAX.tools.utilities.data_handling import save_datasets
from aims_PAX.tools.utilities.input_utils import read_input_files
from aims_PAX.tools.utilities.utilities import get_seeds, create_seeds_tags_dict

def to_chunks(lst, n):
    """
    Divides a list into n approximately equal chunks.

    This function takes a list and splits it into n smaller lists (chunks),
    where the sizes of the chunks are as equal as possible. If the length
    of the list is not perfectly divisible by n, the larger chunks are
    given to the earlier positions in the resulting list of chunks.

    Args:
        lst: The list to be divided into chunks.
        n: The number of chunks to divide the list into.

    Returns:
        A list of n chunks, each being a sublist of the input list. The
        size of the chunks will differ by at most one.

    Raises:
        ValueError: If n is less than or equal to 0.
    """
    k, remainder = divmod(len(lst), n)
    start = 0
    chunks = []
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
        path_to_aimspax_settings: str,
        path_to_model_settings: str
):
    # prepare directories
    model_settings, project_settings, _, _ = read_input_files(
        path_to_model_settings,
        path_to_aimspax_settings,
        procedure="full",
    )
    # create seeds and related tags
    ensemble_size = model_settings.GENERAL.ensemble_size
    ensemble_seeds = get_seeds(model_settings.GENERAL.seed,
                               ensemble_size)
    seeds_tags_dict = create_seeds_tags_dict(
        seeds=ensemble_seeds,
        model_settings=model_settings,
        dataset_dir=project_settings.MISC.dataset_dir,
    )
    tags: list[str] = list(seeds_tags_dict.keys())
    # read and shuffle dataset
    dataset = list(read(path_to_dataset, format="extxyz", index=":"))
    random.shuffle(dataset)
    # split dataset using ensemble size
    chunks = to_chunks(dataset, ensemble_size)
    # split each chunk into training and validation sets
    ase_sets = {}
    for tag, chunk in zip(tags, chunks):
        test, train = split_by_ratio(chunks[0], project_settings.INITIAL_DATASET_GENERATION.valid_ratio)
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


def main():
    """The main function for the command line execution."""
    parser = get_args_parser(description="Aims-PAX split datasets")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset XYZ file to split"
    )
    args = parser.parse_args()
    split_dataset(args.dataset, args.aimsPAX_settings, args.model_settings)
    
if __name__ == "__main__":
    main()